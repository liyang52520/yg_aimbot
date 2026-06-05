import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any

import cv2
import mss
import numpy as np
from screeninfo import get_monitors

from core.services.config_service import config_service
from core.ui_bridge import ui_bridge as image_signal

logger = logging.getLogger(__name__)


class ScreenCaptureService:
    """屏幕捕获服务 — mss (GDI) 后端"""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 锁保护帧数据（竞争极低：主线程 ~0.05ms 一次，截取线程 ~6ms 一次）
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id: Optional[int] = None

        self._monitor: dict = {}
        self._circle_mask_3ch: Optional[np.ndarray] = None
        self._last_config = None
        self._ready = False
        self._fps = config_service.get('capture', 'fps', 160)
        self._interval = 1.0 / self._fps

        # 帧率统计（环形缓冲区，每秒更新一次显示）
        self._capture_times = deque(maxlen=60)
        self._fps_update_interval = 0.1
        self._last_fps_update = 0

        # 帧ID生成（雪花算法）
        self._last_second = -1
        self._frame_id_counter = 0

        self._init_monitor()
        self._last_config = config_service.get_section('capture').copy()
        self._ready = True
        config_service.register_callback(self._on_config_change)

    def _on_config_change(self, section: str, updates: Dict[str, Any]):
        if not self._ready or section != 'capture':
            return
        cfg = config_service.get_section('capture')
        w = cfg.get('window_width', 320)
        h = cfg.get('window_height', 320)
        fps = cfg.get('fps', 160)

        if w != self._monitor.get('width') or h != self._monitor.get('height'):
            self._init_monitor()
            self._circle_mask_3ch = None
            logger.info(f"监控区域已更新: {w}x{h}")

        if fps != self._fps:
            self._fps = fps
            self._interval = 1.0 / self._fps
            logger.info(f"FPS已更新: {fps}")

        self._last_config = cfg.copy()

    def _init_monitor(self):
        try:
            cfg = config_service.get_section('capture')
            width = cfg.get('window_width', 640)
            height = cfg.get('window_height', 640)
            for monitor in get_monitors():
                if monitor.is_primary:
                    sw, sh = monitor.width, monitor.height
                    left = (sw - width) // 2
                    top = (sh - height) // 2
                    self._monitor = {'left': left, 'top': top, 'width': width, 'height': height}
                    logger.info(f"监控区域初始化: {width}x{height} @ ({left}, {top})")
                    break
        except Exception as e:
            logger.error(f"初始化监控区域失败: {e}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("屏幕捕获服务已启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("屏幕捕获服务已停止")

    def _generate_frame_id(self) -> int:
        t = int(time.time())
        if t != self._last_second:
            self._last_second = t
            self._frame_id_counter = 0
        else:
            self._frame_id_counter += 1
        return (t << 20) | self._frame_id_counter

    def get_frame(self) -> Optional[tuple[np.ndarray, int]]:
        """获取一帧（锁保护复制，安全无竞态）"""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy(), self._frame_id
            return None, None

    def get_resolution(self) -> Tuple[int, int]:
        return self._monitor.get('width', 640), self._monitor.get('height', 640)

    def _capture_loop(self):
        sct = None
        try:
            sct = mss.mss()
            last_time = time.perf_counter()

            while self._running:
                now = time.perf_counter()
                if now - last_time >= self._interval:
                    try:
                        raw = sct.grab(self._monitor)
                        # BGRA → BGR view（零拷贝）
                        img = np.frombuffer(raw.bgra, np.uint8).reshape(
                            (raw.height, raw.width, 4))[:, :, :3]
                        self._process_frame(img)
                        self._capture_times.append(now)
                        self._update_fps(now)
                        last_time += self._interval
                    except Exception as e:
                        logger.error(f"捕获帧错误: {e}")
                else:
                    # 自适应休眠：剩余时间 > 1ms 时睡长些，否则短轮
                    remaining = self._interval - (now - last_time)
                    if remaining > 0.002:
                        time.sleep(remaining - 0.001)
                    else:
                        time.sleep(0.0005)
        except Exception as e:
            logger.error(f"捕获循环异常: {e}")
        finally:
            if sct:
                try:
                    sct.close()
                except Exception:
                    pass

    def _update_fps(self, current_time: float):
        if current_time - self._last_fps_update < self._fps_update_interval:
            return
        n = len(self._capture_times)
        if n >= 2:
            d = self._capture_times[-1] - self._capture_times[0]
            if d > 0:
                image_signal.emit_fps(capture_fps=(n - 1) / d)
        self._last_fps_update = current_time

    def _process_frame(self, frame: np.ndarray):
        try:
            cfg = config_service.get_section('capture')
            if cfg.get('circle', False):
                frame = self._apply_circle_mask(frame)
            fid = self._generate_frame_id()

            with self._lock:
                self._frame = frame.copy()
                self._frame_id = fid

            if cfg.get('ai_debug', False):
                image_signal.emit_video_frame(frame)
        except Exception as e:
            logger.error(f"处理帧失败: {e}")

    def _apply_circle_mask(self, frame: np.ndarray) -> np.ndarray:
        """3通道预计算圆形遮罩（避免逐帧 cv2.merge 开销）"""
        h, w = frame.shape[:2]
        if self._circle_mask_3ch is None or self._circle_mask_3ch.shape[:2] != (h, w):
            cx, cy = w // 2, h // 2
            r = min(w, h) // 2
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            self._circle_mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(frame, self._circle_mask_3ch)

    def check_config_change(self) -> bool:
        current = config_service.get_section('capture')
        if self._last_config is None:
            self._last_config = current.copy()
            return False
        if current != self._last_config:
            self._last_config = current.copy()
            self._init_monitor()
            self._circle_mask_3ch = None
            return True
        return False


capture_service = ScreenCaptureService()
