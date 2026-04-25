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
    """屏幕捕获服务"""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_id: Optional[int] = None
        self._lock = threading.Lock()
        self._monitor: dict = {}
        self._circle_mask: Optional[np.ndarray] = None
        self._last_config = None
        self._ready = False
        self._fps = config_service.get('capture', 'fps', 60)
        self._interval = 1.0 / self._fps
        
        # 帧率统计
        self._capture_times = deque(maxlen=30)
        self._fps_update_interval = 0.03
        self._last_fps_update = 0
        
        # 帧ID生成
        self._last_second = -1
        self._frame_id_counter = 0
        
        self._init_monitor()
        self._last_config = config_service.get_section('capture').copy()
        self._ready = True
        config_service.register_callback(self._on_config_change)

    def _on_config_change(self, section: str, updates: Dict[str, Any]):
        """配置变更回调"""
        if not self._ready:
            return

        if section == 'capture':
            current_config = config_service.get_section('capture')

            new_width = current_config.get('window_width', 320)
            new_height = current_config.get('window_height', 320)
            new_fps = current_config.get('fps', 60)

            current_monitor_width = self._monitor.get('width', 0)
            current_monitor_height = self._monitor.get('height', 0)

            logger.debug(
                f"配置变更: 新的宽高={new_width}x{new_height}, 当前监控={current_monitor_width}x{current_monitor_height}, 新的FPS={new_fps}")

            if new_width != current_monitor_width or new_height != current_monitor_height:
                self._init_monitor()
                self._circle_mask = None
                logger.info(f"监控区域已更新: {new_width}x{new_height}")

            if new_fps != self._fps:
                self._fps = new_fps
                self._interval = 1.0 / self._fps
                logger.info(f"FPS已更新: {new_fps}")
            else:
                logger.debug("窗口尺寸和FPS未变化，跳过更新")

            self._last_config = current_config.copy()

    def _init_monitor(self):
        """初始化监控区域"""
        try:
            config = config_service.get_section('capture')
            width = config.get('window_width', 320)
            height = config.get('window_height', 320)

            for monitor in get_monitors():
                if monitor.is_primary:
                    screen_w, screen_h = monitor.width, monitor.height
                    left = (screen_w - width) // 2
                    top = (screen_h - height) // 2
                    self._monitor = {'left': left, 'top': top, 'width': width, 'height': height}
                    logger.info(f"监控区域初始化: {width}x{height} @ ({left}, {top})")
                    break
        except Exception as e:
            logger.error(f"初始化监控区域失败: {e}")

    def start(self):
        """启动捕获"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("屏幕捕获服务已启动")

    def stop(self):
        """停止捕获"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("屏幕捕获服务已停止")

    def _generate_frame_id(self) -> int:
        """生成帧ID，基于时间的雪花算法"""
        current_time = time.time()
        current_second = int(current_time)
        
        # 如果是新的秒，重置计数器
        if current_second != self._last_second:
            self._last_second = current_second
            self._frame_id_counter = 0
        else:
            self._frame_id_counter += 1
        
        # 生成帧ID：秒数左移20位 + 计数器
        frame_id = (current_second << 20) | self._frame_id_counter
        return frame_id

    def get_frame(self) -> Optional[tuple[np.ndarray, int]]:
        """获取一帧图像和帧ID"""
        with self._lock:
            if self._frame is not None and self._frame_id is not None:
                return self._frame.copy(), self._frame_id
            return None, None

    def get_resolution(self) -> Tuple[int, int]:
        """获取捕获分辨率"""
        return self._monitor.get('width', 320), self._monitor.get('height', 320)

    def _capture_loop(self):
        """捕获循环"""
        sct = None
        try:
            sct = mss.mss()
            last_time = time.time()

            while self._running:
                current_time = time.time()
                elapsed = current_time - last_time

                if elapsed >= self._interval:
                    try:
                        frame = self._capture_frame(sct)
                        if frame is not None:
                            self._process_frame(frame)
                            # 记录捕获时间
                            self._capture_times.append(current_time)
                            # 更新FPS
                            self._update_fps(current_time)
                        # 使用累加_interval的方式更新last_time，确保帧率稳定
                        last_time += self._interval
                    except Exception as e:
                        logger.error(f"捕获帧错误: {e}")
                else:
                    sleep_time = max(0, self._interval - elapsed - 0.001)
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"捕获循环异常: {e}")
        finally:
            if sct:
                try:
                    sct.close()
                except:
                    pass
    
    def _update_fps(self, current_time: float):
        """更新捕获帧率"""
        if current_time - self._last_fps_update < self._fps_update_interval:
            return

        if len(self._capture_times) >= 2:
            diff = self._capture_times[-1] - self._capture_times[0]
            if diff > 0:
                fps = (len(self._capture_times) - 1) / diff
                image_signal.emit_fps(capture_fps=fps)
        else:
            image_signal.emit_fps(capture_fps=0.0)

        self._last_fps_update = current_time

    def _capture_frame(self, sct: mss.mss) -> Optional[np.ndarray]:
        """捕获一帧"""
        try:
            screenshot = sct.grab(self._monitor)
            img = np.frombuffer(screenshot.bgra, np.uint8)
            return img.reshape((screenshot.height, screenshot.width, 4))[:, :, :3]
        except Exception as e:
            logger.error(f"捕获帧失败: {e}")
            return None

    def _process_frame(self, frame: np.ndarray):
        """处理帧"""
        try:
            config = config_service.get_section('capture')

            if config.get('circle', False):
                frame = self._apply_circle_mask(frame)

            frame_id = self._generate_frame_id()
            with self._lock:
                self._frame = frame
                self._frame_id = frame_id
            
            # 如果启用了AI调试/视频监控，发送帧到UI
            if config.get('ai_debug', False):
                image_signal.emit_video_frame(frame)
                
        except Exception as e:
            logger.error(f"处理帧失败: {e}")

    def _apply_circle_mask(self, frame: np.ndarray) -> np.ndarray:
        """应用圆形掩码"""
        height, width = frame.shape[:2]

        if self._circle_mask is None or self._circle_mask.shape != (height, width):
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 2

            self._circle_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(self._circle_mask, (center_x, center_y), radius, 255, -1)

        mask_3ch = cv2.merge([self._circle_mask] * 3)
        return cv2.bitwise_and(frame, mask_3ch)

    def check_config_change(self) -> bool:
        """检查配置是否变更"""
        current_config = config_service.get_section('capture')

        if self._last_config is None:
            self._last_config = current_config.copy()
            return False

        has_change = False
        for key in current_config:
            if current_config[key] != self._last_config.get(key):
                has_change = True
                break

        if has_change:
            self._last_config = current_config.copy()
            self._init_monitor()
            self._circle_mask = None
            return True
        return False


capture_service = ScreenCaptureService()