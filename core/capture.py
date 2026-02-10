import logging
import queue
import threading
import time

import cv2
import mss
import numpy as np
from screeninfo import get_monitors

from core.config import cfg

logger = logging.getLogger(__name__)


class Capture(threading.Thread):
    """
    屏幕捕获类
    使用mss库捕获屏幕画面，支持帧率控制和圆形捕获
    """

    def __init__(self):
        super().__init__()
        self.daemon = True
        self.name = "Capture"

        self.frame_queue = queue.Queue(maxsize=1)
        self.sct = None
        self.running = True
        self.last_config_check_time = time.time()
        self.last_capture_window_width = None
        self.last_capture_window_height = None

        # 初始化配置
        self.update_config()

    def run(self):
        """线程运行方法"""
        self.sct = mss.mss()
        last_frame_time = time.time()

        try:
            while self.running:
                current_time = time.time()

                # 定期检查配置变更（每秒一次）
                if current_time - self.last_config_check_time >= 1.0:
                    self.update_config()
                    self.last_config_check_time = current_time

                # 动态获取帧率配置
                target_fps = cfg.capture_fps
                frame_interval = 1.0 / target_fps

                # 控制帧率
                if current_time - last_frame_time >= frame_interval:
                    frame = self.capture_frame()
                    if frame is not None:
                        # 处理图像
                        if cfg.capture_circle:
                            frame = self.convert_to_circle(frame)

                        # 放入队列
                        if self.frame_queue.full():
                            self.frame_queue.get()
                        self.frame_queue.put(frame, block=False)
                        last_frame_time = current_time
                else:
                    # 短暂休眠
                    time.sleep(0.0005)
        finally:
            if self.sct:
                self.sct.close()

    def capture_frame(self):
        """捕获一帧屏幕"""
        screenshot = self.sct.grab(self.monitor)
        img = np.frombuffer(screenshot.bgra, np.uint8).reshape(
            (screenshot.height, screenshot.width, 4)
        )
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def get_new_frame(self):
        """获取新帧"""
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def _calculate_mss_offset(self):
        """计算mss偏移"""
        x, y = self.get_primary_display_resolution()
        left = x / 2 - cfg.capture_window_width / 2
        top = y / 2 - cfg.capture_window_height / 2
        return int(left), int(top), int(cfg.capture_window_width), int(cfg.capture_window_height)

    def get_primary_display_resolution(self):
        """获取主显示器分辨率"""
        for monitor in get_monitors():
            if monitor.is_primary:
                return monitor.width, monitor.height
        return 1920, 1080  # 默认值

    def convert_to_circle(self, image):
        """转换为圆形图像"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = min(width, height) // 2

        # 创建掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        # 应用掩码
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def update_config(self):
        """更新配置，特别是捕获窗口大小"""
        # 首次运行时初始化last_capture_window_width和last_capture_window_height
        if self.last_capture_window_width is None or self.last_capture_window_height is None:
            self.last_capture_window_width = cfg.capture_window_width
            self.last_capture_window_height = cfg.capture_window_height
            self.screen_x_center = int(cfg.capture_window_width / 2)
            self.screen_y_center = int(cfg.capture_window_height / 2)
            left, top, w, h = self._calculate_mss_offset()
            self.monitor = {"left": left, "top": top, "width": w, "height": h}
            logger.info(f"捕获窗口配置已初始化: {w}x{h}")
            return

        # 检查捕获窗口大小是否变更
        if (cfg.capture_window_width != self.last_capture_window_width or
                cfg.capture_window_height != self.last_capture_window_height):

            # 更新屏幕中心坐标
            self.screen_x_center = int(cfg.capture_window_width / 2)
            self.screen_y_center = int(cfg.capture_window_height / 2)

            # 重新计算监控区域
            left, top, w, h = self._calculate_mss_offset()
            self.monitor = {"left": left, "top": top, "width": w, "height": h}

            # 重新初始化mss对象以确保新的捕获区域生效
            if self.sct:
                self.sct.close()
                self.sct = mss.mss()
                logger.info("MSS对象已重新初始化以应用新的捕获窗口大小")

            # 更新上次配置值
            self.last_capture_window_width = cfg.capture_window_width
            self.last_capture_window_height = cfg.capture_window_height

            logger.info(f"捕获窗口配置已更新: {w}x{h}")


# 创建全局实例
capture = Capture()
capture.start()
