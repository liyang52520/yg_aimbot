import logging
import threading
import time
import ctypes
from collections import deque

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

        # 使用deque代替Queue，提高性能
        self.frame_queue = deque(maxlen=3)
        self.sct = None
        self.running = True
        self.last_config_check_time = time.time()
        self.last_capture_window_width = None
        self.last_capture_window_height = None
        self.config_check_interval = 0.2  # 进一步降低配置检查频率
        
        # 预计算的掩码，避免重复计算
        self.circle_mask = None
        self.circle_mask_3ch = None
        
        # 缓存的屏幕中心坐标
        self.screen_x_center = 0
        self.screen_y_center = 0
        
        # 缓存的显示器分辨率
        self.display_width = 1920
        self.display_height = 1080

        # 初始化配置
        self.update_config()

    def run(self):
        """线程运行方法"""
        self.sct = mss.mss()
        last_frame_time = time.time()
        frame_interval = 1.0 / cfg.capture_fps
        config_check_timer = time.time()

        try:
            while self.running:
                current_time = time.time()

                # 定期检查配置变更
                if current_time - config_check_timer >= self.config_check_interval:
                    self.update_config()
                    config_check_timer = current_time
                    # 动态更新帧率配置
                    frame_interval = 1.0 / cfg.capture_fps

                # 控制帧率
                if current_time - last_frame_time >= frame_interval:
                    frame = self.capture_frame()
                    if frame is not None:
                        # 处理图像
                        if cfg.capture_circle:
                            frame = self.convert_to_circle(frame)

                        # 放入队列
                        self.frame_queue.append(frame)
                        last_frame_time = current_time
                else:
                    # 使用更精确的休眠时间，避免CPU空转
                    sleep_time = max(0, frame_interval - (current_time - last_frame_time) - 0.00005)
                    if sleep_time > 0.0001:
                        time.sleep(sleep_time)
        finally:
            if self.sct:
                self.sct.close()

    def capture_frame(self):
        """捕获一帧屏幕"""
        try:
            screenshot = self.sct.grab(self.monitor)
            # 直接从内存缓冲区创建数组，避免额外复制
            # 使用更高效的内存布局
            img = np.frombuffer(screenshot.bgra, np.uint8)
            # 一次性完成reshape和通道提取，避免中间数组和copy
            return img.reshape((screenshot.height, screenshot.width, 4))[:, :, :3]
        except Exception as e:
            # 减少日志开销
            if cfg.capture_ai_debug:
                logger.error(f"捕获帧错误: {e}")
            return None

    def get_new_frame(self):
        """获取新帧"""
        try:
            return self.frame_queue.popleft()
        except IndexError:
            return None

    def _calculate_mss_offset(self):
        """计算mss偏移"""
        left = self.display_width // 2 - cfg.capture_window_width // 2
        top = self.display_height // 2 - cfg.capture_window_height // 2
        return int(left), int(top), int(cfg.capture_window_width), int(cfg.capture_window_height)

    def get_primary_display_resolution(self):
        """获取主显示器分辨率"""
        try:
            for monitor in get_monitors():
                if monitor.is_primary:
                    self.display_width = monitor.width
                    self.display_height = monitor.height
                    return self.display_width, self.display_height
        except Exception as e:
            if cfg.capture_ai_debug:
                logger.error(f"获取显示器分辨率错误: {e}")
        return self.display_width, self.display_height

    def convert_to_circle(self, image):
        """转换为圆形图像"""
        try:
            height, width = image.shape[:2]
            
            # 检查是否需要重新创建掩码
            if (self.circle_mask is None or 
                self.circle_mask.shape[0] != height or 
                self.circle_mask.shape[1] != width):
                center = (width // 2, height // 2)
                radius = min(width, height) // 2
                
                # 创建掩码
                self.circle_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(self.circle_mask, center, radius, 255, -1)
                
                # 创建三通道掩码
                self.circle_mask_3ch = cv2.merge([self.circle_mask, self.circle_mask, self.circle_mask])
            
            # 应用掩码
            return cv2.bitwise_and(image, self.circle_mask_3ch)
        except Exception as e:
            if cfg.capture_ai_debug:
                logger.error(f"转换为圆形图像错误: {e}")
            return image

    def update_config(self):
        """更新配置，特别是捕获窗口大小"""
        try:
            # 首次运行时初始化
            if self.last_capture_window_width is None or self.last_capture_window_height is None:
                self.last_capture_window_width = cfg.capture_window_width
                self.last_capture_window_height = cfg.capture_window_height
                self.screen_x_center = cfg.capture_window_width // 2
                self.screen_y_center = cfg.capture_window_height // 2
                self.get_primary_display_resolution()
                left, top, w, h = self._calculate_mss_offset()
                self.monitor = {"left": left, "top": top, "width": w, "height": h}
                # 重置掩码
                self.circle_mask = None
                if cfg.capture_ai_debug:
                    logger.info(f"捕获窗口配置已初始化: {w}x{h}")
                return

            # 检查捕获窗口大小是否变更
            if (cfg.capture_window_width != self.last_capture_window_width or
                    cfg.capture_window_height != self.last_capture_window_height):

                # 更新屏幕中心坐标
                self.screen_x_center = cfg.capture_window_width // 2
                self.screen_y_center = cfg.capture_window_height // 2

                # 重新计算监控区域
                left, top, w, h = self._calculate_mss_offset()
                self.monitor = {"left": left, "top": top, "width": w, "height": h}

                # 重置掩码
                self.circle_mask = None
                self.circle_mask_3ch = None

                # 重新初始化mss对象以确保新的捕获区域生效
                if self.sct:
                    self.sct.close()
                    self.sct = mss.mss()
                    if cfg.capture_ai_debug:
                        logger.info("MSS对象已重新初始化以应用新的捕获窗口大小")

                # 更新上次配置值
                self.last_capture_window_width = cfg.capture_window_width
                self.last_capture_window_height = cfg.capture_window_height

                if cfg.capture_ai_debug:
                    logger.info(f"捕获窗口配置已更新: {w}x{h}")
        except Exception as e:
            if cfg.capture_ai_debug:
                logger.error(f"更新配置错误: {e}")


# 创建全局实例
capture = Capture()
capture.start()
