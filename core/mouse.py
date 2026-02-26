import logging
import math
import queue
import threading
import time
from collections import deque
from functools import lru_cache

from core.config import cfg
from core.move.makcu_mouse import MakcuMouse

logger = logging.getLogger(__name__)


class MouseController:
    """
    鼠标控制器
    处理鼠标移动和瞄准逻辑
    """

    def __init__(self):
        self._initialize_parameters()
        self._initialize_thread()

    def _initialize_parameters(self):
        """初始化参数"""
        # 基础配置
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.screen_width = cfg.capture_window_width
        self.screen_height = cfg.capture_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2

        # 核心参数
        self.smooth_factor = 0.6

        # 状态变量
        self.current_offset_x = 0.0
        self.current_offset_y = 0.0
        self.target_history = deque(maxlen=3)

        # 微动参数
        self.tremor_amount = 0.02
        self.tremor_phase = 0.0

        # 限制参数
        self.max_move = 20
        self.min_move = 1
        
        # 模型输入大小（默认640，会在模型加载后更新）
        self.model_input_size = 640
        
        # 预计算常量
        self._cache_config()
    
    def _initialize_thread(self):
        """初始化鼠标移动线程"""
        self._move_queue = queue.Queue(maxsize=10)
        self._move_thread_running = True
        self._move_thread = threading.Thread(target=self._move_worker, daemon=True)
        self._move_thread.start()
        logger.info("鼠标移动线程已启动")
    
    def _cache_config(self):
        """缓存配置计算值，避免重复计算"""
        self._dpi_factor = cfg.mouse_dpi * (1.0 / cfg.mouse_sensitivity)
    
    def update_config_cache(self):
        """当配置变更时更新缓存"""
        self._cache_config()
        self.screen_width = cfg.capture_window_width
        self.screen_height = cfg.capture_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2

    def set_model_input_size(self, input_size):
        """设置模型输入大小
        
        Args:
            input_size: 模型输入大小（正方形）
        """
        self.model_input_size = input_size
        logger.info(f"鼠标控制器模型输入大小已更新: {self.model_input_size}x{self.model_input_size}")

    def process_data(self, data):
        """处理目标数据"""
        # 解析数据
        target_x, target_y, target_w, target_h, target_cls = self._parse_data(data)
        if target_x is None:
            return

        # 输入验证 - 使用更快的检查方式
        if not (target_w > 0 and target_h > 0 and 
                target_x == target_x and target_y == target_y):  # NaN检查
            return

        # 计算移动
        move_x, move_y = self._calculate_movement(target_x, target_y, target_w, target_h)

        # 执行移动 - 使用绝对值比较避免重复计算
        if move_x * move_x > self.min_move * self.min_move or \
           move_y * move_y > self.min_move * self.min_move:
            self._execute_movement(move_x, move_y)

    def _parse_data(self, data):
        """解析数据"""
        try:
            if hasattr(data, 'xyxy'):
                if data.xyxy.size > 0:
                    target_x, target_y = data.xyxy.mean(axis=1)[:2]
                    target_w = data.xyxy[0, 2] - data.xyxy[0, 0]
                    target_h = data.xyxy[0, 3] - data.xyxy[0, 1]
                    target_cls = data.class_id[0] if data.class_id.size > 0 else 0
                    return target_x, target_y, target_w, target_h, target_cls
                else:
                    return None, None, None, None, None
            else:
                return data
        except Exception as e:
            logger.error(f"Error parsing data: {e}")
            return None, None, None, None, None

    def _calculate_movement(self, target_x, target_y, target_w, target_h):
        """计算鼠标移动距离"""
        # 考虑捕获窗口大小和模型输入大小之间的比例关系
        # 当捕获窗口大小与模型输入大小不同时，需要调整目标坐标
        capture_to_model_ratio = self.model_input_size / max(cfg.capture_window_width, cfg.capture_window_height)

        # 使用模型输入大小的中心，这样无论捕获窗口大小如何变化，计算出的鼠标移动都是准确的
        center_x = self.model_input_size * 0.5
        center_y = self.model_input_size * 0.5

        # 调整目标坐标，考虑捕获窗口大小和模型输入大小之间的比例关系
        adjusted_target_x = target_x * capture_to_model_ratio
        adjusted_target_y = target_y * capture_to_model_ratio
        adjusted_target_w = target_w * capture_to_model_ratio

        # 计算偏移量
        offset_x = adjusted_target_x - center_x
        offset_y = adjusted_target_y - center_y
        
        # 使用平方距离避免开方运算
        distance_sq = offset_x * offset_x + offset_y * offset_y
        distance = math.sqrt(distance_sq) if distance_sq > 0 else 0

        # 记录目标历史
        current_time = time.time()
        self.target_history.append((adjusted_target_x, adjusted_target_y, current_time))

        # 计算目标速度
        target_vx, target_vy = self._calculate_target_velocity(current_time)

        # 预测目标位置
        prediction_time = 0.025
        predicted_offset_x = offset_x + target_vx * prediction_time
        predicted_offset_y = offset_y + target_vy * prediction_time

        # 平滑处理
        inv_smooth = 1.0 - self.smooth_factor
        self.current_offset_x = self.smooth_factor * predicted_offset_x + inv_smooth * self.current_offset_x
        self.current_offset_y = self.smooth_factor * predicted_offset_y + inv_smooth * self.current_offset_y

        # 计算角度，使用模型输入大小
        degrees_per_pixel_x = cfg.mouse_fov_width / self.model_input_size
        degrees_per_pixel_y = cfg.mouse_fov_height / self.model_input_size

        angle_x = self.current_offset_x * degrees_per_pixel_x
        angle_y = self.current_offset_y * degrees_per_pixel_y

        # 转换为鼠标移动距离 - 使用缓存的dpi因子
        move_x = (angle_x / 360.0) * self._dpi_factor
        move_y = (angle_y / 360.0) * self._dpi_factor

        # 添加微小抖动 - 只在距离较近时
        if distance < adjusted_target_w * 0.3:
            self.tremor_phase += 0.3
            tremor_scale = self.tremor_amount * (distance / adjusted_target_w)
            move_x += math.sin(self.tremor_phase) * tremor_scale
            move_y += math.cos(self.tremor_phase * 1.3) * tremor_scale

        # 限制最大移动距离 - 使用min/max避免条件分支
        move_x = max(-self.max_move, min(self.max_move, move_x))
        move_y = max(-self.max_move, min(self.max_move, move_y))

        return move_x, move_y

    def _calculate_target_velocity(self, current_time):
        """计算目标速度"""
        if len(self.target_history) < 2:
            return 0.0, 0.0

        prev = self.target_history[-2]
        dt = current_time - prev[2]
        if dt <= 0.001:
            return 0.0, 0.0

        inv_dt = 1.0 / dt
        target_vx = (self.target_history[-1][0] - prev[0]) * inv_dt
        target_vy = (self.target_history[-1][1] - prev[1]) * inv_dt
        return target_vx, target_vy

    def _move_worker(self):
        """鼠标移动工作线程"""
        while self._move_thread_running:
            try:
                move_x, move_y = self._move_queue.get(timeout=0.1)
                self._perform_move(move_x, move_y)
                self._move_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"鼠标移动线程错误: {e}")
        logger.info("鼠标移动线程已停止")

    def _perform_move(self, x, y):
        """实际执行鼠标移动"""
        ix, iy = int(x), int(y)
        if cfg.mouse_move == "makcu":
            MakcuMouse.move(ix, iy)
        else:
            logger.warning("Only support Makcu move!")

    def _execute_movement(self, x, y):
        """执行鼠标移动 - 将移动指令放入队列"""
        try:
            self._move_queue.put_nowait((x, y))
        except queue.Full:
            pass

    def stop(self):
        """停止鼠标移动线程"""
        self._move_thread_running = False
        if self._move_thread.is_alive():
            self._move_thread.join(timeout=1.0)
        logger.info("鼠标控制器已停止")


# 创建全局实例
mouse = MouseController()
