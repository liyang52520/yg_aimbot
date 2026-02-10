import logging
import math
import time
from collections import deque

from core.config import cfg
from core.move.makcu import Makcu
from core.move.viGEmBus import ViGEmBus

logger = logging.getLogger(__name__)


class MouseController:
    """
    鼠标控制器
    处理鼠标移动和瞄准逻辑
    """

    def __init__(self):
        self._initialize_parameters()

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
        self.smooth_factor = 0.5

        # 状态变量
        self.current_offset_x = 0.0
        self.current_offset_y = 0.0
        self.target_history = deque(maxlen=3)

        # 微动参数
        self.tremor_amount = 0.02
        self.tremor_phase = 0.0

        # 限制参数
        self.max_move = 40
        self.min_move = 0.8

    def process_data(self, data):
        """处理目标数据"""
        # 解析数据
        target_x, target_y, target_w, target_h, target_cls = self._parse_data(data)
        if target_x is None:
            return

        # 输入验证
        if any(map(math.isnan, (target_x, target_y, target_w, target_h))):
            return

        if target_w <= 0 or target_h <= 0:
            return

        # 计算移动
        move_x, move_y = self._calculate_movement(target_x, target_y, target_w, target_h)

        # 执行移动
        if abs(move_x) > self.min_move or abs(move_y) > self.min_move:
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
        # 使用最新的配置值，确保捕获窗口大小变更时能正确计算
        from core.config import cfg

        # 考虑捕获窗口大小和模型输入大小之间的比例关系
        # 当捕获窗口大小与模型输入大小不同时，需要调整目标坐标
        capture_to_model_ratio = cfg.ai_model_image_size / max(cfg.capture_window_width, cfg.capture_window_height)

        # 使用模型输入大小的中心，这样无论捕获窗口大小如何变化，计算出的鼠标移动都是准确的
        center_x = cfg.ai_model_image_size / 2
        center_y = cfg.ai_model_image_size / 2

        # 调整目标坐标，考虑捕获窗口大小和模型输入大小之间的比例关系
        adjusted_target_x = target_x * capture_to_model_ratio
        adjusted_target_y = target_y * capture_to_model_ratio
        adjusted_target_w = target_w * capture_to_model_ratio

        # 计算偏移量
        offset_x = adjusted_target_x - center_x
        offset_y = adjusted_target_y - center_y
        distance = math.sqrt(offset_x ** 2 + offset_y ** 2)

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
        self.current_offset_x = self.smooth_factor * predicted_offset_x + (
                1 - self.smooth_factor) * self.current_offset_x
        self.current_offset_y = self.smooth_factor * predicted_offset_y + (
                1 - self.smooth_factor) * self.current_offset_y

        # 计算角度，使用模型输入大小，这样无论捕获窗口大小如何变化，计算出的鼠标移动都是准确的
        degrees_per_pixel_x = cfg.mouse_fov_width / cfg.ai_model_image_size
        degrees_per_pixel_y = cfg.mouse_fov_height / cfg.ai_model_image_size

        angle_x = self.current_offset_x * degrees_per_pixel_x
        angle_y = self.current_offset_y * degrees_per_pixel_y

        # 转换为鼠标移动距离
        move_x = (angle_x / 360) * (cfg.mouse_dpi * (1 / cfg.mouse_sensitivity))
        move_y = (angle_y / 360) * (cfg.mouse_dpi * (1 / cfg.mouse_sensitivity))

        # 添加微小抖动
        if distance < adjusted_target_w * 0.3:
            self.tremor_phase += 0.3
            tremor_x = math.sin(self.tremor_phase) * self.tremor_amount * (distance / adjusted_target_w)
            tremor_y = math.cos(self.tremor_phase * 1.3) * self.tremor_amount * (distance / adjusted_target_w)
            move_x += tremor_x
            move_y += tremor_y

        # 限制最大移动距离
        move_x = max(-self.max_move, min(self.max_move, move_x))
        move_y = max(-self.max_move, min(self.max_move, move_y))

        return move_x, move_y

    def _calculate_target_velocity(self, current_time):
        """计算目标速度"""
        if len(self.target_history) < 2:
            return 0, 0

        prev = self.target_history[-2]
        dt = current_time - prev[2]
        if dt <= 0.001:
            return 0, 0

        target_vx = (self.target_history[-1][0] - prev[0]) / dt
        target_vy = (self.target_history[-1][1] - prev[1]) / dt
        return target_vx, target_vy

    def _execute_movement(self, x, y):
        """执行鼠标移动"""
        ix, iy = int(x), int(y)

        if cfg.mouse_move == "makcu":
            Makcu.move(ix, iy)
        elif cfg.mouse_move == "vigembus":
            ViGEmBus.move(ix, iy, cfg.viGEmBus_move_scope, cfg.viGEmBus_move_sleep)
        else:
            logger.warning("Only support Makcu move and ViGEmBus move!")


# 创建全局实例
mouse = MouseController()
