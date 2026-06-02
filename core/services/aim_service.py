import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

from makcu import create_controller

from core.services.config_service import config_service

logger = logging.getLogger(__name__)

MOVE_SLEEP_SEC = 0.001    # 1ms → 运动线程约 500-1000 Hz
MOVE_FRACTION = 0.15       # 每次 tick 逼近剩余距离的比例
TARGET_HISTORY_MAXLEN = 3


@dataclass
class MouseConfig:
    """鼠标配置"""
    dpi: float
    sensitivity: float
    fov_width: float
    fov_height: float
    window_width: int
    window_height: int
    smooth_factor: float = 0.6
    max_move: float = 20.0
    min_move: float = 1.0
    tremor_amount: float = 0.02
    prediction_time: float = 0.05
    tremor_phase_increment: float = 0.3
    tremor_phase_multiplier: float = 1.3
    tremor_distance_threshold: float = 0.3

    @property
    def center_x(self) -> float:
        return self.window_width / 2

    @property
    def center_y(self) -> float:
        return self.window_height / 2


@dataclass
class TargetState:
    """目标状态"""
    offset_x: float = 0.0
    offset_y: float = 0.0
    tremor_phase: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=TARGET_HISTORY_MAXLEN))


class MouseControlService:
    """鼠标控制服务"""

    _device = None
    _scope = 20
    _lock = None
    _initialized = False

    @classmethod
    def _get_lock(cls) -> threading.Lock:
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def get_device(cls):
        """获取设备"""
        if cls._device is None:
            with cls._get_lock():
                if cls._device is None:
                    try:
                        cls._device = create_controller(auto_reconnect=True)
                        cls._initialized = True
                        logger.info("鼠标控制器初始化成功")
                    except Exception as e:
                        logger.error(f"鼠标控制器初始化失败: {e}")
                        cls._device = create_controller(auto_reconnect=True)
                        cls._initialized = False
        return cls._device

    @classmethod
    def move(cls, x: int, y: int) -> bool:
        """移动鼠标"""
        try:
            x = max(-cls._scope, min(x, cls._scope))
            y = max(-cls._scope, min(y, cls._scope))

            device = cls.get_device()
            if device is None:
                logger.error("鼠标控制器未初始化")
                return False
            device.move(x, y)
            return True
        except Exception as e:
            logger.error(f"鼠标移动失败: {e}")
            cls._device = None
            return False

    @classmethod
    def is_initialized(cls) -> bool:
        """检查是否已初始化"""
        return cls._initialized

    @classmethod
    def reset(cls):
        """重置控制器"""
        with cls._get_lock():
            cls._device = None
            cls._initialized = False
            logger.info("鼠标控制器已重置")


class AimService:
    """瞄准服务"""

    def __init__(self):
        self._config = self._load_config()
        self._state = TargetState()
        self._model_input_size = 640
        # 连续运动控制状态
        self._target_remain_x = 0.0
        self._target_remain_y = 0.0
        self._move_lock = threading.Lock()
        self._frac_x = 0.0   # 亚像素累积器 X
        self._frac_y = 0.0   # 亚像素累积器 Y
        self._move_thread_running = True
        self._move_thread = threading.Thread(target=self._move_worker, daemon=True)
        self._move_thread.start()
        self._update_cache()
        config_service.register_callback(self._on_config_change)

    def _on_config_change(self, section: str, updates: Dict[str, Any]):
        """配置变更回调"""
        if section in ('aim', 'mouse', 'capture'):
            self.update_config()

    def _load_config(self) -> MouseConfig:
        """加载配置"""
        mouse_cfg = config_service.get_section('mouse')
        capture_cfg = config_service.get_section('capture')

        return MouseConfig(
            dpi=mouse_cfg.get('dpi', 1100),
            sensitivity=mouse_cfg.get('sensitivity', 3.0),
            fov_width=mouse_cfg.get('fov_width', 40),
            fov_height=mouse_cfg.get('fov_height', 40),
            window_width=capture_cfg.get('window_width', 320),
            window_height=capture_cfg.get('window_height', 320)
        )

    def _update_cache(self):
        """更新缓存"""
        self._dpi_factor = self._config.dpi / self._config.sensitivity
        self._capture_to_model_ratio = self._model_input_size / max(
            self._config.window_width, self._config.window_height
        )
        self._degrees_per_pixel_x = self._config.fov_width / self._model_input_size
        self._degrees_per_pixel_y = self._config.fov_height / self._model_input_size
        self._model_center = self._model_input_size * 0.5
        self._inv_smooth_factor = 1.0 - self._config.smooth_factor

    def update_config(self):
        """更新配置"""
        self._config = self._load_config()
        self._update_cache()
        logger.debug("瞄准服务配置已更新")

    def set_model_input_size(self, input_size: int):
        """设置模型输入大小"""
        self._model_input_size = input_size
        self._update_cache()
        logger.info(f"模型输入大小已更新: {input_size}x{input_size}")

    def process_target(self, x: float, y: float, w: float, h: float, is_predicted: bool = False):
        """处理目标 - 计算移动量并设置为连续运动目标"""
        move_x, move_y = self._calculate_movement(x, y, w, h, is_predicted=is_predicted)
        with self._move_lock:
            self._target_remain_x = move_x
            self._target_remain_y = move_y

    def _calculate_movement(self, target_x: float, target_y: float,
                            target_w: float, target_h: float,
                            is_predicted: bool = False) -> Tuple[float, float]:
        """计算移动"""
        aim_cfg = config_service.get_section('aim')

        adjusted_x = target_x + aim_cfg.get('body_x_offset', 0.1) * target_w * 0.5
        adjusted_y = target_y + aim_cfg.get('body_y_offset', 0.1) * target_h * 0.5

        ratio = self._capture_to_model_ratio
        model_x, model_y = adjusted_x * ratio, adjusted_y * ratio

        offset_x, offset_y = model_x - self._model_center, model_y - self._model_center
        distance = math.sqrt(offset_x ** 2 + offset_y ** 2)

        self._state.history.append((offset_x, offset_y, time.time()))
        velocity_x, velocity_y = self._calculate_velocity()

        predicted_x = offset_x + velocity_x * self._config.prediction_time
        predicted_y = offset_y + velocity_y * self._config.prediction_time

        # 预测模式下降低平滑因子，更快响应真实检测的修正
        smooth_factor = self._config.smooth_factor
        if is_predicted:
            smooth_factor = max(0.3, smooth_factor * 0.7)

        inv_smooth_factor = 1.0 - smooth_factor
        smoothed_x = smooth_factor * predicted_x + inv_smooth_factor * self._state.offset_x
        smoothed_y = smooth_factor * predicted_y + inv_smooth_factor * self._state.offset_y

        self._state.offset_x = smoothed_x
        self._state.offset_y = smoothed_y

        move_x = (smoothed_x * self._degrees_per_pixel_x / 360.0) * self._dpi_factor
        move_y = (smoothed_y * self._degrees_per_pixel_y / 360.0) * self._dpi_factor

        # 预测模式下不添加抖动，避免不确定性叠加
        if not is_predicted and distance < target_w * self._config.tremor_distance_threshold:
            move_x, move_y = self._add_tremor(move_x, move_y, distance, target_w)

        return self._clamp(move_x), self._clamp(move_y)

    def _calculate_velocity(self) -> Tuple[float, float]:
        """计算速度"""
        if len(self._state.history) < 2:
            return 0.0, 0.0

        prev = self._state.history[-2]
        current = self._state.history[-1]
        dt = current[2] - prev[2]

        if dt <= 0.001:
            return 0.0, 0.0

        return (current[0] - prev[0]) / dt, (current[1] - prev[1]) / dt

    def _add_tremor(self, x: float, y: float, distance: float, width: float) -> Tuple[float, float]:
        """添加抖动"""
        phase = self._state.tremor_phase
        self._state.tremor_phase += self._config.tremor_phase_increment

        scale = self._config.tremor_amount * (distance / width)
        tremor_x = math.sin(phase) * scale
        tremor_y = math.cos(phase * self._config.tremor_phase_multiplier) * scale

        return x + tremor_x, y + tremor_y

    def _clamp(self, value: float) -> float:
        """限制范围"""
        return max(-self._config.max_move, min(self._config.max_move, value))

    def _move_worker(self):
        """连续高频率运动工作线程（~500-1000Hz）"""
        while self._move_thread_running:
            try:
                with self._move_lock:
                    rx = self._target_remain_x
                    ry = self._target_remain_y

                # 无有效移动 → 长休眠减少 CPU 占用
                if abs(rx) < self._config.min_move and abs(ry) < self._config.min_move:
                    time.sleep(0.002)
                    continue

                # 指数逼近：每次 tick 移动剩余距离的固定比例
                step_x = rx * MOVE_FRACTION
                step_y = ry * MOVE_FRACTION

                step_x = self._clamp(step_x)
                step_y = self._clamp(step_y)

                # 亚像素累积：避免 int() 截断导致的精度丢失
                self._frac_x += step_x
                self._frac_y += step_y
                ix = int(self._frac_x)
                iy = int(self._frac_y)
                self._frac_x -= ix
                self._frac_y -= iy

                if ix != 0 or iy != 0:
                    MouseControlService.move(ix, iy)

                # 更新剩余目标（锁保护，防止与 process_target 竞争）
                with self._move_lock:
                    self._target_remain_x -= step_x
                    self._target_remain_y -= step_y
                    # 防过冲：符号变化意味着走过头了
                    if self._target_remain_x * rx < 0:
                        self._target_remain_x = 0.0
                    if self._target_remain_y * ry < 0:
                        self._target_remain_y = 0.0

                time.sleep(MOVE_SLEEP_SEC)
            except Exception as e:
                logger.error(f"连续运动线程错误: {e}")
                time.sleep(0.005)

    def stop(self):
        """停止服务"""
        self._move_thread_running = False
        if self._move_thread.is_alive():
            self._move_thread.join(timeout=1.0)
        logger.info("瞄准服务已停止")


aim_service = AimService()
mouse_service = MouseControlService
