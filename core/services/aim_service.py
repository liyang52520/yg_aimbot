import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

from makcu import create_controller

from core.services.config_service import config_service

logger = logging.getLogger(__name__)

MOVE_QUEUE_MAXSIZE = 10
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
        self._move_queue: queue.Queue[Tuple[float, float]] = queue.Queue(maxsize=MOVE_QUEUE_MAXSIZE)
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

    def process_target(self, x: float, y: float, w: float, h: float):
        """处理目标"""
        move_x, move_y = self._calculate_movement(x, y, w, h)

        if self._should_move(move_x, move_y):
            self._execute_movement(move_x, move_y)

    def _calculate_movement(self, target_x: float, target_y: float,
                            target_w: float, target_h: float) -> Tuple[float, float]:
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

        smoothed_x = self._config.smooth_factor * predicted_x + self._inv_smooth_factor * self._state.offset_x
        smoothed_y = self._config.smooth_factor * predicted_y + self._inv_smooth_factor * self._state.offset_y

        self._state.offset_x = smoothed_x
        self._state.offset_y = smoothed_y

        move_x = (smoothed_x * self._degrees_per_pixel_x / 360.0) * self._dpi_factor
        move_y = (smoothed_y * self._degrees_per_pixel_y / 360.0) * self._dpi_factor

        if distance < target_w * self._config.tremor_distance_threshold:
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

    def _should_move(self, move_x: float, move_y: float) -> bool:
        """判断是否需要移动"""
        min_move_sq = self._config.min_move ** 2
        return move_x ** 2 > min_move_sq or move_y ** 2 > min_move_sq

    def _execute_movement(self, x: float, y: float):
        """执行移动"""
        try:
            self._move_queue.put_nowait((x, y))
        except queue.Full:
            pass

    def _move_worker(self):
        """移动工作线程"""
        while self._move_thread_running:
            try:
                move_x, move_y = self._move_queue.get(timeout=0.1)
                ix, iy = int(move_x), int(move_y)
                MouseControlService.move(ix, iy)
                self._move_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"移动线程错误: {e}")

    def stop(self):
        """停止服务"""
        self._move_thread_running = False
        if self._move_thread.is_alive():
            self._move_thread.join(timeout=1.0)
        logger.info("瞄准服务已停止")


aim_service = AimService()
mouse_service = MouseControlService
