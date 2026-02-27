import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from makcu import create_controller

from core.config import cfg

logger = logging.getLogger(__name__)

DEFAULT_SCOPE = 20
DEFAULT_MODEL_INPUT_SIZE = 640
DEFAULT_SMOOTH_FACTOR = 0.6
DEFAULT_MAX_MOVE = 20.0
DEFAULT_MIN_MOVE = 1.0
DEFAULT_TREMOR_AMOUNT = 0.02
DEFAULT_PREDICTION_TIME = 0.05
DEFAULT_TREMOR_PHASE_INCREMENT = 0.3
DEFAULT_TREMOR_PHASE_MULTIPLIER = 1.3
DEFAULT_TREMOR_DISTANCE_THRESHOLD = 0.3
MOVE_QUEUE_MAXSIZE = 10
TARGET_HISTORY_MAXLEN = 3


class MakcuMouse:
    """Makcu鼠标控制器封装类"""

    _device = None
    _scope = DEFAULT_SCOPE
    _lock = None
    _initialized = False

    @classmethod
    def _get_lock(cls) -> threading.Lock:
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def get_device(cls):
        if cls._device is None:
            with cls._get_lock():
                if cls._device is None:
                    try:
                        cls._device = create_controller(auto_reconnect=True)
                        cls._initialized = True
                        logger.info("Makcu鼠标控制器初始化成功")
                    except Exception as e:
                        logger.error(f"Makcu鼠标控制器初始化失败: {e}")
                        cls._device = create_controller(auto_reconnect=True)
                        cls._initialized = False
        return cls._device

    @classmethod
    def move(cls, x: int, y: int) -> bool:
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
        return cls._initialized

    @classmethod
    def reset(cls):
        with cls._get_lock():
            cls._device = None
            cls._initialized = False
            logger.info("Makcu鼠标控制器已重置")


@dataclass
class MouseConfig:
    """鼠标配置"""
    dpi: float
    sensitivity: float
    fov_width: float
    fov_height: float
    window_width: int
    window_height: int
    smooth_factor: float = DEFAULT_SMOOTH_FACTOR
    max_move: float = DEFAULT_MAX_MOVE
    min_move: float = DEFAULT_MIN_MOVE
    tremor_amount: float = DEFAULT_TREMOR_AMOUNT
    prediction_time: float = DEFAULT_PREDICTION_TIME
    tremor_phase_increment: float = DEFAULT_TREMOR_PHASE_INCREMENT
    tremor_phase_multiplier: float = DEFAULT_TREMOR_PHASE_MULTIPLIER
    tremor_distance_threshold: float = DEFAULT_TREMOR_DISTANCE_THRESHOLD

    @property
    def center_x(self) -> float:
        return self.window_width / 2

    @property
    def center_y(self) -> float:
        return self.window_height / 2


@dataclass
class CachedCalculations:
    """缓存计算结果"""
    dpi_factor: float = 0.0
    capture_to_model_ratio: float = 0.0
    degrees_per_pixel_x: float = 0.0
    degrees_per_pixel_y: float = 0.0
    model_center: float = 0.0
    inv_smooth_factor: float = 0.0


@dataclass
class TargetState:
    """目标状态"""
    offset_x: float = 0.0
    offset_y: float = 0.0
    tremor_phase: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=TARGET_HISTORY_MAXLEN))


class MouseController:
    """鼠标控制器"""

    def __init__(self):
        self._config = self._create_config()
        self._cached = CachedCalculations()
        self._state = TargetState()
        self._model_input_size = DEFAULT_MODEL_INPUT_SIZE
        self._move_queue: queue.Queue[Tuple[float, float]] = queue.Queue(maxsize=MOVE_QUEUE_MAXSIZE)
        self._move_thread_running = True
        self._move_thread = threading.Thread(target=self._move_worker, daemon=True)
        self._move_thread.start()
        self._update_cache()
        logger.info("鼠标控制器已初始化")

    def _create_config(self) -> MouseConfig:
        return MouseConfig(
            dpi=cfg.mouse_dpi,
            sensitivity=cfg.mouse_sensitivity,
            fov_width=cfg.mouse_fov_width,
            fov_height=cfg.mouse_fov_height,
            window_width=cfg.capture_window_width,
            window_height=cfg.capture_window_height
        )

    def _update_cache(self):
        self._cached.dpi_factor = self._config.dpi / self._config.sensitivity
        self._cached.capture_to_model_ratio = self._model_input_size / max(
            self._config.window_width, self._config.window_height
        )
        self._cached.degrees_per_pixel_x = self._config.fov_width / self._model_input_size
        self._cached.degrees_per_pixel_y = self._config.fov_height / self._model_input_size
        self._cached.model_center = self._model_input_size * 0.5
        self._cached.inv_smooth_factor = 1.0 - self._config.smooth_factor

    def update_config(self):
        self._config = self._create_config()
        self._update_cache()
        logger.debug("配置已更新")

    def set_model_input_size(self, input_size: int):
        self._model_input_size = input_size
        self._update_cache()
        logger.info(f"模型输入大小已更新: {input_size}x{input_size}")

    def process_data(self, data):
        parsed = self._parse_data(data)
        if not self._validate_data(parsed):
            return

        target_x, target_y, target_w, target_h, _ = parsed
        move_x, move_y = self._calculate_movement(target_x, target_y, target_w, target_h)

        if self._should_move(move_x, move_y):
            self._execute_movement(move_x, move_y)

    def _parse_data(self, data) -> Optional[Tuple[float, float, float, float, int]]:
        try:
            if hasattr(data, 'xyxy'):
                if data.xyxy.size > 0:
                    target_x, target_y = data.xyxy.mean(axis=1)[:2]
                    target_w = data.xyxy[0, 2] - data.xyxy[0, 0]
                    target_h = data.xyxy[0, 3] - data.xyxy[0, 1]
                    target_cls = data.class_id[0] if data.class_id.size > 0 else 0
                    return target_x, target_y, target_w, target_h, target_cls
                return None
            if isinstance(data, tuple) and len(data) == 5:
                return data
            return None
        except Exception as e:
            logger.error(f"数据解析错误: {e}")
            return None

    def _validate_data(self, data: Optional[Tuple]) -> bool:
        if data is None:
            return False
        target_x, target_y, target_w, target_h, _ = data
        return (target_w > 0 and target_h > 0 and
                target_x == target_x and target_y == target_y)

    def _should_move(self, move_x: float, move_y: float) -> bool:
        min_move_sq = self._config.min_move ** 2
        return move_x * move_x > min_move_sq or move_y * move_y > min_move_sq

    def _calculate_movement(self, target_x: float, target_y: float,
                            target_w: float, target_h: float) -> Tuple[float, float]:
        adjusted_x, adjusted_y, adjusted_w = self._adjust_coordinates(target_x, target_y, target_w)
        offset_x, offset_y = self._calculate_offset(adjusted_x, adjusted_y)
        distance = self._calculate_distance(offset_x, offset_y)

        self._update_target_history(adjusted_x, adjusted_y)
        velocity_x, velocity_y = self._calculate_target_velocity()

        predicted_x, predicted_y = self._predict_position(offset_x, offset_y, velocity_x, velocity_y)
        smoothed_x, smoothed_y = self._apply_smoothing(predicted_x, predicted_y)

        move_x, move_y = self._convert_to_mouse_movement(smoothed_x, smoothed_y)

        if distance < adjusted_w * self._config.tremor_distance_threshold:
            move_x, move_y = self._add_tremor(move_x, move_y, distance, adjusted_w)

        return self._clamp_movement(move_x, move_y)

    def _adjust_coordinates(self, x: float, y: float, w: float) -> Tuple[float, float, float]:
        ratio = self._cached.capture_to_model_ratio
        return x * ratio, y * ratio, w * ratio

    def _calculate_offset(self, x: float, y: float) -> Tuple[float, float]:
        center = self._cached.model_center
        return x - center, y - center

    def _calculate_distance(self, x: float, y: float) -> float:
        return math.sqrt(x * x + y * y)

    def _update_target_history(self, x: float, y: float):
        self._state.history.append((x, y, time.time()))

    def _calculate_target_velocity(self) -> Tuple[float, float]:
        if len(self._state.history) < 2:
            return 0.0, 0.0

        prev = self._state.history[-2]
        current = self._state.history[-1]
        dt = current[2] - prev[2]

        if dt <= 0.001:
            return 0.0, 0.0

        inv_dt = 1.0 / dt
        return (current[0] - prev[0]) * inv_dt, (current[1] - prev[1]) * inv_dt

    def _predict_position(self, offset_x: float, offset_y: float,
                          velocity_x: float, velocity_y: float) -> Tuple[float, float]:
        prediction_time = self._config.prediction_time
        return offset_x + velocity_x * prediction_time, offset_y + velocity_y * prediction_time

    def _apply_smoothing(self, x: float, y: float) -> Tuple[float, float]:
        smooth = self._config.smooth_factor
        inv_smooth = self._cached.inv_smooth_factor

        self._state.offset_x = smooth * x + inv_smooth * self._state.offset_x
        self._state.offset_y = smooth * y + inv_smooth * self._state.offset_y

        return self._state.offset_x, self._state.offset_y

    def _convert_to_mouse_movement(self, angle_x: float, angle_y: float) -> Tuple[float, float]:
        deg_per_pixel_x = self._cached.degrees_per_pixel_x
        deg_per_pixel_y = self._cached.degrees_per_pixel_y
        dpi_factor = self._cached.dpi_factor

        move_x = (angle_x * deg_per_pixel_x / 360.0) * dpi_factor
        move_y = (angle_y * deg_per_pixel_y / 360.0) * dpi_factor

        return move_x, move_y

    def _add_tremor(self, x: float, y: float, distance: float, width: float) -> Tuple[float, float]:
        phase = self._state.tremor_phase
        self._state.tremor_phase += self._config.tremor_phase_increment

        scale = self._config.tremor_amount * (distance / width)
        tremor_x = math.sin(phase) * scale
        tremor_y = math.cos(phase * self._config.tremor_phase_multiplier) * scale

        return x + tremor_x, y + tremor_y

    def _clamp_movement(self, x: float, y: float) -> Tuple[float, float]:
        max_move = self._config.max_move
        return max(-max_move, min(max_move, x)), max(-max_move, min(max_move, y))

    def _move_worker(self):
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

    def _perform_move(self, x: float, y: float):
        ix, iy = int(x), int(y)
        if cfg.mouse_move == "makcu":
            MakcuMouse.move(ix, iy)
        else:
            logger.warning("仅支持 Makcu 移动模式")

    def _execute_movement(self, x: float, y: float):
        try:
            self._move_queue.put_nowait((x, y))
        except queue.Full:
            pass

    def stop(self):
        self._move_thread_running = False
        if self._move_thread.is_alive():
            self._move_thread.join(timeout=1.0)
        logger.info("鼠标控制器已停止")


mouse = MouseController()
