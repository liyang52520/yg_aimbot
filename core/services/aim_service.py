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

# === 运动控制参数 ===
# 自适应分数：连续指数曲线，无阈值跳变，r=0→0.25, r=3→0.39, r=10→0.51, r=20+→0.55
MOVE_FRACTION_ASYM_SMALL = 0.25  # 零距离渐近下限
MOVE_FRACTION_ASYM_LARGE = 0.55  # 无穷距离渐近上限
MOVE_SLEEP_ACTIVE = 0.0005   # 运动中休眠 0.5ms → ~2000Hz
MOVE_SLEEP_IDLE = 0.001      # 空闲时休眠 1ms 降低 CPU（原3ms）
TARGET_HISTORY_MAXLEN = 4    # 速度预测用 4 帧


@dataclass
class MouseConfig:
    """鼠标配置"""
    dpi: float
    sensitivity: float
    fov_width: float
    fov_height: float
    window_width: int
    window_height: int
    smooth_factor: float = 0.5       # 原 0.6 → 更少平滑延迟
    max_move: float = 35.0            # 原 20.0 → 单次最大移动
    min_move: float = 0.01            # 几乎消除空闲死区，亚像素累积使输出更平滑
    tremor_amount: float = 0.015      # 原 0.02 → 更小的抖动
    prediction_time: float = 0.04     # 原 0.05 → 更短的预测时间
    tremor_phase_increment: float = 0.25
    tremor_phase_multiplier: float = 1.2
    tremor_distance_threshold: float = 0.25
    max_frame_move: float = 8.0    # 每帧最大鼠标输出(count)，防瞬移

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
    _last_move_time = 0.0
    _lock = None
    _initialized = False

    # 移动限幅
    _scope = 50

    @classmethod
    def _get_lock(cls) -> threading.Lock:
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def get_device(cls):
        """获取设备（双检锁）"""
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
        """移动鼠标（限幅防越界）"""
        try:
            x = max(-cls._scope, min(x, cls._scope))
            y = max(-cls._scope, min(y, cls._scope))

            device = cls.get_device()
            if device is None:
                return False
            device.move(x, y)
            cls._last_move_time = time.perf_counter()
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
            logger.info("鼠标控制器已重置")


class AimService:
    """瞄准服务 — 优化版运动控制"""

    def __init__(self):
        self._config = self._load_config()
        self._state = TargetState()
        self._model_input_size = 640

        # 连续运动控制状态
        self._target_remain_x = 0.0
        self._target_remain_y = 0.0
        self._move_lock = threading.Lock()
        self._frac_x = 0.0  # 亚像素累积器 X
        self._frac_y = 0.0  # 亚像素累积器 Y
        self._new_target_arrived = False  # 标记新目标到来
        self._frame_move_x = 0.0  # 当前帧累计 X 输出
        self._frame_move_y = 0.0  # 当前帧累计 Y 输出

        self._move_thread_running = True
        self._move_paused = False
        self._move_thread = threading.Thread(target=self._move_worker, daemon=True)
        self._move_thread.start()
        self._update_cache()
        self._cache_aim_config()

        config_service.register_callback(self._on_config_change)

    def _cache_aim_config(self):
        """缓存 aim 配置值，避免热路径重复读锁"""
        aim_cfg = config_service.get_section('aim')
        self._cached_body_x_offset = aim_cfg.get('body_x_offset', 0.1)
        self._cached_body_y_offset = aim_cfg.get('body_y_offset', 0.1)

    def _on_config_change(self, section: str, updates: Dict[str, Any]):
        if section in ('aim', 'mouse', 'capture'):
            self.update_config()
        if section == 'aim':
            self._cache_aim_config()

    def _load_config(self) -> MouseConfig:
        mouse_cfg = config_service.get_section('mouse')
        capture_cfg = config_service.get_section('capture')
        return MouseConfig(
            dpi=mouse_cfg.get('dpi', 1100),
            sensitivity=mouse_cfg.get('sensitivity', 3.0),
            fov_width=mouse_cfg.get('fov_width', 40),
            fov_height=mouse_cfg.get('fov_height', 40),
            window_width=capture_cfg.get('window_width', 640),
            window_height=capture_cfg.get('window_height', 640)
        )

    def _update_cache(self):
        """更新缓存的计算因子"""
        self._dpi_factor = self._config.dpi / self._config.sensitivity
        self._capture_to_model_ratio = self._model_input_size / max(
            self._config.window_width, self._config.window_height
        )
        self._degrees_per_pixel_x = self._config.fov_width / self._model_input_size
        self._degrees_per_pixel_y = self._config.fov_height / self._model_input_size
        self._model_center = self._model_input_size * 0.5
        self._inv_smooth_factor = 1.0 - self._config.smooth_factor

    def update_config(self):
        self._config = self._load_config()
        self._update_cache()

    def set_model_input_size(self, input_size: int):
        self._model_input_size = input_size
        self._update_cache()
        logger.info(f"模型输入大小已更新: {input_size}x{input_size}")

    def process_target(self, x: float, y: float, w: float, h: float, is_predicted: bool = False):
        """处理目标 — 计算移动量并设置为连续运动目标"""
        move_x, move_y = self._calculate_movement(x, y, w, h, is_predicted=is_predicted)
        with self._move_lock:
            self._target_remain_x = move_x
            self._target_remain_y = move_y
            self._new_target_arrived = True
            self._frame_move_x = 0.0
            self._frame_move_y = 0.0

    def _calculate_movement(self, target_x: float, target_y: float,
                            target_w: float, target_h: float,
                            is_predicted: bool = False) -> Tuple[float, float]:
        """计算移动量"""
        # 身体偏移（使用缓存的配置值，避免热路径重复读锁）
        adjusted_x = target_x + self._cached_body_x_offset * target_w * 0.5
        adjusted_y = target_y + self._cached_body_y_offset * target_h * 0.5

        # 模型空间映射
        ratio = self._capture_to_model_ratio
        model_x = adjusted_x * ratio
        model_y = adjusted_y * ratio

        offset_x = model_x - self._model_center
        offset_y = model_y - self._model_center
        distance = math.sqrt(offset_x ** 2 + offset_y ** 2)

        # 速度预测补偿
        self._state.history.append((offset_x, offset_y, time.time()))
        velocity_x, velocity_y = self._calculate_velocity()
        predicted_x = offset_x + velocity_x * self._config.prediction_time
        predicted_y = offset_y + velocity_y * self._config.prediction_time

        # 自适应平滑：预测模式下更激进（更快响应真实检测修正）
        smooth_factor = self._config.smooth_factor
        if is_predicted:
            smooth_factor = max(0.25, smooth_factor * 0.6)

        inv = 1.0 - smooth_factor
        smoothed_x = smooth_factor * predicted_x + inv * self._state.offset_x
        smoothed_y = smooth_factor * predicted_y + inv * self._state.offset_y

        self._state.offset_x = smoothed_x
        self._state.offset_y = smoothed_y

        # 空间映射到鼠标坐标
        move_x = (smoothed_x * self._degrees_per_pixel_x / 360.0) * self._dpi_factor
        move_y = (smoothed_y * self._degrees_per_pixel_y / 360.0) * self._dpi_factor

        # 抖动（仅真实检测时）
        if not is_predicted and distance < target_w * self._config.tremor_distance_threshold:
            move_x, move_y = self._add_tremor(move_x, move_y, distance, target_w)

        return self._clamp(move_x), self._clamp(move_y)

    def _calculate_velocity(self) -> Tuple[float, float]:
        """计算目标速度（加权平均后两帧 vs 前三帧）"""
        n = len(self._state.history)
        if n < 2:
            return 0.0, 0.0
        # 使用最后两个速度估计加权：近的权重高
        vx, vy = 0.0, 0.0
        total_w = 0.0
        end = n - 1
        for i in range(max(0, end - 2), end):
            p0 = self._state.history[i]
            p1 = self._state.history[i + 1]
            dt = p1[2] - p0[2]
            if dt <= 0.001:
                continue
            w = i - max(0, end - 2) + 1  # 越近权重越高
            vx += ((p1[0] - p0[0]) / dt) * w
            vy += ((p1[1] - p0[1]) / dt) * w
            total_w += w
        if total_w <= 0:
            return 0.0, 0.0
        return vx / total_w, vy / total_w

    def _add_tremor(self, x: float, y: float, distance: float, width: float) -> Tuple[float, float]:
        """添加微抖动（帮助微调瞄准）"""
        phase = self._state.tremor_phase
        self._state.tremor_phase += self._config.tremor_phase_increment
        scale = self._config.tremor_amount * (distance / max(width, 1.0))
        return (
            x + math.sin(phase) * scale,
            y + math.cos(phase * self._config.tremor_phase_multiplier) * scale
        )

    def _clamp(self, value: float) -> float:
        return max(-self._config.max_move, min(self._config.max_move, value))

    @staticmethod
    def _adaptive_fraction(remaining: float) -> float:
        """连续指数自适应分数：平滑过渡无跳变，远快近稳
        r=0 -> 0.25, r=1 -> 0.30, r=3 -> 0.39, r=5 -> 0.44,
        r=10 -> 0.51, r=20+ -> 0.55
        """
        abs_r = abs(remaining)
        return 0.55 - 0.30 * math.exp(-abs_r * 0.20)

    def _move_worker(self):
        """高频率运动工作线程 — 自适应分数 + 自适应频率"""
        while self._move_thread_running:
            if self._move_paused:
                time.sleep(0.005)
                continue
            try:
                with self._move_lock:
                    rx = self._target_remain_x
                    ry = self._target_remain_y

                # 空闲检测
                idle = abs(rx) < self._config.min_move and abs(ry) < self._config.min_move
                if idle:
                    time.sleep(MOVE_SLEEP_IDLE)
                    continue

                # === 自适应分数：远快近稳 ===
                frac_x = self._adaptive_fraction(rx)
                frac_y = self._adaptive_fraction(ry)
                # 使用两个方向中较大的分数（保证移动协调）
                fraction = max(frac_x, frac_y)

                step_x = rx * fraction
                step_y = ry * fraction

                step_x = self._clamp(step_x)
                step_y = self._clamp(step_y)

                # 每帧运动总量限制（防瞬移）
                remaining_x = self._config.max_frame_move - abs(self._frame_move_x)
                remaining_y = self._config.max_frame_move - abs(self._frame_move_y)
                if abs(step_x) > remaining_x:
                    step_x = math.copysign(max(0.0, remaining_x), step_x)
                if abs(step_y) > remaining_y:
                    step_y = math.copysign(max(0.0, remaining_y), step_y)
                self._frame_move_x += abs(step_x)
                self._frame_move_y += abs(step_y)

                # 亚像素累积：避免 int() 截断丢失精度
                self._frac_x += step_x
                self._frac_y += step_y
                ix = int(self._frac_x)
                iy = int(self._frac_y)
                self._frac_x -= ix
                self._frac_y -= iy

                if ix != 0 or iy != 0:
                    MouseControlService.move(ix, iy)

                # 更新剩余目标
                with self._move_lock:
                    self._target_remain_x -= step_x
                    self._target_remain_y -= step_y
                    # 防过冲
                    if self._target_remain_x * rx < 0:
                        self._target_remain_x = 0.0
                    if self._target_remain_y * ry < 0:
                        self._target_remain_y = 0.0

                # 自适应休眠：大幅移动时高频，快收敛时降频
                remaining = math.sqrt(rx * rx + ry * ry)
                if remaining > 10:
                    time.sleep(MOVE_SLEEP_ACTIVE)
                elif remaining > 2:
                    time.sleep(MOVE_SLEEP_ACTIVE * 1.5)
                else:
                    time.sleep(MOVE_SLEEP_ACTIVE * 2)

            except Exception as e:
                logger.error(f"运动线程错误: {e}")
                time.sleep(0.003)

    def pause_moves(self):
        """暂停运动工作线程（测试用）"""
        self._move_paused = True

    def resume_moves(self):
        """恢复运动工作线程"""
        self._move_paused = False

    def stop(self):
        self._move_thread_running = False
        if self._move_thread.is_alive():
            self._move_thread.join(timeout=1.0)
        logger.info("瞄准服务已停止")


aim_service = AimService()
mouse_service = MouseControlService
