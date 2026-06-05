import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import supervision as sv

from core.services.config_service import config_service

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """目标"""
    x: float
    y: float
    w: float
    h: float
    cls: int


class TargetPredictor:
    """轻量级目标预测器 — 基于历史位置的线性速度外推"""

    def __init__(self, history_size: int = 5):
        self._history = deque(maxlen=history_size)
        self._velocity = (0.0, 0.0)
        self._last_update_time = 0.0

    def update(self, target: Target):
        """用真实检测结果更新预测器"""
        now = time.time()
        self._history.append((target.x, target.y, target.w, target.h, now))
        self._velocity = self._calculate_velocity()
        self._last_update_time = now

    def predict(self, timestamp: float) -> Optional[Target]:
        """预测指定时间的 target 位置"""
        if not self._history:
            return None
        last_x, last_y, last_w, last_h, last_t = self._history[-1]
        dt = max(0.0, timestamp - last_t)
        return Target(
            x=last_x + self._velocity[0] * dt,
            y=last_y + self._velocity[1] * dt,
            w=last_w, h=last_h, cls=-1
        )

    def _calculate_velocity(self) -> Tuple[float, float]:
        """计算目标速度（像素/秒）加权平均，越近权重越高"""
        n = len(self._history)
        if n < 2:
            return 0.0, 0.0

        total_vx = total_vy = 0.0
        total_weight = 0.0

        for i in range(1, n):
            x1, y1, _, _, t1 = self._history[i - 1]
            x2, y2, _, _, t2 = self._history[i]
            dt = t2 - t1
            if dt <= 0.0005:
                continue
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
            w = float(i)  # 越近权重越高
            total_vx += vx * w
            total_vy += vy * w
            total_weight += w

        if total_weight <= 0:
            return 0.0, 0.0
        return total_vx / total_weight, total_vy / total_weight

    def reset(self):
        self._history.clear()
        self._velocity = (0.0, 0.0)
        self._last_update_time = 0.0

    @property
    def has_history(self) -> bool:
        return len(self._history) > 0

    @property
    def last_update_time(self) -> float:
        return self._last_update_time


class TargetTrackerService:
    """目标跟踪服务 — 纯 numpy 实现"""

    def __init__(self):
        self._tracked_target: Optional[Target] = None
        self._tracking_confidence = 0.0

        # 动态追踪距离：从配置加载，按目标尺寸缩放
        self._load_tracking_config()
        self._switch_cooldown = 0.08  # 略降冷却，切换更灵敏
        self._last_switch_time = 0.0
        self._last_process_time = 0.0
        self._min_process_interval = 0.002
        self._last_processed_frame_id = -1

        # 中心点（numpy 标量）
        self._center_x = 0.0
        self._center_y = 0.0
        self._update_center()

        # 目标预测器
        self._predictor = TargetPredictor(history_size=5)
        self._is_predicted = False

        # 预分配工作数组（减少GC）
        self._idx_buffer = np.arange(128, dtype=np.intp)

    def _load_tracking_config(self):
        """从配置文件加载追踪参数"""
        aim_cfg = config_service.get_section('aim')
        self._max_target_distance = aim_cfg.get('max_target_distance', 160)
        self._max_tracking_distance = max(80, self._max_target_distance // 2)
        self._max_tracking_distance_sq = self._max_tracking_distance ** 2

    def _update_center(self):
        """更新中心点坐标"""
        cfg = config_service.get_section('capture')
        self._center_x = cfg.get('window_width', 640) / 2.0
        self._center_y = cfg.get('window_height', 640) / 2.0

    def update(self, detections, frame_id=-1):
        """更新检测结果"""
        if frame_id != -1 and frame_id <= self._last_processed_frame_id:
            return

        current_time = time.time()
        if current_time - self._last_process_time < self._min_process_interval:
            return

        self._last_process_time = current_time
        self._last_processed_frame_id = frame_id

        try:
            if isinstance(detections, sv.Detections):
                self._process_detections(detections)
            elif hasattr(detections, 'boxes'):
                self._process_yolo_results(detections)
        except Exception as e:
            logger.error(f"更新检测结果失败: {e}")

    def _process_detections(self, detections: sv.Detections):
        """处理检测结果"""
        if detections.xyxy.size == 0:
            self._handle_no_detection()
            return

        target = self._select_best_target_numpy(detections)
        if target:
            self._handle_target(target, is_predicted=False)
        else:
            self._handle_no_detection()

    def _process_yolo_results(self, results):
        """处理YOLO结果（兼容旧格式）"""
        for result in results:
            if not result.boxes:
                self._handle_no_detection()
                return
            detections = sv.Detections.from_ultralytics(result)
            target = self._select_best_target_numpy(detections)
            if target:
                self._handle_target(target, is_predicted=False)
            else:
                self._handle_no_detection()

    def _select_best_target_numpy(self, detections: sv.Detections) -> Optional[Target]:
        """纯 numpy 选择最佳目标 — 零 torch 开销"""
        xyxy = detections.xyxy
        if xyxy.size == 0:
            return None

        xyxy = xyxy.reshape(-1, 4)
        n = len(xyxy)

        # 计算中心点、宽高 — 全 numpy
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]

        # 到屏幕中心的距离平方
        dx = cx - self._center_x
        dy = cy - self._center_y
        dist_sq = dx * dx + dy * dy

        # 获取 class_id
        classes = detections.class_id
        if classes is None:
            classes = np.zeros(n, dtype=np.intp)

        aim_cfg = config_service.get_section('aim')
        target_cls = aim_cfg.get('target_cls', 1.0)

        # 如果有跟踪目标且置信度高，优先跟踪（使用尺寸自适应追踪距离）
        if self._tracked_target and self._tracking_confidence > 0.3:
            current_time = time.time()
            if current_time - self._last_switch_time >= self._switch_cooldown:
                # 根据跟踪目标尺寸缩放追踪距离：大目标允许更大帧间位移
                size_scale = max(0.5, min(2.0, self._tracked_target.w / 40.0))
                adaptive_dist_sq = (self._max_tracking_distance * size_scale) ** 2

                tdx = cx - self._tracked_target.x
                tdy = cy - self._tracked_target.y
                tracked_dist_sq = tdx * tdx + tdy * tdy
                min_tracked_idx = np.argmin(tracked_dist_sq)
                if tracked_dist_sq[min_tracked_idx] < adaptive_dist_sq:
                    cls = int(classes[min_tracked_idx])
                    return Target(
                        x=cx[min_tracked_idx], y=cy[min_tracked_idx],
                        w=w[min_tracked_idx], h=h[min_tracked_idx], cls=cls
                    )
            self._last_switch_time = current_time

        # 选择最近的目标
        nearest_idx = np.argmin(dist_sq)
        cls = int(classes[nearest_idx])
        return Target(
            x=cx[nearest_idx], y=cy[nearest_idx],
            w=w[nearest_idx], h=h[nearest_idx], cls=cls
        )

    def _handle_target(self, target: Target, is_predicted: bool = False):
        """处理目标"""
        aim_cfg = config_service.get_section('aim')
        target_cls = aim_cfg.get('target_cls', 1.0)
        max_distance = aim_cfg.get('max_target_distance', 160)

        # 预测目标不检查类别
        if not is_predicted and target.cls != target_cls:
            self.reset()
            return

        dx = target.x - self._center_x
        dy = target.y - self._center_y
        distance_sq = dx * dx + dy * dy

        if distance_sq <= max_distance * max_distance:
            from core.services.aim_service import aim_service
            aim_service.process_target(target.x, target.y, target.w, target.h, is_predicted=is_predicted)

            self._tracked_target = target
            self._is_predicted = is_predicted

            if is_predicted:
                self._tracking_confidence = max(0.0, self._tracking_confidence - 0.1)
            else:
                self._tracking_confidence = min(1.0, self._tracking_confidence + 0.2)
                self._predictor.update(target)
        else:
            self.reset()

    def get_best_target(self) -> Optional[Target]:
        return self._tracked_target

    def _handle_no_detection(self):
        """无检测时用预测器保持跟踪"""
        aim_cfg = config_service.get_section('aim')
        max_miss_time = aim_cfg.get('max_miss_time', 0.15)
        max_miss_distance = aim_cfg.get('max_miss_distance', 120)

        if not self._predictor.has_history:
            self.reset()
            return

        now = time.time()
        if now - self._predictor.last_update_time > max_miss_time:
            self.reset()
            return

        predicted_target = self._predictor.predict(now)
        if predicted_target is None:
            self.reset()
            return

        dx = predicted_target.x - self._center_x
        dy = predicted_target.y - self._center_y
        if dx * dx + dy * dy > max_miss_distance * max_miss_distance:
            self.reset()
            return

        self._handle_target(predicted_target, is_predicted=True)

    def reset(self):
        self._tracked_target = None
        self._tracking_confidence = 0.0
        self._is_predicted = False
        self._predictor.reset()


tracker_service = TargetTrackerService()
