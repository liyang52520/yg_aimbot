import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import supervision as sv
import torch

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
    """轻量级目标预测器 - 基于历史位置的线性速度外推"""

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
        dt = timestamp - last_t

        if dt < 0:
            dt = 0

        pred_x = last_x + self._velocity[0] * dt
        pred_y = last_y + self._velocity[1] * dt

        return Target(x=pred_x, y=pred_y, w=last_w, h=last_h, cls=-1)

    def _calculate_velocity(self) -> Tuple[float, float]:
        """计算目标速度（像素/秒）"""
        if len(self._history) < 2:
            return 0.0, 0.0

        total_vx, total_vy, total_weight = 0.0, 0.0, 0.0

        for i in range(1, len(self._history)):
            x1, y1, _, _, t1 = self._history[i - 1]
            x2, y2, _, _, t2 = self._history[i]
            dt = t2 - t1

            if dt <= 0.001:
                continue

            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
            weight = i

            total_vx += vx * weight
            total_vy += vy * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0, 0.0

        return total_vx / total_weight, total_vy / total_weight

    def reset(self):
        """重置预测器状态"""
        self._history.clear()
        self._velocity = (0.0, 0.0)
        self._last_update_time = 0.0

    @property
    def has_history(self) -> bool:
        """是否有历史数据可用于预测"""
        return len(self._history) > 0

    @property
    def last_update_time(self) -> float:
        """最后更新时间"""
        return self._last_update_time


class TargetTrackerService:
    """目标跟踪服务"""

    def __init__(self):
        self._tracked_target: Optional[Target] = None
        self._tracking_confidence = 0.0
        self._max_tracking_distance = 80
        self._max_tracking_distance_sq = self._max_tracking_distance ** 2
        self._switch_cooldown = 0.1
        self._last_switch_time = 0.0
        self._last_process_time = 0.0
        self._min_process_interval = 0.005
        self._arch = self._get_arch()
        self._center_tensor = None
        self._update_center_tensor()
        self._last_processed_frame_id = -1

        # 目标预测器
        self._predictor = TargetPredictor(history_size=5)
        self._is_predicted = False

    def _get_arch(self) -> str:
        """获取计算架构"""
        device = config_service.get('ai', 'device', '0')
        return 'cpu' if 'cpu' in device else f'cuda:{device}'

    def _update_center_tensor(self):
        """更新中心点张量"""
        capture_cfg = config_service.get_section('capture')
        center_x = capture_cfg.get('window_width', 320) / 2
        center_y = capture_cfg.get('window_height', 320) / 2

        self._center_tensor = torch.tensor(
            [center_x, center_y],
            device=self._arch,
            dtype=torch.float32
        )

    def update(self, detections, frame_id=-1):
        """更新检测结果"""
        # 检查frame_id是否已经处理过
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

        target = self._select_best_target(detections)
        if target:
            self._handle_target(target, is_predicted=False)
        else:
            self._handle_no_detection()

    def _process_yolo_results(self, results):
        """处理YOLO结果"""
        for result in results:
            if not result.boxes:
                self._handle_no_detection()
                return

            detections = sv.Detections.from_ultralytics(result)
            target = self._select_best_target(detections)
            if target:
                self._handle_target(target, is_predicted=False)
            else:
                self._handle_no_detection()

    def _select_best_target(self, detections: sv.Detections) -> Optional[Target]:
        """选择最佳目标"""
        try:
            xyxy = detections.xyxy
            if xyxy.size == 0:
                return None

            if xyxy.ndim == 1:
                xyxy = xyxy.reshape(1, -1)

            cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
            cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]

            xywh_np = np.stack([cx, cy, w, h], axis=1)
            xywh = torch.from_numpy(xywh_np).to(self._arch)

            if hasattr(detections, 'class_id') and detections.class_id is not None:
                if detections.class_id.ndim == 0:
                    class_id_np = np.array([detections.class_id], dtype=np.float32)
                else:
                    class_id_np = detections.class_id.astype(np.float32)
                classes = torch.from_numpy(class_id_np).to(self._arch)
            else:
                classes = torch.zeros(len(xywh), dtype=torch.float32).to(self._arch)

            return self._find_best(xywh, classes)
        except Exception as e:
            logger.error(f"选择最佳目标失败: {e}")
            return None

    def _find_best(self, boxes: torch.Tensor, classes: torch.Tensor) -> Optional[Target]:
        """找到最佳目标"""
        aim_cfg = config_service.get_section('aim')
        target_cls = aim_cfg.get('target_cls', 1.0)
        max_distance = aim_cfg.get('max_target_distance', 90)

        diff = boxes[:, :2] - self._center_tensor
        distances_sq = torch.sum(diff * diff, dim=1)

        if self._tracked_target and self._tracking_confidence > 0.3:
            current_time = time.time()

            if current_time - self._last_switch_time < self._switch_cooldown:
                return self._tracked_target

            tracked_pos = torch.tensor(
                [self._tracked_target.x, self._tracked_target.y],
                device=self._arch,
                dtype=torch.float32
            )
            tracked_diff = boxes[:, :2] - tracked_pos
            tracked_distances_sq = torch.sum(tracked_diff * tracked_diff, dim=1)

            min_tracked_dist_sq, tracked_idx = torch.min(tracked_distances_sq, dim=0)

            if min_tracked_dist_sq.item() < self._max_tracking_distance_sq:
                target_data = boxes[tracked_idx.item()].cpu().numpy()
                target_class = int(classes[tracked_idx.item()].item())
                return Target(*target_data, target_class)
            else:
                self._last_switch_time = current_time

        nearest_idx = torch.argmin(distances_sq).item()
        target_data = boxes[nearest_idx].cpu().numpy()
        target_class = int(classes[nearest_idx].item())

        return Target(*target_data, target_class)

    def _handle_target(self, target: Target, is_predicted: bool = False):
        """处理目标"""
        aim_cfg = config_service.get_section('aim')
        target_cls = aim_cfg.get('target_cls', 1.0)
        max_distance = aim_cfg.get('max_target_distance', 90)

        # 预测目标不检查类别（因为预测时 cls 被设为 -1）
        if not is_predicted and target.cls != target_cls:
            self.reset()
            return

        dx = target.x - self._center_tensor[0].item()
        dy = target.y - self._center_tensor[1].item()
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance <= max_distance:
            from core.services.aim_service import aim_service
            aim_service.process_target(target.x, target.y, target.w, target.h, is_predicted=is_predicted)

            self._tracked_target = target
            self._is_predicted = is_predicted

            if is_predicted:
                # 预测模式下置信度缓慢下降
                self._tracking_confidence = max(0.0, self._tracking_confidence - 0.1)
            else:
                self._tracking_confidence = min(1.0, self._tracking_confidence + 0.2)
                # 用真实检测更新预测器
                self._predictor.update(target)
        else:
            self.reset()

    def get_best_target(self) -> Optional[Target]:
        """获取最佳目标"""
        return self._tracked_target

    def _handle_no_detection(self):
        """处理无检测结果的情况 - 尝试用预测器保持跟踪"""
        aim_cfg = config_service.get_section('aim')
        max_miss_time = aim_cfg.get('max_miss_time', 0.15)
        max_miss_distance = aim_cfg.get('max_miss_distance', 120)

        if not self._predictor.has_history:
            self.reset()
            return

        # 检查丢失时间是否超过阈值
        now = time.time()
        elapsed = now - self._predictor.last_update_time

        if elapsed > max_miss_time:
            self.reset()
            return

        # 尝试预测当前位置
        predicted_target = self._predictor.predict(now)
        if predicted_target is None:
            self.reset()
            return

        # 检查预测位置是否超出允许范围
        dx = predicted_target.x - self._center_tensor[0].item()
        dy = predicted_target.y - self._center_tensor[1].item()
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > max_miss_distance:
            self.reset()
            return

        # 使用预测目标继续跟踪
        self._handle_target(predicted_target, is_predicted=True)

    def reset(self):
        """重置跟踪状态"""
        self._tracked_target = None
        self._tracking_confidence = 0.0
        self._is_predicted = False
        self._predictor.reset()


tracker_service = TargetTrackerService()
