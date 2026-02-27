import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

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

    def update(self, detections):
        """更新检测结果"""
        current_time = time.time()

        if current_time - self._last_process_time < self._min_process_interval:
            return

        self._last_process_time = current_time

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
            self.reset()
            return

        target = self._select_best_target(detections)
        if target:
            self._handle_target(target)
        else:
            self.reset()

    def _process_yolo_results(self, results):
        """处理YOLO结果"""
        for result in results:
            if not result.boxes:
                self.reset()
                return

            detections = sv.Detections.from_ultralytics(result)
            target = self._select_best_target(detections)
            if target:
                self._handle_target(target)
            else:
                self.reset()

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

    def _handle_target(self, target: Target):
        """处理目标"""
        aim_cfg = config_service.get_section('aim')
        target_cls = aim_cfg.get('target_cls', 1.0)
        max_distance = aim_cfg.get('max_target_distance', 90)

        if target.cls != target_cls:
            self.reset()
            return

        dx = target.x - self._center_tensor[0].item()
        dy = target.y - self._center_tensor[1].item()
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance <= max_distance:
            from core.services.aim_service import aim_service
            aim_service.process_target(target.x, target.y, target.w, target.h)

            self._tracked_target = target
            self._tracking_confidence = min(1.0, self._tracking_confidence + 0.2)
        else:
            self.reset()

    def get_best_target(self) -> Optional[Target]:
        """获取最佳目标"""
        return self._tracked_target

    def reset(self):
        """重置跟踪状态"""
        self._tracked_target = None
        self._tracking_confidence = 0.0


tracker_service = TargetTrackerService()
