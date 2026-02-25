import logging
import time

import numpy as np
import supervision as sv
import torch

from core.capture import capture
from core.config import cfg
from core.mouse import mouse

logger = logging.getLogger(__name__)


class Target:
    """目标对象"""

    def __init__(self, x, y, w, h, cls):
        # 使用实时的配置值，确保配置变更时能立即生效
        self.x = x + cfg.aim_body_x_offset * h * 0.5
        self.y = y + cfg.aim_body_y_offset * h * 0.5
        self.w = w
        self.h = h
        self.cls = cls


class FrameParser:
    """
    帧解析器
    处理目标检测结果，选择最佳目标并进行跟踪
    """

    def __init__(self):
        self.arch = self._get_arch()
        self.tracked_target = None
        self.tracking_confidence = 0.0
        self.max_tracking_distance = 80  # 像素
        self.switch_cooldown = 0.1  # 秒
        self.last_switch_time = 0
        self.last_process_time = 0
        self.min_process_interval = 0.005

    def parse(self, result):
        """解析检测结果"""
        current_time = time.time()

        # 限制处理频率
        if current_time - self.last_process_time < self.min_process_interval:
            return

        self.last_process_time = current_time

        try:
            if isinstance(result, sv.Detections):
                self._process_sv_detections(result)
            else:
                self._process_yolo_detections(result)
        except Exception as e:
            logger.error(f"Error in parse: {e}", exc_info=True)

    def _process_sv_detections(self, detections):
        """处理Supervision检测结果"""
        if not detections.xyxy.any():
            self._reset_tracking()
            return

        target = self.select_target(detections)
        if target:
            self._handle_target(target)
        else:
            self._reset_tracking()

    def _process_yolo_detections(self, results):
        """处理YOLO检测结果"""
        for frame in results:
            if not frame.boxes:
                self._reset_tracking()
                return

            target = self.select_target(frame)
            if target:
                self._handle_target(target)
            else:
                self._reset_tracking()

    def _handle_target(self, target):
        """处理目标"""
        if target.cls != cfg.aim_target_cls:
            self._reset_tracking()
            return

        # 计算目标与屏幕中心的距离
        import math
        center_x = capture.screen_x_center
        center_y = capture.screen_y_center
        distance = math.sqrt((target.x - center_x) ** 2 + (target.y - center_y) ** 2)

        # 检查目标是否在配置的最大距离范围内
        if hasattr(cfg, 'aim_max_target_distance'):
            max_distance = cfg.aim_max_target_distance
        else:
            max_distance = 150  # 默认值

        # 只有在范围内才进行瞄准
        if distance <= max_distance:
            # 传递给鼠标控制
            mouse.process_data((target.x, target.y, target.w, target.h, target.cls))

            # 更新跟踪状态
            self.tracked_target = target
            self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)
        else:
            # 目标太远，重置跟踪
            self._reset_tracking()

    def _reset_tracking(self):
        """重置跟踪状态"""
        self.tracked_target = None
        self.tracking_confidence = 0.0

    def select_target(self, frame):
        """选择最佳目标"""
        try:
            # 转换数据格式
            if isinstance(frame, sv.Detections):
                boxes_array, classes_tensor = self._convert_sv_to_tensor(frame)
            else:
                # 旧的YOLO格式支持
                boxes_array = frame.boxes.xywh.to(self.arch)
                classes_tensor = frame.boxes.cls.to(self.arch)

            if not classes_tensor.numel():
                return None

            return self._find_best_target(boxes_array, classes_tensor)
        except Exception as e:
            logger.error(f"Error in select_target: {e}", exc_info=True)
            return None

    def _convert_sv_to_tensor(self, frame):
        """转换Supervision格式到tensor"""
        xyxy = frame.xyxy
        # 计算中心点和宽高
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]

        # 合并为单一numpy数组再转换为tensor
        xywh_np = np.stack([cx, cy, w, h], axis=1)
        xywh = torch.tensor(xywh_np, dtype=torch.float32).to(self.arch)

        classes_tensor = torch.from_numpy(np.array(frame.class_id, dtype=np.float32)).to(self.arch)
        return xywh, classes_tensor

    def _find_best_target(self, boxes_array, classes_tensor):
        """找到最佳目标"""
        center = torch.tensor([capture.screen_x_center, capture.screen_y_center], device=self.arch)

        # 计算距离
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)

        # 优先保持目标跟踪
        if self.tracked_target and self.tracking_confidence > 0.3:
            current_time = time.time()

            # 检查冷却期
            if current_time - self.last_switch_time < self.switch_cooldown:
                return self.tracked_target

            # 查找最接近当前跟踪目标的新目标
            tracked_pos = torch.tensor([self.tracked_target.x, self.tracked_target.y], device=self.arch)
            tracked_distances = torch.sum((boxes_array[:, :2] - tracked_pos) ** 2, dim=1)

            min_tracked_dist, tracked_idx = torch.min(tracked_distances, dim=0)

            # 如果距离在合理范围内，继续跟踪
            if min_tracked_dist.item() < self.max_tracking_distance ** 2:
                target_data = boxes_array[tracked_idx.item(), :4].cpu().numpy()
                target_class = classes_tensor[tracked_idx.item()].item()
                return Target(*target_data, target_class)
            else:
                # 目标丢失，记录时间
                self.last_switch_time = current_time

        # 选择距离最近的目标
        nearest_idx = torch.argmin(distances_sq).item()
        target_data = boxes_array[nearest_idx, :4].cpu().numpy()
        target_class = classes_tensor[nearest_idx].item()
        return Target(*target_data, target_class)

    def _get_arch(self):
        """获取计算架构"""
        return 'cpu' if 'cpu' in cfg.ai_device else f'cuda:{cfg.ai_device}'


# 创建全局实例
frameParser = FrameParser()
