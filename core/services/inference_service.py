import logging
from typing import Optional
import numpy as np
import supervision as sv

from core.services.config_service import config_service

logger = logging.getLogger(__name__)


class ModelInferenceService:
    """模型推理服务"""

    def __init__(self):
        self._model = None
        self._input_size = 640
        self._device_str = None

    def load(self) -> bool:
        """加载模型"""
        try:
            from ultralytics import YOLO

            config = config_service.get_section('ai')
            model_path = f"data/{config.get('model_name', 'YOLOv8s_apex_teammate_enemy.engine')}"
            device = config.get('device', '0')

            self._device_str = f"cuda:{device}" if device.isdigit() else device

            self._model = YOLO(model_path, task="detect")
            self._input_size = self._get_model_input_size()

            self._warmup()

            logger.info(f"模型加载成功: {model_path}")
            logger.info(f"模型输入大小: {self._input_size}x{self._input_size}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _warmup(self):
        """预热模型"""
        try:
            dummy_image = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            self._model(dummy_image, conf=0.2, device=self._device_str, verbose=False)
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def _get_model_input_size(self) -> int:
        """获取模型输入大小"""
        try:
            if hasattr(self._model.model, 'yaml') and 'height' in self._model.model.yaml:
                return int(self._model.model.yaml['height'])
            return 640
        except Exception:
            return 640

    def get_input_size(self) -> int:
        """获取模型输入大小"""
        return self._input_size

    def predict(self, image: np.ndarray):
        """执行预测"""
        if self._model is None:
            logger.error("模型未加载")
            return None

        try:
            config = config_service.get_section('ai')
            conf = config.get('conf', 0.2)

            results = self._model(
                image,
                conf=conf,
                device=self._device_str,
                half=True,
                max_det=3,
                verbose=False
            )

            return self._postprocess(results)
        except Exception as e:
            logger.error(f"预测错误: {e}")
            return None

    def _postprocess(self, outputs):
        """后处理"""
        for result in outputs:
            return sv.Detections.from_ultralytics(result)
        return None

    def unload(self):
        """卸载模型"""
        self._model = None
        logger.info("模型已卸载")


inference_service = ModelInferenceService()
