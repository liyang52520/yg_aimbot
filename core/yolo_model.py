import logging
from typing import Any

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

DEFAULT_INPUT_SIZE = 640

_EMPTY_DETECTIONS = sv.Detections(
    xyxy=np.empty((0, 4)),
    confidence=np.empty(0),
    class_id=np.empty(0, dtype=int)
)


class UltralyticsYOLOModel:
    """Ultralytics YOLO模型实现"""

    DEFAULT_INPUT_SIZE = DEFAULT_INPUT_SIZE

    def __init__(self, model_path: str, device: str, conf: float):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.model = None
        self._input_size = self.DEFAULT_INPUT_SIZE
        self._device_str = None

    def _process_device(self, device: str) -> str:
        if self._device_str is None:
            if device.isdigit():
                self._device_str = f"cuda:{device}"
            else:
                self._device_str = device
        return self._device_str

    def load_model(self) -> bool:
        try:
            from ultralytics import YOLO

            device = self._process_device(self.device)
            self.model = YOLO(self.model_path, task="detect")
            self._input_size = self._get_model_input_size()
            self._warmup()

            logger.info(f"Ultralytics YOLO模型加载成功: {self.model_path}")
            logger.info(f"模型输入大小: {self._input_size}x{self._input_size}")
            return True
        except Exception as e:
            logger.error(f"Ultralytics YOLO模型加载失败: {e}")
            return False

    def _warmup(self):
        try:
            dummy_image = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            device = self._process_device(self.device)
            self.model(dummy_image, conf=self.conf, device=device, verbose=False)
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def _get_model_input_size(self) -> int:
        try:
            if hasattr(self.model.model, 'yaml') and 'height' in self.model.model.yaml:
                return int(self.model.model.yaml['height'])
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'stride'):
                return self.DEFAULT_INPUT_SIZE
            else:
                return self.DEFAULT_INPUT_SIZE
        except Exception as e:
            logger.warning(f"获取模型输入大小失败，使用默认值{self.DEFAULT_INPUT_SIZE}: {e}")
            return self.DEFAULT_INPUT_SIZE

    def get_input_size(self) -> int:
        return self._input_size

    def predict(self, image: np.ndarray) -> sv.Detections:
        if not self.model:
            logger.error("模型未加载")
            return _EMPTY_DETECTIONS

        try:
            device = self._process_device(self.device)
            results = self.model(
                image,
                conf=self.conf,
                device=device,
                half=True,
                max_det=3,
                verbose=False
            )
            return self.postprocess(results, image.shape)
        except Exception as e:
            logger.error(f"预测错误: {e}")
            return _EMPTY_DETECTIONS

    def postprocess(self, outputs: Any, image_shape: tuple) -> sv.Detections:
        for result in outputs:
            detections = sv.Detections.from_ultralytics(result)
            return detections
        return _EMPTY_DETECTIONS
