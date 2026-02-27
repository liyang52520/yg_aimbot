import logging
import numpy as np
from typing import Optional, Any

import supervision as sv

logger = logging.getLogger(__name__)

# 空检测结果常量，避免重复创建
_EMPTY_DETECTIONS = sv.Detections(
    xyxy=np.empty((0, 4)),
    confidence=np.empty(0),
    class_id=np.empty(0, dtype=int)
)


class UltralyticsYOLOModel:
    """
    Ultralytics YOLO模型实现
    支持直接使用ultralytics库加载各种格式的模型
    """
    
    # 默认输入大小
    DEFAULT_INPUT_SIZE = 640
    
    def __init__(self, model_path: str, device: str, conf: float):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.model = None
        self._input_size = self.DEFAULT_INPUT_SIZE
        self._device_str = None  # 缓存处理后的设备字符串
    
    def _process_device(self, device: str) -> str:
        """处理设备字符串，缓存结果避免重复处理"""
        if self._device_str is None:
            if device.isdigit():
                self._device_str = f"cuda:{device}"
            else:
                self._device_str = device
        return self._device_str
    
    def load_model(self) -> bool:
        """
        加载Ultralytics YOLO模型
        使用ultralytics库直接加载各种格式的模型
        """
        try:
            from ultralytics import YOLO
            
            device = self._process_device(self.device)
            
            # 使用ultralytics库加载模型（支持pt、onnx、engine、Openvino等格式）
            self.model = YOLO(self.model_path, task="detect")
            
            # 获取模型输入大小
            self._input_size = self._get_model_input_size()
            
            # 预热模型
            self._warmup()
            
            logger.info(f"Ultralytics YOLO模型加载成功: {self.model_path}")
            logger.info(f"模型输入大小: {self._input_size}x{self._input_size}")
            return True
        except Exception as e:
            logger.error(f"Ultralytics YOLO模型加载失败: {e}")
            return False
    
    def _warmup(self):
        """预热模型"""
        try:
            dummy_image = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            device = self._process_device(self.device)
            self.model(dummy_image, conf=self.conf, device=device, verbose=False)
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def _get_model_input_size(self) -> int:
        """
        获取模型输入大小
        
        Returns:
            int: 模型输入大小（正方形）
        """
        try:
            # 对于Ultralytics YOLO模型，我们可以通过模型的配置获取输入大小
            if hasattr(self.model.model, 'yaml') and 'height' in self.model.model.yaml:
                return int(self.model.model.yaml['height'])
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'stride'):
                # 对于某些模型，我们可以通过stride推断输入大小
                return self.DEFAULT_INPUT_SIZE
            else:
                return self.DEFAULT_INPUT_SIZE
        except Exception as e:
            logger.warning(f"获取模型输入大小失败，使用默认值{self.DEFAULT_INPUT_SIZE}: {e}")
            return self.DEFAULT_INPUT_SIZE
    
    def get_input_size(self) -> int:
        """
        获取模型输入大小
        
        Returns:
            int: 模型输入大小（正方形）
        """
        return self._input_size
    
    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        预测图像
        
        Args:
            image: 输入图像
            
        Returns:
            sv.Detections: 检测结果
        """
        if not self.model:
            logger.error("模型未加载")
            return _EMPTY_DETECTIONS
        
        try:
            device = self._process_device(self.device)
            
            # 执行推理（ultralytics库会自动处理不同格式的模型）
            results = self.model(
                image, 
                conf=self.conf, 
                device=device, 
                half=True, 
                max_det=3, 
                verbose=False
            )
            
            # 后处理
            return self.postprocess(results, image.shape)
        except Exception as e:
            logger.error(f"预测错误: {e}")
            return _EMPTY_DETECTIONS
    
    def postprocess(self, outputs: Any, image_shape: tuple) -> sv.Detections:
        """
        处理Ultralytics YOLO输出
        
        Args:
            outputs: 模型输出
            image_shape: 原始图像形状
            
        Returns:
            sv.Detections: 检测结果
        """
        # 从YOLO结果转换为Supervision Detections
        for result in outputs:
            detections = sv.Detections.from_ultralytics(result)
            return detections
        
        # 如果没有结果，返回空检测
        return _EMPTY_DETECTIONS
