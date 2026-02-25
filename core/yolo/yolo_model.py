import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any

import supervision as sv

logger = logging.getLogger(__name__)

class YOLOModel(ABC):
    """
    YOLO模型抽象基类
    统一不同版本YOLO模型的加载和预测接口
    """
    
    def __init__(self, model_path: str, device: str, conf: float):
        """
        初始化YOLO模型
        
        Args:
            model_path: 模型路径
            device: 设备类型 (cpu, cuda, cuda:0等)
            conf: 置信度阈值
        """
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.model = None
    
    @abstractmethod
    def load_model(self):
        """
        加载模型
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        预测图像
        
        Args:
            image: 输入图像
            
        Returns:
            sv.Detections: 检测结果
        """
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        return image
    
    def postprocess(self, outputs: Any, image_shape: tuple) -> sv.Detections:
        """
        结果后处理
        
        Args:
            outputs: 模型输出
            image_shape: 原始图像形状
            
        Returns:
            sv.Detections: 处理后的检测结果
        """
        pass
    
    @abstractmethod
    def get_input_size(self):
        """
        获取模型输入大小
        
        Returns:
            int: 模型输入大小（正方形）
        """
        pass
