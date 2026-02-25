import logging
from typing import Optional

from .ultralytics_yolo_model import UltralyticsYOLOModel
from .yolo_model import YOLOModel
from .yolov5_model import YOLOv5Model

logger = logging.getLogger(__name__)


def create_yolo_model(model_path: str, device: str, conf: float, model_type: str = "ultralytics") -> Optional[
    YOLOModel]:
    """
    创建YOLO模型实例
    
    Args:
        model_path: 模型路径
        device: 设备类型 (cpu, cuda, cuda:0等)
        conf: 置信度阈值
        model_type: 模型类型 (yolov5, ultralytics)
        
    Returns:
        Optional[YOLOModel]: YOLO模型实例
    """
    logger.info(f"接收到的模型类型: {model_type}")
    logger.info(f"模型路径: {model_path}")
    # 对于PT格式的模型，使用Ultralytics YOLO类加载
    if model_type == "ultralytics":
        logger.info("使用UltralyticsYOLOModel加载模型")
        model = UltralyticsYOLOModel(model_path, device, conf)
    # 对于YOLOv5非PT格式的模型，使用YOLOv5Model类加载
    elif model_type == "yolov5":
        logger.info("使用YOLOv5Model加载模型")
        model = YOLOv5Model(model_path, device, conf)
    else:
        raise ValueError("Invalid model type")

    if model.load_model():
        return model
    raise ValueError("Failed to load model")
