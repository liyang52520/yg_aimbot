import logging
import numpy as np
import cv2
import torch
from typing import Optional, Any

import supervision as sv
from .yolo_model import YOLOModel

logger = logging.getLogger(__name__)

# 空检测结果常量，避免重复创建
_EMPTY_DETECTIONS = sv.Detections(
    xyxy=np.empty((0, 4)),
    confidence=np.empty(0),
    class_id=np.empty(0, dtype=int)
)

# 默认推理尺寸
DEFAULT_IMGSZ = (640, 640)

# 可能的输入尺寸列表
POSSIBLE_SIZES = [(320, 320), (416, 416), (640, 640), (800, 800), (1024, 1024)]


class YOLOv5Model(YOLOModel):
    """
    YOLOv5模型实现
    使用DetectMultiBackend支持不同格式的YOLOv5模型加载和预测
    """
    
    def __init__(self, model_path: str, device: str, conf: float):
        super().__init__(model_path, device, conf)
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        self.imgsz = DEFAULT_IMGSZ
        self._device_obj = None  # 缓存设备对象
    
    def load_model(self) -> bool:
        """
        加载YOLOv5模型
        使用DetectMultiBackend自动处理不同格式的模型
        """
        try:
            from models.common import DetectMultiBackend
            from utils.torch_utils import select_device
            
            # 选择设备并缓存
            self._device_obj = select_device(self.device)
            
            # 对于TensorRT引擎模型，使用固定的640x640尺寸
            if self.model_path.endswith('.engine'):
                return self._load_tensorrt_model()
            else:
                return self._load_standard_model()
                
        except Exception as e:
            logger.error(f"YOLOv5模型加载失败: {e}")
            return False
    
    def _load_tensorrt_model(self) -> bool:
        """加载TensorRT引擎模型"""
        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        
        logger.info(f"检测到TensorRT引擎模型: {self.model_path}，使用固定输入尺寸640x640")
        self.imgsz = DEFAULT_IMGSZ
        
        # 加载模型
        self.model = DetectMultiBackend(
            weights=self.model_path,
            device=self._device_obj,
            dnn=False,
            data=None,
            fp16=True
        )
        
        # 获取模型属性
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        
        # 检查图像尺寸
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        
        # 预热模型
        self._warmup()
        
        logger.info(f"YOLOv5 TensorRT模型加载成功: {self.model_path}, 输入尺寸: {self.imgsz}")
        return True
    
    def _load_standard_model(self) -> bool:
        """加载标准模型（非TensorRT）"""
        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        
        # 尝试不同的输入尺寸，直到找到适合模型的尺寸
        for size in POSSIBLE_SIZES:
            try:
                # 使用DetectMultiBackend加载模型
                self.model = DetectMultiBackend(
                    weights=self.model_path,
                    device=self._device_obj,
                    dnn=False,
                    data=None,
                    fp16=False
                )
                
                # 获取模型属性
                self.stride = self.model.stride
                self.names = self.model.names
                self.pt = self.model.pt
                
                # 使用当前尺寸
                self.imgsz = size
                
                # 检查图像尺寸
                self.imgsz = check_img_size(self.imgsz, s=self.stride)
                
                # 预热模型
                self._warmup()
                
                logger.info(f"YOLOv5模型加载成功: {self.model_path}, 输入尺寸: {self.imgsz}")
                return True
            except Exception as e:
                logger.debug(f"尝试尺寸 {size} 失败: {e}")
                continue
        
        # 如果所有尺寸都失败，尝试使用默认尺寸
        logger.error("所有尝试的输入尺寸都失败，使用默认尺寸重试")
        return self._load_with_default_size()
    
    def _load_with_default_size(self) -> bool:
        """使用默认尺寸加载模型"""
        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        
        self.model = DetectMultiBackend(
            weights=self.model_path,
            device=self._device_obj,
            dnn=False,
            data=None,
            fp16=False
        )
        
        # 获取模型属性
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        
        # 检查图像尺寸
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        
        # 预热模型
        self._warmup()
        
        logger.info(f"YOLOv5模型加载成功: {self.model_path}, 输入尺寸: {self.imgsz}")
        return True
    
    def _warmup(self):
        """预热模型"""
        if self.model is None:
            return
            
        try:
            # 对于不同类型的模型，使用不同的预热方式
            if self.model.pt or self.model.triton:
                # PyTorch或Triton模型
                self.model.warmup(imgsz=(1, 3, *self.imgsz))
            else:
                # 其他类型的模型
                im = torch.empty((1, 3, *self.imgsz), dtype=torch.float, device=self.model.device)
                self.model.warmup(imgsz=(1, 3, *self.imgsz))
        except Exception as e:
            logger.debug(f"模型预热失败: {e}")
            # 预热失败不影响模型加载，继续执行
    
    def get_input_size(self) -> int:
        """
        获取模型输入大小
        
        Returns:
            int: 模型输入大小（正方形）
        """
        if isinstance(self.imgsz, (tuple, list)):
            return self.imgsz[0]
        else:
            return int(self.imgsz)
    
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
            # 图像预处理
            from utils.dataloaders import letterbox
            from utils.general import non_max_suppression, scale_boxes
            
            # 调整图像大小，使用模型实际的输入尺寸
            im = letterbox(image, self.imgsz, stride=self.stride, auto=self.pt)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            
            # 转换为张量
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
            # 执行推理
            pred = self._run_inference(im)
            
            # 非极大值抑制
            pred = non_max_suppression(
                pred, 
                self.conf, 
                0.45,  # IoU阈值
                classes=None, 
                agnostic=False,
                max_det=1000
            )
            
            # 处理预测结果
            return self._process_predictions(pred, im, image)
            
        except Exception as e:
            logger.error(f"预测错误: {e}")
            return _EMPTY_DETECTIONS
    
    def _run_inference(self, im: torch.Tensor):
        """执行推理，处理OpenVINO特殊情况"""
        # 处理OpenVINO模型的特殊情况
        if hasattr(self.model, 'xml') and self.model.xml and im.shape[0] > 1:
            return self._run_openvino_inference(im)
        else:
            return self.model(im, augment=False, visualize=False)
    
    def _run_openvino_inference(self, im: torch.Tensor):
        """处理OpenVINO模型的批量推理"""
        ims = torch.chunk(im, im.shape[0], 0)
        pred = None
        for image_chunk in ims:
            chunk_pred = self.model(image_chunk, augment=False, visualize=False).unsqueeze(0)
            if pred is None:
                pred = chunk_pred
            else:
                pred = torch.cat((pred, chunk_pred), dim=0)
        return [pred, None]
    
    def _process_predictions(self, pred, im: torch.Tensor, original_image: np.ndarray) -> sv.Detections:
        """处理预测结果"""
        from utils.general import scale_boxes
        
        for i, det in enumerate(pred):  # per image
            if len(det):
                # 调整边界框大小到原始图像
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], original_image.shape).round()
                
                # 转换为Supervision Detections
                xyxy = det[:, :4].cpu().numpy()
                confidence = det[:, 4].cpu().numpy()
                class_id = det[:, 5].cpu().numpy().astype(int)
                
                return sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id
                )
        
        # 如果没有检测结果，返回空检测
        return _EMPTY_DETECTIONS
