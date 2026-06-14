"""
异步推理服务 — 单工作线程 + 多后端抽象

后端由 model_type 配置选择：
  - onnx      → ONNX Runtime
  - tensorrt  → TensorRT 10.x
  - ultralytics → Ultralytics YOLO
"""
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, List, Any

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from ultralytics import YOLO

from core.services.config_service import config_service
from core.ui_bridge import ui_bridge as image_signal

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """推理结果"""
    frame_id: int
    detections: Any
    timestamp: float
    processing_time: float
    original_frame: Optional[np.ndarray] = None


# =========================================================================
# Shared ONNX-format pre / post processing (used by ONNX & TensorRT backends)
# =========================================================================

def _preprocess_onnx(image: np.ndarray, input_size: int) -> np.ndarray:
    """HWC BGR uint8 → NCHW RGB float32 [0,1] (letterbox resize + pad)"""
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (input_size - new_h) // 2
    left = (input_size - new_w) // 2
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    tensor = canvas.astype(np.float32) / 255.0
    tensor = tensor[:, :, ::-1]                     # BGR → RGB
    tensor = np.transpose(tensor, (2, 0, 1))        # HWC → CHW
    tensor = np.expand_dims(tensor, axis=0)         # → NCHW
    return tensor


def _postprocess_onnx(raw_output: np.ndarray, conf_thresh: float,
                      orig_w: int, orig_h: int, input_size: int) -> Optional[sv.Detections]:
    """[1, N, 8] → sv.Detections (letterbox de-pad + NMS)"""
    dets = raw_output[0]                     # (N, 8)
    boxes = dets[:, :4]                      # cx, cy, w, h (model space)
    obj_conf = dets[:, 4]
    cls_scores = dets[:, 5:]

    max_cls = cls_scores.max(axis=1)
    confidences = obj_conf * max_cls

    keep = confidences > conf_thresh
    if not keep.any():
        return None

    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = cls_scores[keep].argmax(axis=1).astype(int)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    # Reverse letterbox padding
    scale = min(input_size / orig_h, input_size / orig_w)
    pad_x = (input_size - orig_w * scale) / 2
    pad_y = (input_size - orig_h * scale) / 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    xyxy = np.clip(xyxy, [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h])

    detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
    return detections.with_nms(threshold=0.5)


# =========================================================================
# Backend strategy
# =========================================================================

class _BaseBackend(ABC):
    """推理后端基类"""

    def __init__(self):
        self.input_size = 640

    @abstractmethod
    def load(self, model_path: str, device: str, conf_threshold: float) -> bool:
        ...

    @abstractmethod
    def infer(self, image: np.ndarray, conf: float) -> Optional[sv.Detections]:
        ...

    @abstractmethod
    def unload(self):
        ...

    def warmup_worker_thread(self):
        """Worker 线程 CUDA 上下文初始化（TRT 需要覆盖）"""
        pass


class _UltralyticsBackend(_BaseBackend):
    """Ultralytics YOLO 后端"""

    def __init__(self):
        super().__init__()
        self._model: Optional[YOLO] = None
        self._device = "cpu"

    def load(self, model_path: str, device: str, conf_threshold: float) -> bool:
        try:
            self._model = YOLO(model_path, task="detect")
            self.input_size = self._resolve_input_size()
            self._device = device
            self._warmup(conf_threshold)
            logger.info(f"Ultralytics 加载成功: {model_path}, 输入: {self.input_size}")
            return True
        except Exception as e:
            logger.error(f"Ultralytics 加载失败: {e}")
            self._model = None
            return False

    def _resolve_input_size(self) -> int:
        try:
            if hasattr(self._model.model, 'yaml') and 'height' in self._model.model.yaml:  # type: ignore
                return int(self._model.model.yaml['height'])  # type: ignore
        except Exception:
            pass
        return 640

    def _warmup(self, conf_threshold: float):
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self._model(dummy, conf=conf_threshold, device=self._device)

    def infer(self, image: np.ndarray, conf: float) -> Optional[sv.Detections]:
        if self._model is None:
            return None
        try:
            results = self._model(
                image, conf=conf, device=self._device, half=True,
                max_det=3, verbose=False, stream=True, agnostic_nms=True,
            )
            result = next(results, None)
            return sv.Detections.from_ultralytics(result) if result else None
        except Exception as e:
            logger.error(f"Ultralytics 推理错误: {e}")
            return None

    def unload(self):
        self._model = None


class _ONNXBackend(_BaseBackend):
    """ONNX Runtime 后端"""

    def __init__(self):
        super().__init__()
        self._session: Optional[ort.InferenceSession] = None

    def load(self, model_path: str, device: str, conf_threshold: float) -> bool:
        try:
            available = ort.get_available_providers()
            preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in preferred if p in available] or ['CPUExecutionProvider']
            logger.info(f"ONNX providers: {providers}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options, providers=providers)

            inp = self._session.get_inputs()[0]
            self.input_size = inp.shape[2]
            logger.info(f"ONNX 输入: {inp.name} {inp.shape}")

            self._warmup()
            logger.info(f"ONNX 加载成功: {model_path}, 输入: {self.input_size}")
            return True
        except Exception as e:
            logger.error(f"ONNX 加载失败: {e}")
            return False

    def _warmup(self):
        try:
            dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
            self._session.run(None, {self._session.get_inputs()[0].name: dummy})  # type: ignore
        except Exception as e:
            logger.warning(f"ONNX 预热失败: {e}")

    def infer(self, image: np.ndarray, conf: float) -> Optional[sv.Detections]:
        if self._session is None:
            return None
        try:
            input_tensor = _preprocess_onnx(image, self.input_size)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})
            return _postprocess_onnx(outputs[0], conf, image.shape[1], image.shape[0], self.input_size)
        except Exception as e:
            logger.error(f"ONNX 推理错误: {e}")
            return None

    def unload(self):
        self._session = None


class _TensorRTBackend(_BaseBackend):
    """TensorRT 后端（TRT 10.x，GPU 内存由 torch 管理）"""

    def __init__(self):
        super().__init__()
        self._engine = None
        self._context = None
        self._input_tensor = None
        self._output_tensor = None

    def load(self, model_path: str, device: str, conf_threshold: float) -> bool:
        try:
            import tensorrt as trt
            import torch

            trt_logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(trt_logger)

            with open(model_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            context = engine.create_execution_context()

            input_name = output_name = None
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_name = name
                else:
                    output_name = name

            input_shape = tuple(context.get_tensor_shape(input_name))
            if any(d == -1 for d in input_shape):
                input_shape = (1, 3, 320, 320)
                context.set_input_shape(input_name, input_shape)
            self.input_size = input_shape[2]

            output_shape = tuple(context.get_tensor_shape(output_name))

            self._input_tensor = torch.empty(input_shape, dtype=torch.float32, device='cuda:0')
            self._output_tensor = torch.empty(output_shape, dtype=torch.float32, device='cuda:0')

            context.set_tensor_address(input_name, self._input_tensor.data_ptr())
            context.set_tensor_address(output_name, self._output_tensor.data_ptr())

            self._engine = engine
            self._context = context

            logger.info(f"TensorRT 加载成功: {model_path}, 输入: {self.input_size}")
            return True
        except Exception as e:
            logger.error(f"TensorRT 加载失败: {e}")
            return False

    def infer(self, image: np.ndarray, conf: float) -> Optional[sv.Detections]:
        if self._context is None:
            return None
        try:
            input_tensor = np.ascontiguousarray(_preprocess_onnx(image, self.input_size))
            import torch
            self._input_tensor.copy_(torch.from_numpy(input_tensor))
            self._context.execute_async_v3(0)
            h_output = self._output_tensor.cpu().numpy()
            return _postprocess_onnx(h_output, conf, image.shape[1], image.shape[0], self.input_size)
        except Exception as e:
            logger.error(f"TensorRT 推理错误: {e}")
            return None

    def unload(self):
        self._context = None
        self._engine = None
        self._input_tensor = None
        self._output_tensor = None

    def warmup_worker_thread(self):
        try:
            import torch
            _ = self._input_tensor.device
            torch.cuda.synchronize()
        except Exception:
            pass


# =========================================================================
# Backend registry
# =========================================================================

_BACKENDS = {
    "ultralytics": ("Ultralytics YOLO", _UltralyticsBackend),
    "onnx":       ("ONNX Runtime", _ONNXBackend),
    "tensorrt":   ("TensorRT", _TensorRTBackend),
}


# =========================================================================
# Main service
# =========================================================================

class AsyncInferenceService:
    """异步推理服务 — 单工作线程 + 策略化后端"""

    def __init__(self):
        # 后端（加载后初始化）
        self._backend: Optional[_BaseBackend] = None
        self._backend_label = "未加载"

        # 运行状态
        self._running = False
        self._conf_threshold = 0.2

        # 最新帧（无锁 — 单生产者/消费者）
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = -1
        self._latest_frame_timestamp = 0.0

        # 线程同步
        self._frame_event = threading.Event()
        self._result_event = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None

        # 结果
        self._latest_result: Optional[InferenceResult] = None
        self._result_callbacks: List[Callable] = []
        self._last_processed_frame_id = -1

        # FPS 统计
        self._prediction_times = deque(maxlen=30)
        self._fps_update_interval = 0.1
        self._last_fps_update = 0.0

    # ---- Properties ----

    @property
    def model_type(self) -> str:
        return self._backend_label

    def get_input_size(self) -> int:
        return self._backend.input_size if self._backend else 640

    # ---- Model lifecycle ----

    def load(self) -> bool:
        """加载模型 — 按 model_type 选择后端"""
        try:
            config = config_service.get_section('ai')
            model_type = config.get('model_type', '').strip().lower()
            model_name = config.get('model_name', '').strip()

            if model_type not in _BACKENDS:
                logger.error(f"model_type 无效: '{model_type}'，须为 onnx / tensorrt / ultralytics")
                return False
            if not model_name:
                logger.error("model_name 未配置")
                return False

            model_path = f"data/{model_type}/{model_name}"
            device_raw = config.get('device', '0')
            device_str = f"cuda:{device_raw}" if str(device_raw).isdigit() else device_raw
            self._conf_threshold = config.get('conf', 0.2)

            config_service.register_callback(self._on_config_changed)

            label, backend_cls = _BACKENDS[model_type]
            backend = backend_cls()
            ok = backend.load(model_path, device_str, self._conf_threshold)

            if ok:
                self._backend = backend
                self._backend_label = label
                logger.info(f"模型加载成功 [{label}]: {model_name}, 输入: {backend.input_size}x{backend.input_size}")
            else:
                logger.error(f"模型加载失败: {model_name}")
                self._backend_label = "未加载"
            return ok

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._backend_label = "未加载"
            return False

    def unload(self):
        """卸载模型"""
        self.stop()
        config_service.unregister_callback(self._on_config_changed)
        if self._backend:
            self._backend.unload()
            self._backend = None
        self._backend_label = "未加载"
        logger.info("模型已卸载")

    def reload(self) -> bool:
        """热重载模型（切换模型时调用）"""
        logger.info("正在热重载模型...")
        self.unload()
        ok = self.load()
        if ok:
            self.start()
        else:
            logger.error("模型热重载失败")
        return ok

    # ---- Start / Stop ----

    def start(self):
        """启动服务"""
        if self._running:
            return
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()
        logger.info("异步推理服务已启动")

    def stop(self):
        """停止服务"""
        self._running = False
        if self._inference_thread and self._inference_thread.is_alive():
            self._frame_event.set()         # 唤醒等待中的线程
            self._inference_thread.join(timeout=3.0)
        image_signal.emit_fps(predict_fps=0)
        logger.info("异步推理服务已停止")

    # ---- Frame submission / result retrieval ----

    def submit_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """提交帧进行推理（无锁写入）"""
        if not self._running or frame is None:
            return False
        self._latest_frame = frame
        self._latest_frame_id = frame_id
        self._latest_frame_timestamp = time.time()
        self._frame_event.set()
        return True

    def get_latest_result(self) -> Optional[InferenceResult]:
        """获取最新结果（无锁读取）"""
        return self._latest_result

    def wait_for_result(self, timeout: float = 0.005) -> Optional[InferenceResult]:
        """等待新推理结果（阻塞调用者线程，建议在 executor 中使用）"""
        if self._result_event.wait(timeout=timeout):
            self._result_event.clear()
            return self._latest_result
        return None

    def add_result_callback(self, callback: Callable):
        """添加结果回调"""
        self._result_callbacks.append(callback)

    # ---- Config callback ----

    def _on_config_changed(self, section: str, updates: dict):
        """配置变更回调 — 保持 _conf_threshold 实时更新"""
        if section == 'ai' and 'conf' in updates:
            self._conf_threshold = updates['conf']

    # ---- Inference worker ----

    def _inference_worker(self):
        """推理工作线程 — 单线程处理帧 + 回调"""
        logger.info("推理工作线程已启动")

        if self._backend:
            self._backend.warmup_worker_thread()

        while self._running:
            try:
                if not self._frame_event.wait(timeout=0.005):
                    continue
                self._frame_event.clear()

                frame_id = self._latest_frame_id
                if frame_id <= self._last_processed_frame_id:
                    continue

                frame = self._latest_frame
                timestamp = self._latest_frame_timestamp
                self._last_processed_frame_id = frame_id

                t0 = time.time()
                detections = self._backend.infer(frame, self._conf_threshold) if self._backend else None
                inference_time = time.time() - t0

                result = InferenceResult(
                    frame_id=frame_id,
                    detections=detections,
                    timestamp=timestamp,
                    processing_time=inference_time,
                    original_frame=frame,
                )

                self._latest_result = result
                self._result_event.set()

                # 更新帧率
                self._prediction_times.append(time.time())
                self._update_prediction_fps()

                # 直接执行回调（不需要队列线程）
                for cb in self._result_callbacks:
                    try:
                        cb(result)
                    except Exception as e:
                        logger.error(f"回调执行失败: {e}")

            except Exception as e:
                logger.error(f"推理工作线程错误: {e}")
                time.sleep(0.005)

    def _update_prediction_fps(self):
        """更新预测帧率"""
        now = time.time()
        if now - self._last_fps_update < self._fps_update_interval:
            return
        count = len(self._prediction_times)
        if count >= 2:
            diff = now - self._prediction_times[0]
            fps = (count - 1) / diff if diff > 0 else 0.0
            image_signal.emit_fps(predict_fps=fps)
        self._last_fps_update = now


# 全局异步推理服务实例
inference_service = AsyncInferenceService()
