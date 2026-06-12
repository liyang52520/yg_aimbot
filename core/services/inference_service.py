import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, List

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


class AsyncInferenceService:
    """异步推理服务 — 单工作线程 + 原地回调"""

    def __init__(self):
        self._model = None
        self._onnx_session = None
        self._input_size = 640
        self._device_str = None
        self._is_onnx = False
        self._running = False

        # 最新帧存储（无锁 — 单一生产者/消费者模式）
        self._latest_frame = None
        self._latest_frame_id = -1
        self._latest_frame_timestamp = 0.0

        # 线程同步
        self._frame_event = threading.Event()
        self._wake_event = threading.Event()

        # 推理线程
        self._inference_thread: Optional[threading.Thread] = None

        # 结果管理（无锁读取）
        self._latest_result: Optional[InferenceResult] = None

        # 结果回调
        self._result_callbacks: List[Callable] = []

        # 预测帧率统计
        self._prediction_times = deque(maxlen=30)
        self._fps_update_interval = 0.1
        self._last_fps_update = 0

        # 帧ID跟踪
        self._last_processed_frame_id = -1

        # 配置缓存（减少 get_section 调用）
        self._conf_threshold = 0.2

    def load(self) -> bool:
        """加载模型 — 支持 .pt/.engine (ultralytics) 和 .onnx (ONNX Runtime)"""
        try:
            config = config_service.get_section('ai')
            model_name = config.get('model_name', 'YOLOv8s_apex_teammate_enemy.engine')
            model_path = f"data/{model_name}"
            device = config.get('device', '0')
            self._device_str = f"cuda:{device}" if device.isdigit() else device
            self._conf_threshold = config.get('conf', 0.2)

            if model_name.endswith('.onnx'):
                self._is_onnx = True
                return self._load_onnx(model_path)
            else:
                self._is_onnx = False
                return self._load_ultralytics(model_path)

        except Exception as e:
            logger.error(f"异步推理模型加载失败: {e}")
            return False

    def _load_ultralytics(self, model_path: str) -> bool:
        """加载 ultralytics 模型"""
        try:
            self._model = YOLO(model_path, task="detect")
            self._input_size = self._get_model_input_size()
            self._warmup()
            logger.info(f"Ultralytics 模型加载成功: {model_path}, 输入: {self._input_size}x{self._input_size}")
            return True
        except Exception as e:
            logger.error(f"Ultralytics 模型加载失败: {e}")
            return False

    def _load_onnx(self, model_path: str) -> bool:
        """加载 ONNX 模型"""
        try:
            # 仅使用可用的 provider，避免 CUDA 未安装时的警告
            available = ort.get_available_providers()
            preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in preferred if p in available] or ['CPUExecutionProvider']
            logger.info(f"ONNX providers: {providers}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._onnx_session = ort.InferenceSession(model_path, sess_options, providers=providers)

            # 解析输入尺寸
            inp = self._onnx_session.get_inputs()[0]
            self._input_size = inp.shape[2]  # NCHW → H=320
            logger.info(f"ONNX 输入: {inp.name} {inp.shape}")

            self._warmup_onnx()
            logger.info(f"ONNX 模型加载成功: {model_path}, 输入: {self._input_size}x{self._input_size}")
            return True
        except Exception as e:
            logger.error(f"ONNX 模型加载失败: {e}")
            return False

    def _warmup(self):
        """预热 ultralytics 模型"""
        try:
            dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            self._model(dummy, conf=self._conf_threshold, device=self._device_str)
            logger.info("异步推理模型预热完成")
        except Exception as e:
            logger.warning(f"异步推理模型预热失败: {e}")

    def _warmup_onnx(self):
        """预热 ONNX 模型"""
        try:
            dummy = np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32)
            self._onnx_session.run(None, {self._onnx_session.get_inputs()[0].name: dummy})
            logger.info("ONNX 模型预热完成")
        except Exception as e:
            logger.warning(f"ONNX 模型预热失败: {e}")

    def _get_model_input_size(self) -> int:
        """获取模型输入大小"""
        try:
            if hasattr(self._model.model, 'yaml') and 'height' in self._model.model.yaml:
                return int(self._model.model.yaml['height'])
        except Exception:
            pass
        return 640

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
        self._wake_event.set()  # 唤醒线程
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=3.0)
        logger.info("异步推理服务已停止")

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

    def add_result_callback(self, callback: Callable):
        """添加结果回调"""
        self._result_callbacks.append(callback)

    def _inference_worker(self):
        """推理工作线程 — 单线程处理帧 + 回调"""
        logger.info("推理工作线程已启动")
        conf = self._conf_threshold
        device = self._device_str

        while self._running:
            try:
                # 等待新帧（事件唤醒，无轮询）
                if not self._frame_event.wait(timeout=0.005):
                    continue
                self._frame_event.clear()

                frame_id = self._latest_frame_id
                if frame_id <= self._last_processed_frame_id:
                    continue

                frame = self._latest_frame
                timestamp = self._latest_frame_timestamp
                self._last_processed_frame_id = frame_id

                # 执行推理
                start_time = time.time()
                detections = self._run_inference(frame, conf, device)
                inference_time = time.time() - start_time

                # 创建结果
                result = InferenceResult(
                    frame_id=frame_id,
                    detections=detections,
                    timestamp=timestamp,
                    processing_time=inference_time,
                    original_frame=frame if self._result_callbacks else None
                )

                # 更新最新结果（单赋值，线程安全）
                self._latest_result = result

                # 更新帧率
                self._prediction_times.append(time.time())
                self._update_prediction_fps(time.time())

                # 直接执行回调（不需要队列线程）
                for cb in self._result_callbacks:
                    try:
                        cb(result)
                    except Exception as e:
                        logger.error(f"回调执行失败: {e}")

            except Exception as e:
                logger.error(f"推理工作线程错误: {e}")
                time.sleep(0.005)

    def _run_inference(self, image: np.ndarray, conf: float, device: str):
        """运行推理 — 自动选择 ONNX 或 ultralytics 路径"""
        if self._is_onnx:
            return self._run_onnx_inference(image, conf)
        else:
            return self._run_ultralytics_inference(image, conf, device)

    def _run_ultralytics_inference(self, image: np.ndarray, conf: float, device: str):
        """ultralytics YOLO 推理"""
        if self._model is None:
            return None
        try:
            results = self._model(
                image,
                conf=conf,
                device=device,
                half=True,
                max_det=3,
                verbose=False,
                stream=True,
                agnostic_nms=True
            )
            result = next(results, None)
            if result:
                return sv.Detections.from_ultralytics(result)
            return None
        except Exception as e:
            logger.error(f"Ultralytics 推理错误: {e}")
            return None

    def _run_onnx_inference(self, image: np.ndarray, conf: float):
        """ONNX Runtime 推理"""
        if self._onnx_session is None:
            return None
        try:
            # 预处理: resize → BGR→RGB → HWC→CHW → normalize
            input_tensor = self._preprocess_onnx(image, self._input_size)

            # 推理
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: input_tensor})

            return self._postprocess_onnx(outputs[0], conf, image.shape[1], image.shape[0])
        except Exception as e:
            logger.error(f"ONNX 推理错误: {e}")
            return None

    @staticmethod
    def _preprocess_onnx(image: np.ndarray, input_size: int) -> np.ndarray:
        """ONNX 预处理: HWC BGR uint8 → NCHW RGB float32 [0,1]"""
        h, w = image.shape[:2]
        # 保持宽高比的 resize + pad
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充到正方形
        top = (input_size - new_h) // 2
        left = (input_size - new_w) // 2
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized

        # BGR → RGB, HWC → CHW, 归一化
        tensor = canvas.astype(np.float32) / 255.0
        tensor = tensor[:, :, ::-1]  # BGR → RGB
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW
        tensor = np.expand_dims(tensor, axis=0)  # → NCHW
        return tensor

    def _postprocess_onnx(self, raw_output: np.ndarray, conf_thresh: float,
                          orig_w: int, orig_h: int) -> Optional[sv.Detections]:
        """ONNX 后处理: [1, N, 8] → sv.Detections"""
        # 输出格式: [cx, cy, w, h, obj_conf, cls0, cls1, cls2]
        dets = raw_output[0]  # (N, 8)
        boxes = dets[:, :4]   # cx, cy, w, h (在 320x320 坐标空间)
        obj_conf = dets[:, 4]
        cls_scores = dets[:, 5:]  # softmax 后的类别概率

        # 最终置信度 = objectness × max(class_prob)
        max_cls = cls_scores.max(axis=1)
        confidences = obj_conf * max_cls

        # 阈值过滤
        keep = confidences > conf_thresh
        if not keep.any():
            return None

        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = cls_scores[keep].argmax(axis=1).astype(int)

        # cxcywh → xyxy
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        half_w, half_h = w / 2, h / 2
        xyxy = np.column_stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h])

        # 去除填充区域的偏移 (与 _preprocess_onnx 中的 pad 对应)
        scale = min(self._input_size / orig_h, self._input_size / orig_w)
        pad_x = (self._input_size - orig_w * scale) / 2
        pad_y = (self._input_size - orig_h * scale) / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

        # 裁剪到图像边界
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids
        )

        # NMS
        detections = detections.with_nms(threshold=0.5)
        return detections

    def unload(self):
        """卸载模型"""
        self.stop()
        self._model = None
        self._onnx_session = None
        logger.info("异步推理模型已卸载")

    def get_input_size(self) -> int:
        """获取模型输入大小"""
        return self._input_size

    def _update_prediction_fps(self, current_time: float):
        """更新预测帧率"""
        if current_time - self._last_fps_update < self._fps_update_interval:
            return
        count = len(self._prediction_times)
        if count >= 2:
            diff = current_time - self._prediction_times[0]
            fps = (count - 1) / diff if diff > 0 else 0.0
            image_signal.emit_fps(predict_fps=fps)
        self._last_fps_update = current_time


# 全局异步推理服务实例
inference_service = AsyncInferenceService()
