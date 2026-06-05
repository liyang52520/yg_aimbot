import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, List

import numpy as np
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
        self._input_size = 640
        self._device_str = None
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
        """加载模型"""
        try:
            config = config_service.get_section('ai')
            model_path = f"data/{config.get('model_name', 'YOLOv8s_apex_teammate_enemy.engine')}"
            device = config.get('device', '0')
            self._device_str = f"cuda:{device}" if device.isdigit() else device
            self._conf_threshold = config.get('conf', 0.2)

            self._model = YOLO(model_path, task="detect")
            self._input_size = self._get_model_input_size()
            self._warmup()

            logger.info(f"异步推理模型加载成功: {model_path}, 输入: {self._input_size}x{self._input_size}")
            return True
        except Exception as e:
            logger.error(f"异步推理模型加载失败: {e}")
            return False

    def _warmup(self):
        """预热模型"""
        try:
            dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            self._model(dummy, conf=self._conf_threshold, device=self._device_str)
            logger.info("异步推理模型预热完成")
        except Exception as e:
            logger.warning(f"异步推理模型预热失败: {e}")

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
        """运行推理"""
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
            return self._postprocess(results)
        except Exception as e:
            logger.error(f"推理错误: {e}")
            return None

    @staticmethod
    def _postprocess(outputs):
        """后处理"""
        try:
            result = next(outputs, None)
            if result:
                return sv.Detections.from_ultralytics(result)
            return None
        except Exception as e:
            logger.error(f"后处理错误: {e}")
            return None

    def unload(self):
        """卸载模型"""
        self.stop()
        self._model = None
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
