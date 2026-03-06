import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, List

import numpy as np
import supervision as sv
from ultralytics import YOLO

from core.services.config_service import config_service
from ui.signals import image_signal

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
    """异步推理服务 - 完全异步化的AI推理引擎"""

    def __init__(self):
        self._model = None
        self._input_size = 640
        self._device_str = None
        self._running = False

        # 最新帧存储
        self._latest_frame = None
        self._latest_frame_id = None
        self._latest_frame_timestamp = None
        self._frame_lock = threading.Lock()

        # 工作线程
        self._inference_thread = None
        self._result_thread = None

        # 结果管理
        self._latest_result: Optional[InferenceResult] = None
        self._result_lock = threading.Lock()
        self._result_callbacks: List[Callable] = []
        
        # 回调队列
        self._callback_queue = queue.Queue(maxsize=100)

        # 预测帧率统计
        self._prediction_times = deque(maxlen=15)
        self._fps_update_interval = 0.05
        self._last_fps_update = 0

        # 帧ID跟踪
        self._last_processed_frame_id = -1

    def load(self) -> bool:
        """加载模型"""
        try:
            config = config_service.get_section('ai')
            model_path = f"data/{config.get('model_name', 'YOLOv8s_apex_teammate_enemy.engine')}"
            device = config.get('device', '0')

            self._device_str = f"cuda:{device}" if device.isdigit() else device

            self._model = YOLO(model_path, task="detect")
            self._input_size = self._get_model_input_size()

            self._warmup()

            logger.info(f"异步推理模型加载成功: {model_path}")
            logger.info(f"模型输入大小: {self._input_size}x{self._input_size}")
            return True
        except Exception as e:
            logger.error(f"异步推理模型加载失败: {e}")
            return False

    def _warmup(self):
        """预热模型"""
        try:
            dummy_image = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            self._model(dummy_image, conf=0.2, device=self._device_str)
            logger.info("异步推理模型预热完成")
        except Exception as e:
            logger.warning(f"异步推理模型预热失败: {e}")

    def _get_model_input_size(self) -> int:
        """获取模型输入大小"""
        try:
            if hasattr(self._model.model, 'yaml') and 'height' in self._model.model.yaml:
                return int(self._model.model.yaml['height'])
            return 640
        except Exception:
            return 640

    def start(self):
        """启动服务"""
        if self._running:
            return

        self._running = True

        # 启动推理线程
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

        # 启动结果处理线程（用于处理回调）
        self._result_thread = threading.Thread(target=self._result_worker, daemon=True)
        self._result_thread.start()

        logger.info("异步推理服务已启动")

    def stop(self):
        """停止服务"""
        self._running = False

        # 等待线程结束
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=3.0)

        if self._result_thread and self._result_thread.is_alive():
            self._result_thread.join(timeout=3.0)

        logger.info("异步推理服务已停止")

    def submit_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """提交帧进行推理"""
        if not self._running or frame is None:
            return False

        try:
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_id = frame_id
                self._latest_frame_timestamp = time.time()
            return True

        except Exception as e:
            logger.error(f"提交帧失败: {e}")
            return False

    def get_latest_result(self) -> Optional[InferenceResult]:
        """获取最新结果（线程安全）"""
        with self._result_lock:
            return self._latest_result

    def add_result_callback(self, callback: Callable):
        """添加结果回调"""
        self._result_callbacks.append(callback)

    def _inference_worker(self):
        """推理工作线程"""
        logger.info("推理工作线程已启动")

        while self._running:
            try:
                # 获取最新帧
                # 减少锁持有时间，只在必要时获取锁
                frame = None
                frame_id = None
                timestamp = None
                has_frame = False
                
                with self._frame_lock:
                    if self._latest_frame is not None and self._latest_frame_id > self._last_processed_frame_id:
                        frame = self._latest_frame
                        frame_id = self._latest_frame_id
                        timestamp = self._latest_frame_timestamp
                        has_frame = True
                
                if not has_frame:
                    time.sleep(0.001)  # 没有帧时稍作等待
                    continue

                # 执行推理
                start_time = time.time()
                detections = self._run_inference(frame)
                inference_time = time.time() - start_time

                # 创建结果
                result = InferenceResult(
                    frame_id=frame_id,
                    detections=detections,
                    timestamp=timestamp,
                    processing_time=inference_time,
                    original_frame=frame if len(self._result_callbacks) > 0 else None
                )

                # 更新最后处理的帧ID（不需要锁）
                self._last_processed_frame_id = frame_id

                # 更新最新结果（只在必要时获取锁）
                with self._result_lock:
                    self._latest_result = result

                # 记录预测时间并更新FPS
                self._prediction_times.append(time.time())
                self._update_prediction_fps(time.time())

                # 执行回调
                self._notify_callbacks(result)

            except Exception as e:
                logger.error(f"推理工作线程错误: {e}")
                time.sleep(0.1)

    def _result_worker(self):
        """结果处理工作线程"""
        logger.info("结果处理工作线程已启动")

        while self._running:
            try:
                # 处理回调队列
                try:
                    result = self._callback_queue.get(timeout=0.01)
                    self._process_callbacks(result)
                    self._callback_queue.task_done()
                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(f"结果处理工作线程错误: {e}")

    def _run_inference(self, image: np.ndarray):
        """运行推理"""
        if self._model is None:
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
                verbose=False,
                stream=True,
                agnostic_nms=True
            )

            return self._postprocess(results)
        except Exception as e:
            logger.error(f"异步推理错误: {e}")
            return None

    def _postprocess(self, outputs):
        """后处理"""
        try:
            # 直接获取第一个结果，避免不必要的循环
            result = next(outputs, None)
            if result:
                return sv.Detections.from_ultralytics(result)
            return None
        except Exception as e:
            logger.error(f"异步后处理错误: {e}")
            return None

    def _process_callbacks(self, result: InferenceResult):
        """处理回调"""
        for callback in self._result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"执行回调失败: {e}")

    def _notify_callbacks(self, result: InferenceResult):
        """通知回调"""
        try:
            self._callback_queue.put_nowait(result)
        except queue.Full:
            pass

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
            if diff > 0:
                fps = (count - 1) / diff
                image_signal.predict_fps.emit(fps)
        else:
            image_signal.predict_fps.emit(0.0)

        self._last_fps_update = current_time


# 全局异步推理服务实例
inference_service = AsyncInferenceService()
