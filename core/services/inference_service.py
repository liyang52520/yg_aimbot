import logging
import threading
import time
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import queue
import asyncio

from core.services.config_service import config_service

logger = logging.getLogger(__name__)


@dataclass
class FrameBuffer:
    """帧缓冲区"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    submitted: bool = False
    processed: bool = False


@dataclass
class InferenceResult:
    """推理结果"""
    frame_id: int
    detections: Any
    timestamp: float
    processing_time: float
    original_frame: Optional[np.ndarray] = None


class DoubleBuffer:
    """双缓冲管理器"""
    
    def __init__(self, size: int = 3):
        self._size = size
        self._buffers: List[Optional[FrameBuffer]] = [None] * size
        self._current_index = 0
        self._lock = threading.Lock()
        self._frame_counter = 0
    
    def add_frame(self, frame: np.ndarray) -> Optional[int]:
        """添加帧到缓冲区"""
        with self._lock:
            self._frame_counter += 1
            frame_id = self._frame_counter
            
            # 找到最旧的未提交帧或空槽位
            target_index = -1
            oldest_time = float('inf')
            
            for i, buffer in enumerate(self._buffers):
                if buffer is None:
                    target_index = i
                    break
                elif not buffer.submitted and buffer.timestamp < oldest_time:
                    oldest_time = buffer.timestamp
                    target_index = i
            
            if target_index == -1:
                # 所有槽位都被占用，丢弃最旧的已处理帧
                for i, buffer in enumerate(self._buffers):
                    if buffer and buffer.processed:
                        target_index = i
                        break
            
            if target_index != -1:
                self._buffers[target_index] = FrameBuffer(
                    frame=frame.copy(),
                    frame_id=frame_id,
                    timestamp=time.time()
                )
                return frame_id
            
            return None
    
    def get_frame_to_process(self) -> Optional[FrameBuffer]:
        """获取待处理的帧"""
        with self._lock:
            # 找到最新的未提交帧
            best_frame = None
            best_index = -1
            
            for i, buffer in enumerate(self._buffers):
                if buffer and not buffer.submitted:
                    if best_frame is None or buffer.timestamp > best_frame.timestamp:
                        best_frame = buffer
                        best_index = i
            
            if best_frame and best_index != -1:
                self._buffers[best_index].submitted = True
                return best_frame
            
            return None
    
    def mark_processed(self, frame_id: int):
        """标记帧已处理"""
        with self._lock:
            for buffer in self._buffers:
                if buffer and buffer.frame_id == frame_id:
                    buffer.processed = True
                    break
    
    def get_latest_frame_id(self) -> Optional[int]:
        """获取最新帧ID"""
        with self._lock:
            latest_frame = None
            for buffer in self._buffers:
                if buffer and (latest_frame is None or buffer.timestamp > latest_frame.timestamp):
                    latest_frame = buffer
            
            return latest_frame.frame_id if latest_frame else None


class AsyncInferenceService:
    """异步推理服务 - 完全异步化的AI推理引擎"""

    def __init__(self):
        self._model = None
        self._input_size = 640
        self._device_str = None
        self._running = False
        
        # 双缓冲机制
        self._double_buffer = DoubleBuffer(size=3)
        
        # 工作线程
        self._inference_thread = None
        self._result_thread = None
        
        # 结果管理
        self._latest_result: Optional[InferenceResult] = None
        self._result_lock = threading.Lock()
        self._result_callbacks: List[Callable] = []
        
        # 性能统计
        self._stats = {
            'frame_counter': 0,
            'dropped_frames': 0,
            'processed_frames': 0,
            'inference_times': deque(maxlen=100),
            'queue_waits': deque(maxlen=100),
            'last_inference_time': 0
        }
        
        # 智能降频
        self._enable_adaptive_fps = True
        self._target_inference_time = 0.033  # 目标推理时间 33ms (30 FPS)
        self._current_skip_rate = 1  # 当前跳帧率
        self._frame_skip_counter = 0

    def load(self) -> bool:
        """加载模型"""
        try:
            from ultralytics import YOLO
            import supervision as sv

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
            self._model(dummy_image, conf=0.2, device=self._device_str, verbose=False)
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
        
        # 启动结果处理线程
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

    def submit_frame(self, frame: np.ndarray) -> bool:
        """提交帧进行推理"""
        if not self._running or frame is None:
            return False

        # 智能帧跳过逻辑
        if self._enable_adaptive_fps and self._current_skip_rate > 1:
            self._frame_skip_counter += 1
            if self._frame_skip_counter < self._current_skip_rate:
                return False  # 跳过此帧
            self._frame_skip_counter = 0

        try:
            frame_id = self._double_buffer.add_frame(frame)
            if frame_id is None:
                self._stats['dropped_frames'] += 1
                return False
            
            self._stats['frame_counter'] += 1
            return True
            
        except Exception as e:
            logger.error(f"提交帧失败: {e}")
            self._stats['dropped_frames'] += 1
            return False

    def get_latest_result(self) -> Optional[InferenceResult]:
        """获取最新结果（线程安全）"""
        with self._result_lock:
            return self._latest_result

    def add_result_callback(self, callback: Callable):
        """添加结果回调"""
        self._result_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._result_lock:
            stats = self._stats.copy()
            stats['avg_inference_time'] = (
                sum(self._stats['inference_times']) / len(self._stats['inference_times'])
                if self._stats['inference_times'] else 0
            )
            stats['current_skip_rate'] = self._current_skip_rate
            return stats

    def _inference_worker(self):
        """推理工作线程"""
        logger.info("推理工作线程已启动")
        
        while self._running:
            try:
                # 获取待处理帧
                frame_buffer = self._double_buffer.get_frame_to_process()
                if frame_buffer is None:
                    time.sleep(0.001)  # 没有帧时稍作等待
                    continue
                
                # 执行推理
                start_time = time.time()
                detections = self._run_inference(frame_buffer.frame)
                inference_time = time.time() - start_time
                
                # 创建结果
                result = InferenceResult(
                    frame_id=frame_buffer.frame_id,
                    detections=detections,
                    timestamp=frame_buffer.timestamp,
                    processing_time=inference_time,
                    original_frame=frame_buffer.frame if len(self._result_callbacks) > 0 else None
                )
                
                # 更新统计
                self._stats['processed_frames'] += 1
                self._stats['last_inference_time'] = inference_time
                self._stats['inference_times'].append(inference_time)
                
                # 自适应FPS调整
                if self._enable_adaptive_fps:
                    self._adapt_fps(inference_time)
                
                # 标记帧已处理
                self._double_buffer.mark_processed(frame_buffer.frame_id)
                
                # 更新最新结果
                with self._result_lock:
                    self._latest_result = result
                
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
                # 这里可以添加额外的结果处理逻辑
                time.sleep(0.1)  # 降低CPU占用
                
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
                verbose=False
            )

            return self._postprocess(results)
        except Exception as e:
            logger.error(f"异步推理错误: {e}")
            return None

    def _postprocess(self, outputs):
        """后处理"""
        try:
            import supervision as sv
            for result in outputs:
                return sv.Detections.from_ultralytics(result)
            return None
        except Exception as e:
            logger.error(f"异步后处理错误: {e}")
            return None

    def _adapt_fps(self, inference_time: float):
        """自适应FPS调整"""
        # 如果推理时间超过目标时间，增加跳帧率
        if inference_time > self._target_inference_time:
            if self._current_skip_rate < 5:  # 最大跳帧率限制
                self._current_skip_rate += 1
                logger.debug(f"增加跳帧率到: {self._current_skip_rate}")
        elif inference_time < self._target_inference_time * 0.7:  # 如果推理时间远低于目标
            if self._current_skip_rate > 1:
                self._current_skip_rate -= 1
                logger.debug(f"减少跳帧率到: {self._current_skip_rate}")

    def _notify_callbacks(self, result: InferenceResult):
        """通知回调"""
        for callback in self._result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"执行回调失败: {e}")

    def unload(self):
        """卸载模型"""
        self.stop()
        self._model = None
        logger.info("异步推理模型已卸载")

    def get_input_size(self) -> int:
        """获取模型输入大小"""
        return self._input_size

    def set_adaptive_fps(self, enabled: bool, target_time: float = 0.033):
        """设置自适应FPS"""
        self._enable_adaptive_fps = enabled
        self._target_inference_time = target_time
        logger.info(f"自适应FPS: {'启用' if enabled else '禁用'}, 目标时间: {target_time}s")


# 全局异步推理服务实例
inference_service = AsyncInferenceService()