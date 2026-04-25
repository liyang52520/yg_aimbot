"""
UI桥接模块 - 用于替代PyQt信号，支持Electron UI通信
"""
import logging
from typing import Callable, List, Any

logger = logging.getLogger(__name__)


class Signal:
    """简单的信号类，替代PyQt信号"""
    
    def __init__(self, name: str):
        self.name = name
        self._callbacks: List[Callable] = []
        self._enabled = True
    
    def connect(self, callback: Callable):
        """连接回调函数"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def disconnect(self, callback: Callable):
        """断开回调函数"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def emit(self, *args, **kwargs):
        """触发信号"""
        if not self._enabled:
            return
        
        for callback in self._callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Signal {self.name} callback error: {e}")
    
    def set_enabled(self, enabled: bool):
        """设置信号是否启用"""
        self._enabled = enabled


class UIBridge:
    """UI桥接类 - 管理所有UI信号，同时支持 WebSocket 通信"""
    
    def __init__(self):
        self.capture_fps = Signal("capture_fps")
        self.predict_fps = Signal("predict_fps")
        self.image = Signal("image")
        self.detection_result = Signal("detection_result")
        self.clear_predict_fps = Signal("clear_predict_fps")
        self.log = Signal("log")
        
        # WebSocket 服务器引用（在 run.py 中初始化）
        self._websocket = None
        
    def set_websocket(self, websocket_server):
        """设置 WebSocket 服务器"""
        self._websocket = websocket_server
        
    def emit_fps(self, capture_fps: float = None, predict_fps: float = None):
        """发送FPS数据"""
        if capture_fps is not None:
            self.capture_fps.emit(capture_fps)
        if predict_fps is not None:
            self.predict_fps.emit(predict_fps)
            
        # 同时发送到 WebSocket
        if self._websocket:
            self._websocket.emit_fps(capture_fps, predict_fps)
    
    def emit_video_frame(self, frame):
        """发送视频帧"""
        self.image.emit(frame)
        
        # 同时发送到 WebSocket
        if self._websocket:
            self._websocket.emit_video_frame(frame)
    
    def emit_detections(self, detections):
        """发送检测结果"""
        self.detection_result.emit(detections)
        
        # 同时发送到 WebSocket
        if self._websocket:
            self._websocket.emit_detections(detections)
    
    def emit_log(self, message: str):
        """发送日志消息"""
        self.log.emit(message)
        
        # 同时发送到 WebSocket
        if self._websocket:
            self._websocket.emit_log(message)


# 全局UI桥接实例
ui_bridge = UIBridge()
