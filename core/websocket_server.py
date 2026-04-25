"""
WebSocket 服务器 - 用于 Electron UI 和 Python 核心之间的实时通信
"""
import asyncio
import base64
import json
import logging
import threading
import time
from typing import Set, Optional
import numpy as np

try:
    import websockets
    import websockets.exceptions
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: websockets not installed. Real-time communication disabled.")

logger = logging.getLogger(__name__)


class WebSocketServer:
    """WebSocket 服务器 - 处理 Electron 连接和数据传输"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.server = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        
        # 消息队列
        self._message_queue = []
        self._queue_lock = threading.Lock()
        
        # 最新数据缓存
        self._last_capture_fps = 0.0
        self._last_predict_fps = 0.0
        self._last_frame = None
        self._last_frame_time = 0
        
    def start(self):
        """启动 WebSocket 服务器"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available. Install websockets: pip install websockets")
            return
            
        if self._running:
            return
            
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket 服务器启动中: ws://{self.host}:{self.port}")
        
    def stop(self):
        """停止 WebSocket 服务器"""
        self._running = False
        self._stop_event.set()
        
        if self._loop:
            try:
                # 在事件循环中关闭服务器
                future = asyncio.run_coroutine_threadsafe(
                    self._close_server(),
                    self._loop
                )
                try:
                    future.result(timeout=2.0)
                except:
                    pass
            except:
                pass
                
        if self._thread:
            try:
                self._thread.join(timeout=2.0)
            except:
                pass
        logger.info("WebSocket 服务器已停止")
        
    async def _close_server(self):
        """关闭服务器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
    def _run_server(self):
        """在独立线程中运行服务器"""
        # 创建新的事件循环
        self._loop = asyncio.new_event_loop()
        
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"WebSocket 服务器错误: {e}")
        finally:
            try:
                # 清理所有任务
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    
                self._loop.close()
            except:
                pass
            
    async def _serve(self):
        """启动 WebSocket 服务"""
        try:
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            )
            logger.info(f"WebSocket 服务器已启动: ws://{self.host}:{self.port}")
            
            # 保持服务器运行
            while self._running and not self._stop_event.is_set():
                await asyncio.sleep(0.1)
                
            # 关闭服务器
            self.server.close()
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"WebSocket 服务器错误: {e}")
            
    async def _handle_client(self, websocket):
        """处理客户端连接"""
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address}"
        logger.info(f"Electron 客户端已连接: {client_info}")
        
        try:
            # 发送当前配置给新客户端
            await self._send_current_config(websocket)
            
            # 保持连接并处理消息
            while self._running:
                try:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=1.0
                    )
                    try:
                        data = json.loads(message)
                        await self._handle_message(websocket, data)
                    except json.JSONDecodeError:
                        logger.error(f"收到无效的 JSON: {message}")
                except asyncio.TimeoutError:
                    # 超时检查是否仍在运行
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Electron 客户端断开连接: {client_info}")
        except Exception as e:
            logger.error(f"处理客户端错误: {e}")
        finally:
            self.clients.discard(websocket)
            
    async def _handle_message(self, websocket, data: dict):
        """处理客户端消息"""
        command = data.get('command')
        
        if command == 'get-config':
            await self._send_current_config(websocket)
        elif command == 'save-config':
            await self._handle_save_config(data.get('data', {}))
        elif command == 'apply-config':
            await self._handle_apply_config(data.get('data', {}))
        elif command == 'scan-models':
            await self._handle_scan_models(websocket)
        else:
            logger.warning(f"未知命令: {command}")
            
    async def _send_current_config(self, websocket):
        """发送当前配置给客户端"""
        try:
            from core.services.config_service import config_service
            
            config = {
                'ai': config_service.get_section('ai'),
                'capture': config_service.get_section('capture'),
                'aim': config_service.get_section('aim'),
                'mouse': config_service.get_section('mouse')
            }
            
            await websocket.send(json.dumps({
                'type': 'config',
                'data': config
            }))
        except Exception as e:
            logger.error(f"发送配置失败: {e}")
            
    async def _handle_save_config(self, config: dict):
        """处理保存配置"""
        try:
            from core.services.config_service import config_service
            
            if 'ai' in config:
                config_service.update_section('ai', config['ai'])
            if 'capture' in config:
                config_service.update_section('capture', config['capture'])
            if 'aim' in config:
                config_service.update_section('aim', config['aim'])
            if 'mouse' in config:
                config_service.update_section('mouse', config['mouse'])
                
            config_service.save()
            logger.info("配置已保存")
            
            # 广播给所有客户端
            await self.broadcast({'type': 'config-saved', 'data': {'success': True}})
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            await self.broadcast({'type': 'config-saved', 'data': {'success': False, 'error': str(e)}})
            
    async def _handle_apply_config(self, config: dict):
        """处理应用配置"""
        try:
            from core.services.config_service import config_service
            
            if 'ai' in config:
                config_service.update_section('ai', config['ai'])
            if 'capture' in config:
                config_service.update_section('capture', config['capture'])
            if 'aim' in config:
                config_service.update_section('aim', config['aim'])
            if 'mouse' in config:
                config_service.update_section('mouse', config['mouse'])
                
            logger.info("配置已应用到内存")
        except Exception as e:
            logger.error(f"应用配置失败: {e}")
            
    async def _handle_scan_models(self, websocket):
        """处理扫描模型文件"""
        import os
        
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        extensions = ['.pt', '.onnx', '.engine']
        
        models = []
        try:
            if os.path.exists(models_dir) and os.path.isdir(models_dir):
                for file in os.listdir(models_dir):
                    if any(file.endswith(ext) for ext in extensions):
                        models.append(file)
        except Exception as e:
            logger.error(f"扫描模型失败: {e}")
            
        if not models:
            models = ['YOLOv8s_apex_teammate_enemy.engine']
            
        await websocket.send(json.dumps({
            'type': 'models',
            'data': models
        }))
        
    def broadcast_sync(self, message: dict):
        """同步广播消息给所有客户端（从其他线程调用）"""
        if not self._running or not self.clients:
            return
            
        # 添加到队列
        with self._queue_lock:
            self._message_queue.append(message)
        
        # 如果事件循环正在运行，发送消息
        if self._loop and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._process_queue(),
                    self._loop
                )
            except:
                pass
                
    async def _process_queue(self):
        """处理消息队列"""
        messages = []
        with self._queue_lock:
            messages = self._message_queue[:]
            self._message_queue = []
        
        for message in messages:
            await self.broadcast(message)
            
    async def broadcast(self, message: dict):
        """广播消息给所有客户端"""
        if not self.clients:
            return
            
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                disconnected.add(client)
                
        # 清理断开的客户端
        self.clients -= disconnected
        
    # ===== 数据推送接口 =====
    
    def emit_fps(self, capture_fps: float = None, predict_fps: float = None):
        """发送 FPS 数据"""
        if capture_fps is not None:
            self._last_capture_fps = capture_fps
        if predict_fps is not None:
            self._last_predict_fps = predict_fps
            
        self.broadcast_sync({
            'type': 'fps',
            'data': {
                'capture': self._last_capture_fps,
                'predict': self._last_predict_fps
            }
        })
        
    def emit_video_frame(self, frame: np.ndarray):
        """发送视频帧（base64 编码）"""
        # 限制帧率，避免过度传输
        current_time = time.time()
        if current_time - self._last_frame_time < 0.033:  # 最多 30fps
            return
        self._last_frame_time = current_time
        
        try:
            # 压缩并编码为 base64
            import cv2
            
            # 缩小尺寸以减少传输量
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                
            # 压缩为 JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            self.broadcast_sync({
                'type': 'video-frame',
                'data': f'data:image/jpeg;base64,{img_base64}'
            })
        except Exception as e:
            logger.error(f"编码视频帧失败: {e}")
            
    def emit_log(self, message: str):
        """发送日志消息"""
        self.broadcast_sync({
            'type': 'log',
            'data': message
        })
        
    def emit_detections(self, detections):
        """发送检测结果"""
        try:
            # 简化检测结果
            detection_data = []
            if hasattr(detections, 'xyxy'):
                for i, box in enumerate(detections.xyxy):
                    detection_data.append({
                        'box': box.tolist() if hasattr(box, 'tolist') else list(box),
                        'confidence': float(detections.confidence[i]) if hasattr(detections, 'confidence') and i < len(detections.confidence) else 0.0,
                        'class_id': int(detections.class_id[i]) if hasattr(detections, 'class_id') and i < len(detections.class_id) else 0
                    })
                    
            self.broadcast_sync({
                'type': 'detections',
                'data': detection_data
            })
        except Exception as e:
            logger.error(f"发送检测结果失败: {e}")


# 全局 WebSocket 服务器实例
websocket_server = WebSocketServer()
