import asyncio
import logging
import threading
import time

import numpy as np
import win32api

from core.services.aim_service import aim_service
from core.services.capture_service import capture_service
from core.services.config_service import config_service
from core.services.inference_service import inference_service
from core.services.tracker_service import tracker_service
from core.ui_bridge import ui_bridge as image_signal

logger = logging.getLogger(__name__)


class Aimbot:
    """自瞄应用主类"""

    def __init__(self):
        self._running = False
        self._stop_event = threading.Event()

        self._toggle_enabled = False
        self._key_states = {}

        self._config_check_interval = 0.2
        self._last_config_check = 0

        self._cached_hotkey_codes = []
        self._last_hotkeys = ''
        self._cache_hotkey_codes()

        self._last_model_name = config_service.get('ai', 'model_name', '')

        self._was_inferring = False
        self._need_prediction = False
        self._last_fps_zero_emit = 0.0

        # 缓存热路径配置值，避免 200Hz 路径重复读锁
        self._cached_ai_debug = config_service.get('capture', 'ai_debug', False)
        self._cached_auto = config_service.get('aim', 'auto', False)
        self._cached_mode = config_service.get('aim', 'mode', 'hold')
        config_service.register_callback(self._on_config_changed)

        # 事件驱动：帧提交去重
        self._last_submitted_frame_id = -1

        # 事件驱动：限速热键检查
        self._last_hotkey_check = 0.0
        self._hotkey_check_interval = 0.005  # 200Hz

    def _on_config_changed(self, section: str, updates: dict):
        """配置变更回调 — 更新缓存的热路径值"""
        if section == 'capture' and 'ai_debug' in updates:
            self._cached_ai_debug = updates['ai_debug']
        if section == 'aim':
            if 'auto' in updates:
                self._cached_auto = updates['auto']
            if 'mode' in updates:
                self._cached_mode = updates['mode']

    def _cache_hotkey_codes(self):
        """缓存热键代码"""
        from core.buttons import Buttons
        hotkeys = config_service.get('aim', 'hotkeys', 'X1MouseButton,X2MouseButton')
        self._cached_hotkey_codes = [
            Buttons.KEY_CODES.get(key.strip())
            for key in hotkeys.split(',')
            if Buttons.KEY_CODES.get(key.strip())
        ]

    def start(self):
        """启动应用"""
        if self._running:
            return

        logger.info("启动自瞄应用...")

        # 加载异步推理模型
        logger.info("使用异步推理模式")
        if not inference_service.load():
            logger.error("异步模型加载失败，无法启动")
            return False

        # 启动异步推理服务
        inference_service.start()

        # 获取模型输入大小
        model_input_size = inference_service.get_input_size()
        aim_service.set_model_input_size(model_input_size)
        logger.info(f"模型输入大小: {model_input_size}x{model_input_size}")

        # 根据模型输入大小自动设置捕获窗口
        config_service.update_section('capture', {
            'window_width': model_input_size,
            'window_height': model_input_size
        }, notify=True)

        capture_service.start()

        self._running = True
        self._last_config_check = time.time()

        logger.info("自瞄应用已启动")
        return True

    async def run(self):
        """运行主循环"""
        if not self.start():
            return

        try:
            await self._main_loop()
        except Exception as e:
            logger.error(f"主循环异常: {e}")
        finally:
            self.stop()

    async def _main_loop(self):
        """事件驱动主循环 — 仅在需要推理时阻塞等待结果，空闲时休眠"""
        loop = asyncio.get_event_loop()

        while self._running and not self._stop_event.is_set():
            try:
                now = time.time()

                # 限速热键检查 (200Hz)
                if now - self._last_hotkey_check >= self._hotkey_check_interval:
                    self._need_prediction = self._check_need_prediction()
                    self._last_hotkey_check = now

                is_inferring = self._need_prediction or self._cached_ai_debug

                if is_inferring:
                    # 1. 提交最新帧到推理（非阻塞）
                    frame, frame_id = capture_service.get_frame()
                    need_wait = False
                    if frame is not None and frame_id != self._last_submitted_frame_id:
                        inference_service.submit_frame(frame, frame_id)
                        self._last_submitted_frame_id = frame_id
                        need_wait = True

                    # 2. 等待推理结果 — 仅在有新帧提交时才阻塞
                    if need_wait:
                        result = await loop.run_in_executor(
                            None, inference_service.wait_for_result, 0.005
                        )
                    else:
                        # 无新帧 → 直接取最新结果，避免 5ms 空等
                        result = inference_service.get_latest_result()

                    now = time.time()

                    # 3. 再次检查热键（200Hz）
                    if now - self._last_hotkey_check >= self._hotkey_check_interval:
                        self._need_prediction = self._check_need_prediction()
                        self._last_hotkey_check = now
                    is_inferring = self._need_prediction or self._cached_ai_debug

                    # 4. 处理推理结果
                    if result and result.detections is not None:
                        if self._need_prediction:
                            tracker_service.update(result.detections, result.frame_id)

                    # ai_debug 模式：始终发送检测结果（可能为空）和视频帧
                    if self._cached_ai_debug and result:
                        image_signal.emit_detections(result.detections)
                        if result.original_frame is not None:
                            image_signal.emit_video_frame(result.original_frame)
                else:
                    # 无需推理，短暂休眠避免 CPU 空转
                    await asyncio.sleep(0.005)
                    # 竞态修复：推理线程可能仍在处理已提交的最后一帧，
                    # 完成后 _update_prediction_fps 会发射非零值覆盖归零结果。
                    # 这里定时重新归零，确保前端显示正确。
                    now = time.time()
                    if now - self._last_fps_zero_emit >= 0.2:  # 5Hz节流
                        image_signal.emit_fps(predict_fps=0)
                        self._last_fps_zero_emit = now

                # 检测状态转换：上一轮在推理 → 本轮已停止 → 通知前端
                if self._was_inferring and not is_inferring:
                    image_signal.emit_fps(predict_fps=0)
                self._was_inferring = is_inferring

                # 5. 周期性配置检查 (200ms)
                if now - self._last_config_check >= self._config_check_interval:
                    self._check_config()
                    self._last_config_check = now

            except Exception as e:
                logger.error(f"主循环错误: {e}")

    def _check_config(self):
        """检查配置变更"""
        self._check_model_change()
        capture_service.check_config_change()
        aim_service.update_config()

        hotkeys = config_service.get('aim', 'hotkeys', 'X1MouseButton,X2MouseButton')
        if hotkeys != self._last_hotkeys:
            self._cache_hotkey_codes()
            self._last_hotkeys = hotkeys

        # 刷新 hot path 配置缓存
        self._cached_auto = config_service.get('aim', 'auto', False)
        self._cached_mode = config_service.get('aim', 'mode', 'hold')

    def _check_model_change(self):
        """检测模型切换并热重载"""
        current = config_service.get('ai', 'model_name', '')
        if current and current != self._last_model_name:
            self._last_model_name = current
            logger.info(f"检测到模型切换: {current}")
            if inference_service.reload():
                new_size = inference_service.get_input_size()
                aim_service.set_model_input_size(new_size)
                config_service.update_section('capture', {
                    'window_width': new_size,
                    'window_height': new_size
                }, notify=True)
                logger.info(f"捕获窗口已自动调整为: {new_size}x{new_size}")
                # 推送更新后的配置到前端
                self._broadcast_config()

    def _broadcast_config(self):
        """推送当前配置到前端 WebSocket"""
        try:
            from core.websocket_server import websocket_server
            config = {
                'ai': config_service.get_section('ai'),
                'capture': config_service.get_section('capture'),
                'aim': config_service.get_section('aim'),
                'mouse': config_service.get_section('mouse')
            }
            websocket_server.broadcast_sync({
                'type': 'config',
                'data': config
            })
        except Exception as e:
            logger.error(f"推送配置到前端失败: {e}")

    def _check_need_prediction(self) -> bool:
        """检查是否需要预测"""
        if self._cached_auto:
            return True

        mode = self._cached_mode

        if mode == "toggle":
            for key_code in self._cached_hotkey_codes:
                try:
                    current_state = win32api.GetKeyState(key_code) < 0
                    last_state = self._key_states.get(key_code, False)

                    if current_state and not last_state:
                        self._toggle_enabled = not self._toggle_enabled
                        logger.info(f"自瞄已{'开启' if self._toggle_enabled else '关闭'}")

                    self._key_states[key_code] = current_state
                except Exception as e:
                    logger.error(f"热键检查错误: {e}")

            return self._toggle_enabled
        else:
            for key_code in self._cached_hotkey_codes:
                try:
                    if win32api.GetKeyState(key_code) < 0:
                        return True
                except Exception as e:
                    logger.error(f"热键检查错误: {e}")

        return False

    def stop(self):
        """停止应用"""
        if not self._running:
            return

        logger.info("停止自瞄应用...")
        self._running = False
        self._stop_event.set()

        capture_service.stop()
        aim_service.stop()

        # 卸载推理模型
        inference_service.unload()

        logger.info("自瞄应用已停止")

    def is_running(self) -> bool:
        """检查是否运行中"""
        return self._running


app = Aimbot()
