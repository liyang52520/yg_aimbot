import asyncio
import logging
import threading
import time
from collections import deque

import numpy as np
import win32api

from core.services.aim_service import aim_service
from core.services.capture_service import capture_service
from core.services.config_service import config_service
from core.services.inference_service import inference_service
from core.services.tracker_service import tracker_service
from ui.signals import image_signal

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
        """主循环"""
        while self._running and not self._stop_event.is_set():
            try:
                current_time = time.time()

                if current_time - self._last_config_check >= self._config_check_interval:
                    self._check_config()
                    self._last_config_check = current_time

                frame, frame_id = capture_service.get_frame()
                if frame is None:
                    await asyncio.sleep(0.0001)
                    continue

                ai_debug = config_service.get('capture', 'ai_debug', False)
                need_prediction = self._check_need_prediction()

                if ai_debug or need_prediction:
                    # 异步推理模式
                    await self._handle_async_inference(frame, frame_id, ai_debug, need_prediction)
                else:
                    # 不需要预测时，确保预测帧率为0
                    inference_service._prediction_times.clear()
                    image_signal.predict_fps.emit(0.0)

                await asyncio.sleep(0.00001)

            except Exception as e:
                logger.error(f"主循环错误: {e}")

    async def _handle_async_inference(self, frame: np.ndarray, frame_id: int, ai_debug: bool, need_prediction: bool):
        """处理异步推理"""
        try:
            # 提交帧进行异步推理
            inference_service.submit_frame(frame, frame_id)

            # 获取最新推理结果（非阻塞）
            result = inference_service.get_latest_result()

            if result and result.detections is not None:
                if ai_debug:
                    image_signal.detection_result.emit(result.detections)

                if need_prediction:
                    tracker_service.update(result.detections, result.frame_id)

            if ai_debug:
                image_signal.image.emit(frame)

        except Exception as e:
            logger.error(f"异步推理处理错误: {e}")

    async def _handle_sync_inference(self, frame: np.ndarray, ai_debug: bool, need_prediction: bool):
        """处理同步推理（原有逻辑）"""
        detections = inference_service.predict(frame)
        if detections is not None:
            if ai_debug:
                image_signal.detection_result.emit(detections)
            if need_prediction:
                tracker_service.update(detections)
        if ai_debug:
            image_signal.image.emit(frame)

    def _check_config(self):
        """检查配置变更"""
        capture_service.check_config_change()
        aim_service.update_config()

        hotkeys = config_service.get('aim', 'hotkeys', 'X1MouseButton,X2MouseButton')
        if hotkeys != self._last_hotkeys:
            self._cache_hotkey_codes()
            self._last_hotkeys = hotkeys

    def _check_need_prediction(self) -> bool:
        """检查是否需要预测"""
        if config_service.get('aim', 'auto', False):
            return True

        mode = config_service.get('aim', 'mode', 'hold')

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