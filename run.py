import asyncio
import concurrent.futures
import logging
import os
import queue
import sys
import threading
import time
from collections import deque

import torch
import win32api
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

from core.buttons import Buttons
from core.capturer import capture
from core.config import cfg
from core.frame_parser import frameParser
from core.logger import setup_logger
from core.yolo_model import UltralyticsYOLOModel
from ui.main_window import MainWindow
from ui.signals import log_signal, image_signal


class GUILogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_signal.log.emit(msg + '\n')
        except Exception:
            pass


setup_logger()
logger = logging.getLogger(__name__)

gui_handler = GUILogHandler()
gui_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gui_handler.setFormatter(formatter)
logger.addHandler(gui_handler)


class Aimbot:
    def __init__(self):
        self.model = None
        self.executor = None
        self.running = False
        self.last_config = self._get_current_config()
        self.toggle_aim_enabled = False
        self.key_states = {}
        self.capture_times = deque(maxlen=30)
        self.prediction_times = deque(maxlen=30)
        self.config_check_interval = 0.2
        self.last_config_check_time = 0
        self.last_fps_update_time = 0
        self.fps_update_interval = 0.03
        self.cached_hotkey_codes = []
        self._cache_hotkey_codes()
        self._prediction_task_queue = queue.Queue(maxsize=2)
        self._prediction_result_queue = queue.Queue(maxsize=2)
        self._prediction_worker_thread = None
        self._prediction_worker_running = False

    def _cache_hotkey_codes(self):
        self.cached_hotkey_codes = []
        for key_name in cfg.aim_hotkeys:
            key_code = Buttons.KEY_CODES.get(key_name.strip())
            if key_code:
                self.cached_hotkey_codes.append(key_code)

    def _prediction_worker(self):
        while self._prediction_worker_running:
            try:
                image, timestamp = self._prediction_task_queue.get(timeout=0.1)
                result = self.model.predict(image)
                try:
                    self._prediction_result_queue.put_nowait((result, timestamp))
                except queue.Full:
                    pass
                self._prediction_task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                if cfg.capture_ai_debug:
                    logger.error(f"预测工作线程错误: {e}")

    def _get_current_config(self):
        return {
            'ai_model_name': cfg.ai_model_name,
            'ai_conf': cfg.ai_conf,
            'ai_device': cfg.ai_device,
            'capture_window_width': cfg.capture_window_width,
            'capture_window_height': cfg.capture_window_height,
            'capture_fps': cfg.capture_fps,
            'capture_circle': cfg.capture_circle,
            'capture_ai_debug': cfg.capture_ai_debug,
            'aim_auto': cfg.aim_auto,
            'aim_target_cls': cfg.aim_target_cls,
            'aim_body_x_offset': cfg.aim_body_x_offset,
            'aim_body_y_offset': cfg.aim_body_y_offset,
            'aim_hotkeys': cfg.aim_hotkeys.copy(),
            'aim_mode': cfg.aim_mode,
            'mouse_move': cfg.mouse_move,
            'mouse_dpi': cfg.mouse_dpi,
            'mouse_sensitivity': cfg.mouse_sensitivity,
            'mouse_fov_width': cfg.mouse_fov_width,
            'mouse_fov_height': cfg.mouse_fov_height
        }

    def initialize(self):
        try:
            model_path = f"data/{cfg.ai_model_name}"
            self.model = UltralyticsYOLOModel(model_path, cfg.ai_device, cfg.ai_conf)

            if not self.model or not self.model.load_model():
                logger.error("模型加载失败")
                return False

            from core.mouse import mouse
            model_input_size = self.model.get_input_size()
            mouse.set_model_input_size(model_input_size)
            logger.info(f"模型输入大小已设置为: {model_input_size}x{model_input_size}")

            try:
                from ui.main_window import MainWindow
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, MainWindow):
                        widget.update_capture_window_limits(model_input_size)
                        break
            except Exception as e:
                logger.debug(f"更新捕获窗口大小限制失败: {e}")

            cpu_count = os.cpu_count() or 4
            max_workers = min(cpu_count, 8)
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="AimbotWorker"
            )

            self.running = True
            self.last_config = self._get_current_config()
            self.last_config_check_time = time.time()
            self.last_fps_update_time = time.time()

            self._prediction_worker_running = True
            self._prediction_worker_thread = threading.Thread(target=self._prediction_worker, daemon=True)
            self._prediction_worker_thread.start()

            return True
        except Exception as e:
            logger.error("初始化失败:\n", exc_info=e)
            return False

    async def run(self):
        if not self.initialize():
            return

        frame_count = 0
        prediction_count = 0
        start_time = time.time()
        last_sleep_time = time.time()

        while self.running and not stop_event.is_set():
            try:
                current_time = time.time()

                if current_time - self.last_config_check_time >= self.config_check_interval:
                    self._check_config_changes()
                    self.last_config_check_time = current_time

                image = capture.get_new_frame()
                if image is None:
                    if current_time - last_sleep_time >= 0.001:
                        await asyncio.sleep(0.0001)
                        last_sleep_time = current_time
                    continue

                self.capture_times.append(current_time)
                frame_count += 1

                if cfg.capture_ai_debug:
                    image_signal.image.emit(image)

                if current_time - self.last_fps_update_time >= self.fps_update_interval:
                    if len(self.capture_times) >= 2:
                        capture_time_diff = self.capture_times[-1] - self.capture_times[0]
                        if capture_time_diff > 0:
                            instant_capture_fps = (len(self.capture_times) - 1) / capture_time_diff
                            image_signal.capture_fps.emit(instant_capture_fps)
                    else:
                        image_signal.capture_fps.emit(0.0)

                    if len(self.prediction_times) >= 2:
                        predict_time_diff = self.prediction_times[-1] - self.prediction_times[0]
                        if predict_time_diff > 0:
                            instant_predict_fps = (len(self.prediction_times) - 1) / predict_time_diff
                            image_signal.predict_fps.emit(instant_predict_fps)
                    else:
                        image_signal.predict_fps.emit(0.0)

                    self.last_fps_update_time = current_time

                need_prediction = self._check_need_prediction()

                if need_prediction:
                    try:
                        self._prediction_task_queue.put_nowait((image, current_time))
                    except queue.Full:
                        pass

                try:
                    result, result_timestamp = self._prediction_result_queue.get_nowait()
                    prediction_count += 1
                    self.prediction_times.append(result_timestamp)

                    if cfg.capture_ai_debug and result is not None:
                        image_signal.detection_result.emit(result)

                    if result is not None and self._check_need_aim():
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, frameParser.parse, result
                        )
                except queue.Empty:
                    pass

            except Exception as e:
                if cfg.capture_ai_debug:
                    logger.error("主循环错误:\n", exc_info=e)

            if current_time - last_sleep_time >= 0.001:
                await asyncio.sleep(0.00001)
                last_sleep_time = current_time

        self.stop()

    def _check_config_changes(self):
        current_config = self._get_current_config()

        changes = {}
        for key, value in current_config.items():
            if key not in self.last_config or self.last_config[key] != value:
                changes[key] = {'old': self.last_config.get(key), 'new': value}

        if changes:
            if cfg.capture_ai_debug:
                logger.info(f"配置变更: {changes}")
            if self._needs_restart(changes):
                if cfg.capture_ai_debug:
                    logger.info("配置变更需要重启服务")
                self._restart_service()
                return
            else:
                if cfg.capture_ai_debug:
                    logger.info("配置变更只需要刷新参数")
                self.last_config = current_config
                if 'aim_hotkeys' in changes:
                    self._cache_hotkey_codes()

    def _needs_restart(self, changes):
        restart_keys = ['ai_model_name', 'ai_device']
        return any(key in changes for key in restart_keys)

    def _restart_service(self):
        if cfg.capture_ai_debug:
            logger.info("正在重启服务...")
        self.stop()
        time.sleep(0.2)
        self.initialize()
        if cfg.capture_ai_debug:
            logger.info("服务重启完成")

    def stop(self):
        if not self.running:
            return

        if cfg.capture_ai_debug:
            logger.info("正在停止自瞄系统...")
        self.running = False

        self._prediction_worker_running = False
        if self._prediction_worker_thread and self._prediction_worker_thread.is_alive():
            self._prediction_worker_thread.join(timeout=1.0)

        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            if cfg.capture_ai_debug:
                logger.info("线程池已关闭")

        if hasattr(self, 'model') and self.model:
            del self.model
            if cfg.capture_ai_debug:
                logger.info("模型已释放")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if cfg.capture_ai_debug:
                logger.info("CUDA缓存已清理")

        if cfg.capture_ai_debug:
            logger.info("自瞄系统已停止")

    def _check_need_prediction(self):
        if cfg.capture_ai_debug:
            return True

        if cfg.aim_auto:
            return True

        if cfg.aim_mode == "toggle":
            for key_code in self.cached_hotkey_codes:
                try:
                    current_state = win32api.GetKeyState(key_code) < 0
                    last_state = self.key_states.get(key_code, False)

                    if current_state and not last_state:
                        self.toggle_aim_enabled = not self.toggle_aim_enabled
                        if cfg.capture_ai_debug:
                            logger.info(f"自瞄已{'开启' if self.toggle_aim_enabled else '关闭'} (切换模式)")

                    self.key_states[key_code] = current_state
                except Exception as e:
                    if cfg.capture_ai_debug:
                        logger.error(f"热键检查错误: {e}")

            return self.toggle_aim_enabled
        else:
            for key_code in self.cached_hotkey_codes:
                try:
                    if win32api.GetKeyState(key_code) < 0:
                        return True
                except Exception as e:
                    if cfg.capture_ai_debug:
                        logger.error(f"热键检查错误: {e}")

        try:
            image_signal.clear_predict_fps.emit()
        except Exception as e:
            if cfg.capture_ai_debug:
                logger.error(f"发送信号错误: {e}")
        self.prediction_times.clear()
        return False

    def _check_need_aim(self):
        if cfg.aim_auto:
            return True

        if cfg.aim_mode == "toggle":
            return self.toggle_aim_enabled
        else:
            for key_code in self.cached_hotkey_codes:
                try:
                    if win32api.GetKeyState(key_code) < 0:
                        return True
                except Exception as e:
                    if cfg.capture_ai_debug:
                        logger.error(f"热键检查错误: {e}")

        return False


stop_event = threading.Event()
aimbot_instance = None
aimbot_thread = None


def run_aimbot():
    global aimbot_instance
    logger.info("YG Aimbot is started! (Version 1.0.0)")
    aimbot_instance = Aimbot()
    asyncio.run(aimbot_instance.run())


def start_aimbot_service():
    global aimbot_thread
    aimbot_thread = threading.Thread(target=run_aimbot, daemon=True)
    aimbot_thread.start()


def run_gui():
    print("YG Aimbot Configurator is started!")
    app = QApplication(sys.argv)

    class CustomMainWindow(MainWindow):
        def closeEvent(self, event: QCloseEvent):
            logger.info("正在关闭系统...")
            stop_event.set()

            if aimbot_instance:
                aimbot_instance.stop()

            if aimbot_thread and aimbot_thread.is_alive():
                logger.info("等待自瞄线程结束...")
                aimbot_thread.join(timeout=5.0)
                if aimbot_thread.is_alive():
                    logger.warning("自瞄线程未能在超时时间内结束")

            logger.info("所有服务已停止，正在关闭GUI页面...")
            event.accept()

    window = CustomMainWindow()
    log_signal.log.connect(window.append_log)
    image_signal.image.connect(window.update_video)
    window.show()
    start_aimbot_service()
    sys.exit(app.exec())


def main():
    try:
        run_gui()
    finally:
        logger.info("程序正在退出，释放所有资源...")
        if aimbot_instance:
            aimbot_instance.stop()
        logger.info("程序已退出")


if __name__ == "__main__":
    main()
