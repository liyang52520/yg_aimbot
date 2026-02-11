import asyncio
import concurrent.futures
import logging
import os
import sys
import threading
import time

import supervision as sv
import torch
import win32api
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication
from ultralytics import YOLO

from core.buttons import Buttons
from core.capture import capture
from core.config import cfg
from core.frame_parser import frameParser
# 导入日志配置
from core.logger import setup_logger
from ui.main_window import MainWindow
# 从signals.py导入信号实例
from ui.signals import log_signal, image_signal

# 创建全局停止事件
stop_event = threading.Event()

# 创建全局配置刷新事件
config_refresh_event = threading.Event()


# 创建日志处理器类
class GUILogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_signal.log.emit(msg + '\n')
        except Exception:
            pass


setup_logger()
logger = logging.getLogger(__name__)

# 添加GUI日志处理器
gui_handler = GUILogHandler()
gui_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gui_handler.setFormatter(formatter)
logger.addHandler(gui_handler)

tracker = sv.ByteTrack() if cfg.ai_tracker else None


def perform_detection(model, image, tracker=None):
    """执行目标检测"""
    kwargs = {
        'source': image,
        'imgsz': cfg.ai_model_image_size,
        'conf': cfg.ai_conf,
        'iou': 0.45,
        'device': cfg.ai_device,
        'half': not "cpu" in cfg.ai_device,
        'max_det': 8,
        'agnostic_nms': False,
        'augment': False,
        'vid_stride': False,
        'visualize': False,
        'verbose': False,
        'show_boxes': False,
        'show_labels': False,
        'show_conf': False,
        'save': False,
        'show': False,
        'stream': True,
        'batch': False,
        'retina_masks': False,
        'classes': None,
        'simplify': True,
        'cfg': "config/tracker.yaml"
    }

    results = model.predict(**kwargs)

    if tracker:
        for res in results:
            det = sv.Detections.from_ultralytics(res)
            return tracker.update_with_detections(det)
        return None
    else:
        return next(results)


class Aimbot:
    """
    自瞄主类
    """

    def __init__(self):
        self.model = None
        self.executor = None
        self.running = False
        self.last_config = self._get_current_config()
        # 切换模式状态变量
        self.toggle_aim_enabled = False
        # 热键状态跟踪，用于检测按键按下事件
        self.key_states = {}
        # 用于计算瞬时帧率的变量
        self.capture_times = []  # 保存最近的采集时间
        self.prediction_times = []  # 保存最近的预测时间
        self.max_time_history = 10  # 保存最近10个时间点

    def _get_current_config(self):
        """获取当前配置快照"""
        return {
            'ai_model_name': cfg.ai_model_name,
            'ai_model_image_size': cfg.ai_model_image_size,
            'ai_conf': cfg.ai_conf,
            'ai_device': cfg.ai_device,
            'ai_tracker': cfg.ai_tracker,
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
            'mouse_move': cfg.mouse_move,
            'mouse_dpi': cfg.mouse_dpi,
            'mouse_sensitivity': cfg.mouse_sensitivity,
            'mouse_fov_width': cfg.mouse_fov_width,
            'mouse_fov_height': cfg.mouse_fov_height
        }

    def initialize(self):
        """初始化"""
        try:
            # 加载模型
            self.model = YOLO(f"models/{cfg.ai_model_name}", task="detect")

            # 预热模型
            import numpy as np
            dummy_image = np.zeros((cfg.ai_model_image_size, cfg.ai_model_image_size, 3), dtype=np.uint8)
            perform_detection(self.model, dummy_image, tracker)
            logger.info("模型加载成功并预热完成")

            # 创建线程池
            cpu_count = os.cpu_count()
            max_workers = min(cpu_count, 4)
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

            self.running = True
            self.last_config = self._get_current_config()
            return True
        except Exception as e:
            logger.error("初始化失败:\n", exc_info=e)
            return False

    async def run(self):
        """运行主循环"""
        if not self.initialize():
            return

        # 性能统计
        frame_count = 0
        prediction_count = 0
        start_time = time.time()
        last_print_time = time.time()
        last_config_check_time = time.time()

        while self.running and not stop_event.is_set():
            try:
                # 检查配置是否变更
                current_time = time.time()
                if current_time - last_config_check_time >= 1.0:  # 每秒检查一次配置
                    self._check_config_changes()
                    last_config_check_time = current_time

                # 获取图像并记录时间
                current_capture_time = time.time()
                image = capture.get_new_frame()
                if image is None:
                    await asyncio.sleep(0.0001)
                    continue

                # 记录采集时间
                self.capture_times.append(current_capture_time)
                if len(self.capture_times) > self.max_time_history:
                    self.capture_times.pop(0)

                # 性能统计
                frame_count += 1

                # 发送图像到GUI（当capture_ai_debug开启时）
                if cfg.capture_ai_debug:
                    image_signal.image.emit(image)

                # 定期计算并发送瞬时帧率
                if current_time - last_print_time >= 0.1:
                    # 计算瞬时采集帧率
                    if len(self.capture_times) >= 2:
                        capture_time_diff = self.capture_times[-1] - self.capture_times[0]
                        if capture_time_diff > 0:
                            instant_capture_fps = (len(self.capture_times) - 1) / capture_time_diff
                            image_signal.capture_fps.emit(instant_capture_fps)
                    else:
                        image_signal.capture_fps.emit(0.0)

                    # 计算瞬时预测帧率
                    if len(self.prediction_times) >= 2:
                        predict_time_diff = self.prediction_times[-1] - self.prediction_times[0]
                        if predict_time_diff > 0:
                            instant_predict_fps = (len(self.prediction_times) - 1) / predict_time_diff
                            image_signal.predict_fps.emit(instant_predict_fps)
                    else:
                        image_signal.predict_fps.emit(0.0)

                # 清除性能统计，不再打印
                if current_time - last_print_time >= 10:
                    frame_count = 0
                    prediction_count = 0
                    start_time = current_time
                    last_print_time = current_time

                # 处理图像
                if cfg.capture_circle:
                    image = capture.convert_to_circle(image)

                # 检查是否需要预测
                need_prediction = self._check_need_prediction()

                # 执行预测
                if need_prediction:
                    # 记录预测开始时间
                    predict_start_time = time.time()
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, perform_detection, self.model, image, tracker
                    )
                    predict_end_time = time.time()
                    prediction_count += 1

                    # 记录预测时间
                    self.prediction_times.append(predict_end_time)
                    if len(self.prediction_times) > self.max_time_history:
                        self.prediction_times.pop(0)

                    # 解析结果
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, frameParser.parse, result
                    )

            except Exception as e:
                logger.error("主循环错误:\n", exc_info=e)

            await asyncio.sleep(0.0001)

        # 释放资源
        self.stop()

    def _check_config_changes(self):
        """检查配置是否变更"""
        current_config = self._get_current_config()

        # 检查是否有配置变更
        changes = {}
        for key, value in current_config.items():
            if key not in self.last_config or self.last_config[key] != value:
                changes[key] = {'old': self.last_config.get(key), 'new': value}

        if changes:
            logger.info(f"配置变更: {changes}")
            # 检查是否需要重启服务
            if self._needs_restart(changes):
                logger.info("配置变更需要重启服务")
                # 重启服务
                self._restart_service()
            else:
                logger.info("配置变更只需要刷新参数")
                # 只刷新参数
                self.last_config = current_config

    def _needs_restart(self, changes):
        """检查是否需要重启服务"""
        # 需要重启服务的配置项
        restart_keys = [
            'ai_model_name',
            'ai_model_image_size',
            'ai_device',
            'ai_tracker'
        ]

        for key in changes:
            if key in restart_keys:
                return True

        return False

    def _restart_service(self):
        """重启服务"""
        logger.info("正在重启服务...")

        # 停止当前服务
        self.stop()

        # 等待一段时间
        time.sleep(1.0)

        # 重新初始化
        self.initialize()
        logger.info("服务重启完成")

    def stop(self):
        """停止自瞄系统并释放资源"""
        if not self.running:
            return

        logger.info("正在停止自瞄系统...")
        self.running = False

        # 关闭线程池
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("线程池已关闭")

        # 释放模型
        if self.model:
            del self.model
            logger.info("模型已释放")

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA缓存已清理")

        logger.info("自瞄系统已停止")

    def _print_performance(self, frame_count, prediction_count, start_time, current_time):
        """打印性能统计"""
        elapsed_time = current_time - start_time
        fps = frame_count / elapsed_time
        prediction_fps = prediction_count / elapsed_time
        logger.info(
            f"性能统计: 总帧率={fps:.2f}, 预测帧率={prediction_fps:.2f}, 预测次数={prediction_count}"
        )

    def _check_need_prediction(self):
        """检查是否需要预测"""
        # 检查自动瞄准
        if cfg.aim_auto:
            return True

        # 检查自瞄模式
        if cfg.aim_mode == "toggle":
            # 切换模式：按一下开启，再按一下关闭
            for key_name in cfg.aim_hotkeys:
                key_code = Buttons.KEY_CODES.get(key_name.strip())
                if key_code:
                    # 获取当前按键状态
                    current_state = win32api.GetKeyState(key_code) < 0
                    # 获取上次按键状态
                    last_state = self.key_states.get(key_code, False)

                    # 检测按键按下事件（从释放到按下）
                    if current_state and not last_state:
                        # 切换自瞄状态
                        self.toggle_aim_enabled = not self.toggle_aim_enabled
                        logger.info(f"自瞄已{'开启' if self.toggle_aim_enabled else '关闭'} (切换模式)")

                    # 更新按键状态
                    self.key_states[key_code] = current_state

            # 返回切换模式的自瞄状态
            return self.toggle_aim_enabled
        else:
            # 按住模式：保持原有的逻辑
            for key_name in cfg.aim_hotkeys:
                key_code = Buttons.KEY_CODES.get(key_name.strip())
                if key_code and win32api.GetKeyState(key_code) < 0:
                    return True

        # 默认不预测
        # 当自瞄关闭时，发送信号清零预测帧率并清空预测时间记录
        image_signal.clear_predict_fps.emit()
        # 清空预测时间记录，这样下次开启自瞄时会立即计算帧率
        self.prediction_times.clear()
        return False


# 全局Aimbot实例
aimbot_instance = None
# 全局自瞄线程实例
aimbot_thread = None


def run_aimbot():
    """运行自瞄系统"""
    global aimbot_instance
    logger.info("YG Aimbot is started! (Version 1.0.0)")
    aimbot_instance = Aimbot()
    asyncio.run(aimbot_instance.run())


def start_aimbot_service():
    """启动自瞄服务线程"""
    global aimbot_thread
    aimbot_thread = threading.Thread(target=run_aimbot, daemon=True)
    aimbot_thread.start()


def run_gui():
    """运行GUI页面"""
    print("YG Aimbot Configurator is started!")
    app = QApplication(sys.argv)

    # 创建自定义MainWindow子类，重写closeEvent方法
    class CustomMainWindow(MainWindow):
        def closeEvent(self, event: QCloseEvent):
            """处理关闭事件"""
            logger.info("正在关闭系统...")

            # 首先设置停止事件
            stop_event.set()

            # 等待自瞄系统停止
            if aimbot_instance:
                aimbot_instance.stop()

            # 等待自瞄线程结束
            if aimbot_thread and aimbot_thread.is_alive():
                logger.info("等待自瞄线程结束...")
                aimbot_thread.join(timeout=5.0)  # 最多等待5秒
                if aimbot_thread.is_alive():
                    logger.warning("自瞄线程未能在超时时间内结束")

            logger.info("所有服务已停止，正在关闭GUI页面...")
            # 关闭应用
            event.accept()

    window = CustomMainWindow()

    # 连接日志信号到GUI的append_log方法
    log_signal.log.connect(window.append_log)

    # 连接图像信号到GUI的update_video方法
    image_signal.image.connect(window.update_video)

    window.show()

    # GUI页面显示后启动自瞄服务
    start_aimbot_service()

    sys.exit(app.exec())


def main():
    """主函数"""
    try:
        # 首先启动GUI页面，GUI页面启动后会自动启动自瞄服务
        run_gui()
    finally:
        # 确保所有资源都被释放
        logger.info("程序正在退出，释放所有资源...")
        if aimbot_instance:
            aimbot_instance.stop()
        logger.info("程序已退出")


if __name__ == "__main__":
    main()
