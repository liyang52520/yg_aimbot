import asyncio
import concurrent.futures
import logging
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import supervision as sv
import torch
import win32api
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

from core.buttons import Buttons
from core.capture import capture
from core.config import cfg
from core.frame_parser import frameParser
# 导入日志配置
from core.logger import setup_logger
from core.yolo import create_yolo_model
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

# 全局检测参数，避免重复创建
detection_kwargs = {
    'iou': 0.45,
    'agnostic_nms': False,
    'augment': False,
    'vid_stride': False,
    'visualize': False,
    'show_boxes': False,
    'show_labels': False,
    'show_conf': False,
    'save': False,
    'show': False,
    'batch': False,
    'retina_masks': False,
    'classes': None,
    'simplify': True,
    'cfg': "config/tracker.yaml"
}


def perform_detection(model, image, tracker=None):
    """执行目标检测"""
    try:
        # 使用模型的predict方法执行检测
        detections = model.predict(image)

        if tracker:
            return tracker.update_with_detections(detections)
        else:
            return detections
    except Exception as e:
        # 减少日志开销
        if cfg.capture_ai_debug:
            logger.error(f"检测错误: {e}")
        return None


def parse_onnx_output(outputs, image_shape, input_size, conf_threshold):
    """解析ONNX模型输出"""
    # 假设输出是[batch, num_detections, 7]，其中7是[cx, cy, w, h, conf, class_id, ...]
    if len(outputs) == 1:
        output = outputs[0]
    else:
        # 对于YOLOv5 ONNX输出，通常是三个输出张量，需要合并处理
        output = outputs[0]

    # 过滤低置信度检测
    mask = output[0, :, 4] > conf_threshold
    filtered = output[0, mask]

    if len(filtered) == 0:
        # 返回空检测结果
        return sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty(0),
            class_id=np.empty(0, dtype=int)
        )

    # 转换为xyxy格式
    cx, cy, w, h, conf, class_id = filtered.T[:6]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 调整坐标到原始图像大小
    orig_h, orig_w = image_shape[:2]
    input_w, input_h = input_size  # (width, height)

    x1 = (x1 / input_w) * orig_w
    y1 = (y1 / input_h) * orig_h
    x2 = (x2 / input_w) * orig_w
    y2 = (y2 / input_h) * orig_h

    # 应用非极大值抑制
    xyxy = np.column_stack((x1, y1, x2, y2))
    indices = nms(xyxy, conf, class_id, iou_threshold=0.45)

    if len(indices) == 0:
        return sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty(0),
            class_id=np.empty(0, dtype=int)
        )

    # 构建最终检测结果
    final_xyxy = xyxy[indices]
    final_conf = conf[indices]
    final_class_id = class_id[indices].astype(int)

    return sv.Detections(
        xyxy=final_xyxy,
        confidence=final_conf,
        class_id=final_class_id
    )


def nms(boxes, scores, class_ids, iou_threshold):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []

    # 按类别分组
    unique_classes = np.unique(class_ids)
    keep_indices = []

    for cls in unique_classes:
        # 获取当前类别的边界框和分数
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        # 按分数排序
        sorted_indices = np.argsort(cls_scores)[::-1]
        cls_boxes = cls_boxes[sorted_indices]
        cls_indices = cls_indices[sorted_indices]

        # 应用NMS
        while len(cls_boxes) > 0:
            # 保留分数最高的边界框
            keep_indices.append(cls_indices[0])

            # 计算与其他边界框的IoU
            ious = calculate_iou(cls_boxes[0], cls_boxes[1:])

            # 过滤掉IoU大于阈值的边界框
            mask = ious <= iou_threshold
            cls_boxes = cls_boxes[1:][mask]
            cls_indices = cls_indices[1:][mask]

    return keep_indices


def calculate_iou(box, boxes):
    """计算IoU"""
    # 计算交集
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # 计算交集面积
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 计算并集面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    # 计算IoU
    iou = intersection / union

    return iou


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
        self.capture_times = deque(maxlen=30)  # 使用deque，提高性能
        self.prediction_times = deque(maxlen=30)  # 增加时间点保存数量
        self.config_check_interval = 0.2  # 进一步降低配置检查频率
        self.last_config_check_time = 0
        self.last_fps_update_time = 0
        self.fps_update_interval = 0.03  # 进一步提高帧率更新频率

        # 缓存的热键代码
        self.cached_hotkey_codes = []
        self._cache_hotkey_codes()

    def _cache_hotkey_codes(self):
        """缓存热键代码，避免重复查找"""
        self.cached_hotkey_codes = []
        for key_name in cfg.aim_hotkeys:
            key_code = Buttons.KEY_CODES.get(key_name.strip())
            if key_code:
                self.cached_hotkey_codes.append(key_code)

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
            'aim_mode': cfg.aim_mode,
            'mouse_move': cfg.mouse_move,
            'mouse_dpi': cfg.mouse_dpi,
            'mouse_sensitivity': cfg.mouse_sensitivity,
            'mouse_fov_width': cfg.mouse_fov_width,
            'mouse_fov_height': cfg.mouse_fov_height
        }

    def initialize(self):
        """初始化"""
        try:
            # 加载YOLO模型
            model_path = f"models/{cfg.ai_model_name}"
            self.model = create_yolo_model(model_path, cfg.ai_device, cfg.ai_conf, cfg.ai_model_type)

            if not self.model:
                logger.error("模型加载失败")
                return False

            # 动态创建线程池，根据系统资源调整
            cpu_count = os.cpu_count() or 4
            # 优化线程池配置，根据任务类型调整
            # 检测和追踪任务是CPU密集型，使用接近CPU核心数的线程数
            max_workers = min(cpu_count, 8)
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="AimbotWorker"
            )

            self.running = True
            self.last_config = self._get_current_config()
            self.last_config_check_time = time.time()
            self.last_fps_update_time = time.time()
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
        last_sleep_time = time.time()

        while self.running and not stop_event.is_set():
            try:
                current_time = time.time()

                # 检查配置是否变更
                if current_time - self.last_config_check_time >= self.config_check_interval:
                    self._check_config_changes()
                    self.last_config_check_time = current_time

                # 获取图像
                image = capture.get_new_frame()
                if image is None:
                    # 无图像时短暂休眠，避免CPU空转
                    if current_time - last_sleep_time >= 0.001:
                        await asyncio.sleep(0.0001)
                        last_sleep_time = current_time
                    continue

                # 记录采集时间
                self.capture_times.append(current_time)

                # 性能统计
                frame_count += 1

                # 发送图像到GUI（当capture_ai_debug开启时）
                if cfg.capture_ai_debug:
                    image_signal.image.emit(image)

                # 定期计算并发送瞬时帧率
                if current_time - self.last_fps_update_time >= self.fps_update_interval:
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

                    self.last_fps_update_time = current_time

                # 检查是否需要预测
                need_prediction = self._check_need_prediction()

                # 执行预测
                if need_prediction:
                    # 执行预测
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, self.model.predict, image
                    )
                    prediction_count += 1

                    # 记录预测时间
                    self.prediction_times.append(time.time())

                    # 应用跟踪器
                    if tracker and result is not None:
                        result = tracker.update_with_detections(result)

                    # 解析结果
                    if result is not None:
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, frameParser.parse, result
                        )

            except Exception as e:
                # 减少日志开销
                if cfg.capture_ai_debug:
                    logger.error("主循环错误:\n", exc_info=e)

            # 控制休眠频率，避免过多的上下文切换
            if current_time - last_sleep_time >= 0.001:
                await asyncio.sleep(0.00001)
                last_sleep_time = current_time

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
            if cfg.capture_ai_debug:
                logger.info(f"配置变更: {changes}")
            # 检查是否需要重启服务
            if self._needs_restart(changes):
                if cfg.capture_ai_debug:
                    logger.info("配置变更需要重启服务")
                # 重启服务
                self._restart_service()
            else:
                if cfg.capture_ai_debug:
                    logger.info("配置变更只需要刷新参数")
                # 只刷新参数
                self.last_config = current_config
                # 如果热键变更，重新缓存
                if 'aim_hotkeys' in changes:
                    self._cache_hotkey_codes()

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
        if cfg.capture_ai_debug:
            logger.info("正在重启服务...")

        # 停止当前服务
        self.stop()

        # 等待一段时间
        time.sleep(0.2)  # 进一步减少等待时间

        # 重新初始化
        self.initialize()
        if cfg.capture_ai_debug:
            logger.info("服务重启完成")

    def stop(self):
        """停止自瞄系统并释放资源"""
        if not self.running:
            return

        if cfg.capture_ai_debug:
            logger.info("正在停止自瞄系统...")
        self.running = False

        # 关闭线程池
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)  # 非阻塞关闭
            if cfg.capture_ai_debug:
                logger.info("线程池已关闭")

        # 释放模型
        if hasattr(self, 'model') and self.model:
            del self.model
            if cfg.capture_ai_debug:
                logger.info("模型已释放")

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if cfg.capture_ai_debug:
                logger.info("CUDA缓存已清理")

        if cfg.capture_ai_debug:
            logger.info("自瞄系统已停止")

    def _check_need_prediction(self):
        """检查是否需要预测"""
        # 检查自动瞄准
        if cfg.aim_auto:
            return True

        # 检查自瞄模式
        if cfg.aim_mode == "toggle":
            # 切换模式：按一下开启，再按一下关闭
            for key_code in self.cached_hotkey_codes:
                try:
                    # 获取当前按键状态
                    current_state = win32api.GetKeyState(key_code) < 0
                    # 获取上次按键状态
                    last_state = self.key_states.get(key_code, False)

                    # 检测按键按下事件（从释放到按下）
                    if current_state and not last_state:
                        # 切换自瞄状态
                        self.toggle_aim_enabled = not self.toggle_aim_enabled
                        if cfg.capture_ai_debug:
                            logger.info(f"自瞄已{'开启' if self.toggle_aim_enabled else '关闭'} (切换模式)")

                    # 更新按键状态
                    self.key_states[key_code] = current_state
                except Exception as e:
                    if cfg.capture_ai_debug:
                        logger.error(f"热键检查错误: {e}")

            # 返回切换模式的自瞄状态
            return self.toggle_aim_enabled
        else:
            # 按住模式：保持原有的逻辑
            for key_code in self.cached_hotkey_codes:
                try:
                    if win32api.GetKeyState(key_code) < 0:
                        return True
                except Exception as e:
                    if cfg.capture_ai_debug:
                        logger.error(f"热键检查错误: {e}")

        # 默认不预测
        # 当自瞄关闭时，发送信号清零预测帧率并清空预测时间记录
        try:
            image_signal.clear_predict_fps.emit()
        except Exception as e:
            if cfg.capture_ai_debug:
                logger.error(f"发送信号错误: {e}")
        # 清空预测时间记录
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
