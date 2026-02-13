import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QSlider, QTextEdit
)

from ..styles import Styles
from ..widgets import CheckBoxStyle


class AIConfigTab(QWidget):
    """AI配置标签页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_label = None
        self.log_text_edit = None
        self.fps_labels = None
        self.calculated_capture_fps = 0.0
        self.calculated_predict_fps = 0.0
        self.last_capture_time = 0.0
        self.last_predict_time = 0.0
        # 移动平均计算
        self.capture_fps_history = []
        self.predict_fps_history = []
        self.max_history = 10  # 保存10个最近的帧率值
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()

        layout.addWidget(left_widget)
        layout.addWidget(right_widget, 1)

    def _create_left_panel(self):
        """创建左侧面板 - 视频监控和日志"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)
        left_widget.setMinimumWidth(400)

        # 视频监控区域
        video_widget = self._create_video_section()
        left_layout.addWidget(video_widget)

        # 日志显示区域
        log_widget = self._create_log_section()
        left_layout.addWidget(log_widget)

        return left_widget

    def _create_video_section(self):
        """创建视频监控区域"""
        video_widget = QWidget()
        video_widget.setStyleSheet(Styles.get_group_box_style())
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(16, 16, 16, 16)
        video_layout.setSpacing(12)

        video_title = QLabel("视频监控")
        video_title.setStyleSheet(Styles.get_title_label_style())
        video_layout.addWidget(video_title)

        video_display = QWidget()
        video_display.setMinimumHeight(200)
        video_display.setStyleSheet(Styles.get_video_display_style())
        video_display_layout = QVBoxLayout(video_display)
        video_display_layout.setContentsMargins(0, 0, 0, 0)
        video_display_layout.setSpacing(0)

        # 视频显示区域
        self.video_label = QLabel("视频监控区域")
        self.video_label.setStyleSheet("color: #999999;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_display_layout.addWidget(self.video_label, 1)

        # 帧率显示区域
        fps_widget = QWidget()
        fps_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0.7); color: white;")
        fps_layout = QHBoxLayout(fps_widget)
        fps_layout.setContentsMargins(8, 4, 8, 4)
        fps_layout.setSpacing(20)

        # 采集帧率
        capture_fps_label = QLabel("采集帧率: 0.0 FPS")
        capture_fps_label.setStyleSheet("color: #4CAF50;")
        fps_layout.addWidget(capture_fps_label)

        # 预测帧率
        predict_fps_label = QLabel("预测帧率: 0.0 FPS")
        predict_fps_label.setStyleSheet("color: #2196F3;")
        fps_layout.addWidget(predict_fps_label)

        self.fps_labels = (capture_fps_label, predict_fps_label)
        video_display_layout.addWidget(fps_widget)

        video_layout.addWidget(video_display)

        return video_widget

    def _create_log_section(self):
        """创建日志显示区域"""
        log_widget = QWidget()
        log_widget.setStyleSheet(Styles.get_group_box_style())
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.setSpacing(12)

        log_title = QLabel("系统日志")
        log_title.setStyleSheet(Styles.get_title_label_style())
        log_layout.addWidget(log_title)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet(Styles.get_log_text_style())
        log_layout.addWidget(self.log_text_edit)

        return log_widget

    def _create_right_panel(self):
        """创建右侧面板 - AI和捕获配置"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(20)
        right_widget.setMinimumWidth(400)

        # AI配置
        ai_widget = self._create_ai_config_section()
        right_layout.addWidget(ai_widget)

        # 捕获配置
        capture_widget = self._create_capture_config_section()
        right_layout.addWidget(capture_widget)

        return right_widget

    def _create_ai_config_section(self):
        """创建AI配置区域"""
        ai_widget = QWidget()
        ai_widget.setObjectName("ai_config")
        ai_widget.setStyleSheet("""
            QWidget#ai_config {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        ai_layout = QGridLayout(ai_widget)
        ai_layout.setContentsMargins(20, 20, 20, 20)
        ai_layout.setSpacing(12)
        ai_layout.setColumnStretch(1, 1)

        ai_title = QLabel("AI配置")
        ai_title.setStyleSheet(Styles.get_title_label_style())
        ai_layout.addWidget(ai_title, 0, 0, 1, 3, Qt.AlignmentFlag.AlignLeft)

        row = 1

        # AI Model
        label = QLabel("AI模型:")
        label.setStyleSheet("color: #666666;")
        ai_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        model_layout = QHBoxLayout()
        self.ai_model_name = QComboBox()
        self.ai_model_name.setStyleSheet(Styles.get_combobox_style())
        self._scan_model_files()

        refresh_button = QPushButton("⟳")
        refresh_button.setStyleSheet(Styles.get_refresh_button_style())
        refresh_button.clicked.connect(self._scan_model_files)

        model_layout.addWidget(self.ai_model_name)
        model_layout.addWidget(refresh_button)
        ai_layout.addLayout(model_layout, row, 1, 1, 2)
        row += 1

        # AI Model Image Size
        label = QLabel("模型图像尺寸:")
        label.setStyleSheet("color: #666666;")
        ai_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.ai_model_image_size = QSpinBox()
        self.ai_model_image_size.setMinimum(128)
        self.ai_model_image_size.setMaximum(1024)
        self.ai_model_image_size.setSingleStep(32)
        self.ai_model_image_size.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.ai_model_image_size.setStyleSheet(Styles.get_spinbox_style())
        ai_layout.addWidget(self.ai_model_image_size, row, 1, 1, 2)
        row += 1

        # AI Conf
        label = QLabel("置信度阈值:")
        label.setStyleSheet("color: #666666;")
        ai_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        conf_layout = QHBoxLayout()
        self.ai_conf = QDoubleSpinBox()
        self.ai_conf.setMinimum(0.1)
        self.ai_conf.setMaximum(1.0)
        self.ai_conf.setSingleStep(0.1)
        self.ai_conf.setFixedWidth(80)
        self.ai_conf.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.ai_conf.setStyleSheet(Styles.get_spinbox_style())

        self.ai_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.ai_conf_slider.setMinimum(10)
        self.ai_conf_slider.setMaximum(100)
        self.ai_conf_slider.setSingleStep(10)
        self.ai_conf_slider.setStyleSheet(Styles.get_slider_style())
        self.ai_conf_slider.valueChanged.connect(lambda value: self.ai_conf.setValue(value / 100))
        self.ai_conf.valueChanged.connect(lambda value: self.ai_conf_slider.setValue(int(value * 100)))

        conf_layout.addWidget(self.ai_conf_slider)
        conf_layout.addWidget(self.ai_conf)
        ai_layout.addLayout(conf_layout, row, 1, 1, 2)
        row += 1

        # AI Device
        label = QLabel("设备ID:")
        label.setStyleSheet("color: #666666;")
        ai_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.ai_device = QSpinBox()
        self.ai_device.setMinimum(0)
        self.ai_device.setMaximum(10)
        self.ai_device.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.ai_device.setStyleSheet(Styles.get_spinbox_style())
        ai_layout.addWidget(self.ai_device, row, 1, 1, 2)
        row += 1

        # AI Tracker
        label = QLabel("轨迹预测:")
        label.setStyleSheet("color: #666666;")
        ai_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.ai_tracker = QCheckBox()
        self.ai_tracker.setStyle(CheckBoxStyle())
        ai_layout.addWidget(self.ai_tracker, row, 1, 1, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        return ai_widget

    def _create_capture_config_section(self):
        """创建捕获配置区域"""
        capture_widget = QWidget()
        capture_widget.setObjectName("capture_config")
        capture_widget.setStyleSheet("""
            QWidget#capture_config {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        capture_layout = QGridLayout(capture_widget)
        capture_layout.setContentsMargins(20, 20, 20, 20)
        capture_layout.setSpacing(12)
        capture_layout.setColumnStretch(1, 1)

        capture_title = QLabel("捕获配置")
        capture_title.setStyleSheet(Styles.get_title_label_style())
        capture_layout.addWidget(capture_title, 0, 0, 1, 3, Qt.AlignmentFlag.AlignLeft)

        row = 1

        # Capture Window
        label = QLabel("捕获窗口:")
        label.setStyleSheet("color: #666666;")
        capture_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        window_layout = QHBoxLayout()
        window_layout.setSpacing(0)
        self.capture_window_width = QSpinBox()
        self.capture_window_width.setMinimum(160)
        self.capture_window_width.setMaximum(1920)
        self.capture_window_width.setSingleStep(10)
        self.capture_window_width.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.capture_window_width.setStyleSheet(Styles.get_spinbox_style())
        window_layout.addWidget(self.capture_window_width)

        times_label = QLabel("x")
        times_label.setStyleSheet("color: #666666; margin: 0 4px;")
        window_layout.addWidget(times_label)

        self.capture_window_height = QSpinBox()
        self.capture_window_height.setMinimum(160)
        self.capture_window_height.setMaximum(1080)
        self.capture_window_height.setSingleStep(10)
        self.capture_window_height.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.capture_window_height.setStyleSheet(Styles.get_spinbox_style())
        window_layout.addWidget(self.capture_window_height)

        capture_layout.addLayout(window_layout, row, 1, 1, 2)
        row += 1

        # Capture FPS
        label = QLabel("捕获帧率:")
        label.setStyleSheet("color: #666666;")
        capture_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        fps_layout = QHBoxLayout()
        self.capture_fps = QSpinBox()
        self.capture_fps.setMinimum(1)
        self.capture_fps.setMaximum(240)
        self.capture_fps.setFixedWidth(80)
        self.capture_fps.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.capture_fps.setStyleSheet(Styles.get_spinbox_style())

        self.capture_fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.capture_fps_slider.setMinimum(1)
        self.capture_fps_slider.setMaximum(240)
        self.capture_fps_slider.setSingleStep(5)
        self.capture_fps_slider.setStyleSheet(Styles.get_slider_style())
        self.capture_fps_slider.valueChanged.connect(lambda value: self.capture_fps.setValue(value))
        self.capture_fps.valueChanged.connect(lambda value: self.capture_fps_slider.setValue(value))

        fps_layout.addWidget(self.capture_fps_slider)
        fps_layout.addWidget(self.capture_fps)
        capture_layout.addLayout(fps_layout, row, 1, 1, 2)
        row += 1

        # 圆形捕获和视频监控
        circle_label = QLabel("圆形捕获:")
        circle_label.setStyleSheet("color: #666666;")
        capture_layout.addWidget(circle_label, row, 0, 1, 1,
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        circle_layout = QHBoxLayout()
        circle_layout.setSpacing(40)

        self.capture_circle = QCheckBox()
        self.capture_circle.setStyle(CheckBoxStyle())
        circle_layout.addWidget(self.capture_circle, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        debug_label = QLabel("视频监控:")
        debug_label.setStyleSheet("color: #666666;")
        circle_layout.addWidget(debug_label, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.capture_ai_debug = QCheckBox()
        self.capture_ai_debug.setStyle(CheckBoxStyle())
        circle_layout.addWidget(self.capture_ai_debug, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        capture_layout.addLayout(circle_layout, row, 1, 1, 2)

        return capture_widget

    def _scan_model_files(self):
        """扫描models目录下的模型文件"""
        current_model = self.ai_model_name.currentText()
        self.ai_model_name.clear()
        models_dir = "models"
        extensions = [".pt", ".onnx", ".engine"]

        try:
            if os.path.exists(models_dir) and os.path.isdir(models_dir):
                for file in os.listdir(models_dir):
                    if any(file.endswith(ext) for ext in extensions):
                        self.ai_model_name.addItem(file)

            if self.ai_model_name.count() == 0:
                self.ai_model_name.addItem("YOLOv8s_apex_teammate_enemy.engine")
            else:
                index = self.ai_model_name.findText(current_model)
                if index >= 0:
                    self.ai_model_name.setCurrentIndex(index)
        except Exception:
            self.ai_model_name.addItem("YOLOv8s_apex_teammate_enemy.engine")

    def append_log(self, text):
        """添加日志到日志显示区域"""
        if self.log_text_edit:
            self.log_text_edit.append(text.strip())
            self.log_text_edit.ensureCursorVisible()

    def update_video(self, image):
        """更新视频监控区域"""
        from PyQt6.QtGui import QPixmap, QImage
        import cv2

        if image is None or self.video_label is None:
            return

        try:
            # 帧率计算现在通过run.py中的信号获取，这里不再计算

            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)

                self.video_label.setPixmap(scaled_pixmap)
                self.video_label.setScaledContents(False)
                self.video_label.setText("")
        except Exception:
            pass

    def clear_video(self):
        """清除视频监控区域的最后一帧"""
        if self.video_label:
            self.video_label.clear()
            self.video_label.setText("视频监控区域")
            self.video_label.setStyleSheet("color: #999999;")

    def update_capture_fps(self, fps):
        """更新采集帧率显示（使用移动平均）"""
        # 添加到历史记录
        self.capture_fps_history.append(fps)
        # 保持历史记录长度
        if len(self.capture_fps_history) > self.max_history:
            self.capture_fps_history.pop(0)
        # 计算移动平均
        avg_fps = sum(self.capture_fps_history) / len(self.capture_fps_history)
        # 更新显示
        if self.fps_labels:
            capture_fps_label, _ = self.fps_labels
            capture_fps_label.setText(f"采集帧率: {avg_fps:.1f} FPS")

    def update_predict_fps(self, fps):
        """更新预测帧率显示（使用移动平均）"""
        # 添加到历史记录
        self.predict_fps_history.append(fps)
        # 保持历史记录长度
        if len(self.predict_fps_history) > self.max_history:
            self.predict_fps_history.pop(0)
        # 计算移动平均
        avg_fps = sum(self.predict_fps_history) / len(self.predict_fps_history)
        # 更新显示
        if self.fps_labels:
            _, predict_fps_label = self.fps_labels
            predict_fps_label.setText(f"预测帧率: {avg_fps:.1f} FPS")

    def clear_predict_fps(self):
        """清零预测帧率显示"""
        # 清空历史记录
        self.predict_fps_history.clear()
        # 更新显示
        if self.fps_labels:
            _, predict_fps_label = self.fps_labels
            predict_fps_label.setText("预测帧率: 0.0 FPS")
