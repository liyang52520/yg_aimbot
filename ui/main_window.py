from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox
)

from .config_manager import ConfigManager
from .styles import Styles
from .tabs import AIConfigTab, AimConfigTab
from .widgets import LogCapture


class MainWindow(QMainWindow):
    """主窗口 - 负责协调各个组件"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YG Aimbot Configurator")
        self.setGeometry(100, 100, 700, 500)

        # 初始化配置管理器
        self.config_manager = ConfigManager()

        # 设置全局字体
        font = QFont("Segoe UI", 10)
        QApplication.setFont(font)

        # 设置主题
        self._setup_theme()

        # 创建UI
        self._setup_ui()

        # 加载配置
        self._load_config()

        # 初始化日志捕获
        self.log_capture = LogCapture()
        self.log_capture.log_signal.connect(self.append_log)

        # 连接图像信号
        try:
            from ui.signals import image_signal
            image_signal.capture_fps.connect(self.update_capture_fps)
            image_signal.predict_fps.connect(self.update_predict_fps)
            image_signal.clear_predict_fps.connect(self.clear_predict_fps)
            image_signal.detection_result.connect(self.update_detections)
        except ImportError:
            pass

    def _setup_theme(self):
        """设置主题样式 - IDEA Light风格"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(74, 144, 226))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)

    def _setup_ui(self):
        """设置UI布局"""
        # 中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 创建标签页
        self._create_tabs()
        main_layout.addWidget(self.tab_widget)

        # 创建底部按钮
        button_layout = self._create_button_layout()
        main_layout.addLayout(button_layout)

    def _create_tabs(self):
        """创建标签页"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setStyleSheet(Styles.get_tab_widget_style())

        # AI配置标签页
        self.ai_config_tab = AIConfigTab()
        self.tab_widget.addTab(self.ai_config_tab, "AI配置")

        # 瞄准配置标签页
        self.aim_config_tab = AimConfigTab()
        self.tab_widget.addTab(self.aim_config_tab, "瞄准配置")

        # 连接配置变化信号
        self._connect_config_signals()

    def _create_button_layout(self):
        """创建底部按钮布局"""
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        button_layout.setSpacing(10)

        self.save_button = QPushButton("保存配置")
        self.save_button.setStyleSheet(Styles.get_button_style(primary=True))
        self.save_button.clicked.connect(self._save_config)

        button_layout.addStretch()
        button_layout.addWidget(self.save_button)

        return button_layout

    def _get_ui_components(self):
        """获取所有UI组件的字典"""
        return {
            # AI配置
            'ai_model_name': self.ai_config_tab.ai_model_name,
            'ai_conf': self.ai_config_tab.ai_conf,
            'ai_device': self.ai_config_tab.ai_device,
            'ai_tracker': self.ai_config_tab.ai_tracker,
            # Capture配置
            'capture_window_width': self.ai_config_tab.capture_window_width,
            'capture_window_height': self.ai_config_tab.capture_window_height,
            'capture_fps': self.ai_config_tab.capture_fps,
            'capture_circle': self.ai_config_tab.capture_circle,
            'capture_ai_debug': self.ai_config_tab.capture_ai_debug,
            # Aim配置
            'auto': self.aim_config_tab.auto,
            'aim_mode': self.aim_config_tab.aim_mode,
            'target_cls': self.aim_config_tab.target_cls,
            'body_x_offset': self.aim_config_tab.body_x_offset,
            'body_y_offset': self.aim_config_tab.body_y_offset,
            'hotkeys': self.aim_config_tab.hotkeys,
            'max_target_distance': self.aim_config_tab.max_target_distance,
            # Mouse配置
            'mouse_move': self.aim_config_tab.mouse_move,
            'mouse_dpi': self.aim_config_tab.mouse_dpi,
            'mouse_sensitivity': self.aim_config_tab.mouse_sensitivity,
            'mouse_fov_width': self.aim_config_tab.mouse_fov_width,
            'mouse_fov_height': self.aim_config_tab.mouse_fov_height,
        }

    def _load_config(self):
        """加载配置文件"""
        ui_components = self._get_ui_components()
        if not self.config_manager.load_config(ui_components):
            QMessageBox.warning(self, "警告", "配置文件不存在，将使用默认值")

    def auto_apply_config(self):
        """自动应用配置，不保存到文件"""
        try:
            ui_components = self._get_ui_components()
            self.config_manager.apply_config_to_memory(ui_components)
        except Exception as e:
            print(f"自动应用配置失败: {e}")

    def _on_video_debug_changed(self, state):
        """当视频监控复选框状态变化时的处理"""
        # 应用配置
        self.auto_apply_config()
        # 如果视频监控被关闭，清除最后一帧
        if state == 0:
            self.ai_config_tab.clear_video()

    def _connect_config_signals(self):
        """连接所有配置组件的信号到自动应用配置"""
        # AI配置信号
        self.ai_config_tab.ai_model_name.currentTextChanged.connect(self.auto_apply_config)
        self.ai_config_tab.ai_conf.valueChanged.connect(self.auto_apply_config)
        self.ai_config_tab.ai_device.valueChanged.connect(self.auto_apply_config)
        self.ai_config_tab.ai_tracker.stateChanged.connect(self.auto_apply_config)

        # 捕获配置信号
        self.ai_config_tab.capture_window_width.valueChanged.connect(self.auto_apply_config)
        self.ai_config_tab.capture_window_height.valueChanged.connect(self.auto_apply_config)
        self.ai_config_tab.capture_fps.valueChanged.connect(self.auto_apply_config)
        self.ai_config_tab.capture_circle.stateChanged.connect(self.auto_apply_config)
        self.ai_config_tab.capture_ai_debug.stateChanged.connect(self._on_video_debug_changed)

        # 瞄准配置信号
        self.aim_config_tab.auto.stateChanged.connect(self.auto_apply_config)
        self.aim_config_tab.aim_mode.currentTextChanged.connect(self.auto_apply_config)
        self.aim_config_tab.target_cls.currentTextChanged.connect(self.auto_apply_config)
        self.aim_config_tab.body_x_offset.valueChanged.connect(self.auto_apply_config)
        self.aim_config_tab.body_y_offset.valueChanged.connect(self.auto_apply_config)
        self.aim_config_tab.hotkeys.selectionChanged.connect(self.auto_apply_config)
        self.aim_config_tab.max_target_distance.valueChanged.connect(self.auto_apply_config)

        # 鼠标配置信号
        self.aim_config_tab.mouse_move.currentTextChanged.connect(self.auto_apply_config)
        self.aim_config_tab.mouse_dpi.valueChanged.connect(self.auto_apply_config)
        self.aim_config_tab.mouse_sensitivity.valueChanged.connect(self.auto_apply_config)
        self.aim_config_tab.mouse_fov_width.valueChanged.connect(self.auto_apply_config)
        self.aim_config_tab.mouse_fov_height.valueChanged.connect(self.auto_apply_config)

    def _save_config(self):
        """保存配置文件"""
        original_text = self.save_button.text()
        self.save_button.setText("保存中...")
        self.save_button.setEnabled(False)

        QApplication.processEvents()

        try:
            ui_components = self._get_ui_components()
            self.config_manager.save_config(ui_components)
            self.config_manager.apply_config_to_memory(ui_components)
            print("配置已保存并应用")
        finally:
            self.save_button.setText(original_text)
            self.save_button.setEnabled(True)
            QApplication.processEvents()

    def append_log(self, text):
        """添加日志到日志显示区域"""
        self.ai_config_tab.append_log(text)

    def update_video(self, image):
        """更新视频监控区域"""
        self.ai_config_tab.update_video(image)

    def update_capture_fps(self, fps):
        """更新采集帧率"""
        self.ai_config_tab.update_capture_fps(fps)

    def update_predict_fps(self, fps):
        """更新预测帧率"""
        self.ai_config_tab.update_predict_fps(fps)

    def clear_predict_fps(self):
        self.ai_config_tab.clear_predict_fps()
    
    def update_detections(self, detections):
        self.ai_config_tab.update_detections(detections)
    
    def update_capture_window_limits(self, max_size):
        self.ai_config_tab.update_capture_window_limits(max_size)
