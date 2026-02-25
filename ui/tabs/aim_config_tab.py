from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QSlider
)

from ..styles import Styles
from ..widgets import CheckBoxStyle, BodyOffsetVisualizer, MultiSelectDropDown


class AimConfigTab(QWidget):
    """瞄准配置标签页"""

    offset_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
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
        """创建左侧面板 - 身体偏移设置"""
        left_widget = QWidget()
        left_widget.setMinimumWidth(280)
        left_widget.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        offset_title = QLabel("身体偏移设置")
        offset_title.setStyleSheet(Styles.get_title_label_style())
        left_layout.addWidget(offset_title)

        # 偏移输入框
        offset_input_layout = QGridLayout()
        offset_input_layout.setContentsMargins(0, 0, 0, 0)
        offset_input_layout.setSpacing(10)
        offset_input_layout.setColumnStretch(1, 1)

        # Body X Offset
        label = QLabel("X偏移:")
        label.setStyleSheet("color: #666666;")
        label.setFixedWidth(60)
        offset_input_layout.addWidget(label, 0, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.body_x_offset = QDoubleSpinBox()
        self.body_x_offset.setMinimum(-1.0)
        self.body_x_offset.setMaximum(1.0)
        self.body_x_offset.setSingleStep(0.1)
        self.body_x_offset.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.body_x_offset.setStyleSheet(Styles.get_spinbox_style())
        offset_input_layout.addWidget(self.body_x_offset, 0, 1, 1, 1)

        # Body Y Offset
        label = QLabel("Y偏移:")
        label.setStyleSheet("color: #666666;")
        label.setFixedWidth(60)
        offset_input_layout.addWidget(label, 1, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.body_y_offset = QDoubleSpinBox()
        self.body_y_offset.setMinimum(-1.0)
        self.body_y_offset.setMaximum(1.0)
        self.body_y_offset.setSingleStep(0.1)
        self.body_y_offset.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.body_y_offset.setStyleSheet(Styles.get_spinbox_style())
        offset_input_layout.addWidget(self.body_y_offset, 1, 1, 1, 1)
        left_layout.addLayout(offset_input_layout)

        # 可视化组件
        visualizer_title = QLabel("可视化调整")
        visualizer_title.setStyleSheet(Styles.get_title_label_style())
        left_layout.addWidget(visualizer_title)

        self.body_offset_visualizer = BodyOffsetVisualizer()
        self.body_offset_visualizer.setStyleSheet(Styles.get_body_visualizer_style())
        self.body_offset_visualizer.setFixedSize(240, 320)
        left_layout.addWidget(self.body_offset_visualizer)

        # 连接信号
        self.body_x_offset.valueChanged.connect(self._on_offset_changed)
        self.body_y_offset.valueChanged.connect(self._on_offset_changed)
        self.body_offset_visualizer.offsetChanged.connect(self._on_visualizer_offset_changed)

        return left_widget

    def _create_right_panel(self):
        """创建右侧面板 - 瞄准配置"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(18)

        # 配置标题
        config_title = QLabel("瞄准配置")
        config_title.setStyleSheet(Styles.get_title_label_style())
        right_layout.addWidget(config_title)

        # 配置网格
        config_layout = QGridLayout()
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(15)
        config_layout.setColumnStretch(1, 1)
        config_layout.setColumnMinimumWidth(0, 100)
        config_layout.setColumnMinimumWidth(1, 200)

        row = 0

        # Auto
        label = QLabel("自动瞄准:")
        label.setStyleSheet("color: #666666;")
        config_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.auto = QCheckBox()
        self.auto.setStyle(CheckBoxStyle())
        config_layout.addWidget(self.auto, row, 1, 1, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row += 1

        # Aim Mode
        label = QLabel("自瞄模式:")
        label.setStyleSheet("color: #666666;")
        config_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.aim_mode = QComboBox()
        self.aim_mode.addItems(["hold", "toggle"])
        self.aim_mode.setStyleSheet(Styles.get_combobox_style())
        config_layout.addWidget(self.aim_mode, row, 1, 1, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row += 1

        # Target Cls
        label = QLabel("目标类别:")
        label.setStyleSheet("color: #666666;")
        config_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.target_cls = QComboBox()
        self.target_cls.addItems(["0", "1", "2", "3", "4", "5"])
        self.target_cls.setStyleSheet(Styles.get_combobox_style())
        config_layout.addWidget(self.target_cls, row, 1, 1, 1,
                                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row += 1

        # Hotkeys
        label = QLabel("热键:")
        label.setStyleSheet("color: #666666;")
        config_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.hotkeys = MultiSelectDropDown()
        hotkey_options = ["X1MouseButton", "X2MouseButton", "RightMouseButton", "LeftMouseButton"]
        self.hotkeys.addItems(hotkey_options)
        self.hotkeys.setMinimumWidth(300)
        config_layout.addWidget(self.hotkeys, row, 1, 1, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        row += 1

        # Max Target Distance
        label = QLabel("最大目标距离:")
        label.setStyleSheet("color: #666666;")
        config_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        max_distance_layout = QHBoxLayout()
        self.max_target_distance = QSpinBox()
        self.max_target_distance.setMinimum(10)
        self.max_target_distance.setMaximum(160)
        self.max_target_distance.setFixedWidth(80)
        self.max_target_distance.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.max_target_distance.setStyleSheet(Styles.get_spinbox_style())

        self.max_target_distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_target_distance_slider.setMinimum(10)
        self.max_target_distance_slider.setMaximum(160)
        self.max_target_distance_slider.setSingleStep(5)
        self.max_target_distance_slider.valueChanged.connect(lambda value: self.max_target_distance.setValue(value))
        self.max_target_distance.valueChanged.connect(lambda value: self.max_target_distance_slider.setValue(value))
        self.max_target_distance_slider.setStyleSheet(Styles.get_slider_style())

        max_distance_layout.addWidget(self.max_target_distance_slider)
        max_distance_layout.addWidget(self.max_target_distance)
        config_layout.addLayout(max_distance_layout, row, 1, 1, 1)

        right_layout.addLayout(config_layout)

        # 分割线
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #e0e0e0;")
        right_layout.addWidget(separator)

        # 鼠标配置
        mouse_title = QLabel("鼠标配置")
        mouse_title.setStyleSheet(Styles.get_title_label_style())
        right_layout.addWidget(mouse_title)

        mouse_layout = self._create_mouse_config_layout()
        right_layout.addLayout(mouse_layout)

        right_layout.addStretch()

        return right_widget

    def _create_mouse_config_layout(self):
        """创建鼠标配置布局"""
        mouse_layout = QGridLayout()
        mouse_layout.setContentsMargins(0, 0, 0, 0)
        mouse_layout.setSpacing(12)
        mouse_layout.setColumnStretch(1, 1)
        mouse_layout.setColumnMinimumWidth(0, 100)
        mouse_layout.setColumnMinimumWidth(1, 200)

        row = 0

        # Mouse Move
        label = QLabel("鼠标移动方式:")
        label.setStyleSheet("color: #666666;")
        mouse_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.mouse_move = QComboBox()
        self.mouse_move.addItems(["makcu"])
        self.mouse_move.setStyleSheet(Styles.get_combobox_style())
        mouse_layout.addWidget(self.mouse_move, row, 1, 1, 1,
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row += 1

        # Mouse DPI
        label = QLabel("鼠标DPI:")
        label.setStyleSheet("color: #666666;")
        mouse_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dpi_layout = QHBoxLayout()
        self.mouse_dpi = QSpinBox()
        self.mouse_dpi.setMinimum(100)
        self.mouse_dpi.setMaximum(20000)
        self.mouse_dpi.setSingleStep(100)
        self.mouse_dpi.setFixedWidth(80)
        self.mouse_dpi.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.mouse_dpi.setStyleSheet(Styles.get_spinbox_style())

        self.mouse_dpi_slider = QSlider(Qt.Orientation.Horizontal)
        self.mouse_dpi_slider.setMinimum(100)
        self.mouse_dpi_slider.setMaximum(20000)
        self.mouse_dpi_slider.setSingleStep(100)
        self.mouse_dpi_slider.valueChanged.connect(lambda value: self.mouse_dpi.setValue(value))
        self.mouse_dpi.valueChanged.connect(lambda value: self.mouse_dpi_slider.setValue(value))
        self.mouse_dpi_slider.setStyleSheet(Styles.get_slider_style())

        dpi_layout.addWidget(self.mouse_dpi_slider)
        dpi_layout.addWidget(self.mouse_dpi)
        mouse_layout.addLayout(dpi_layout, row, 1, 1, 1)
        row += 1

        # Mouse Sensitivity
        label = QLabel("鼠标灵敏度:")
        label.setStyleSheet("color: #666666;")
        mouse_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        sensitivity_layout = QHBoxLayout()
        self.mouse_sensitivity = QDoubleSpinBox()
        self.mouse_sensitivity.setMinimum(0.1)
        self.mouse_sensitivity.setMaximum(10.0)
        self.mouse_sensitivity.setSingleStep(0.1)
        self.mouse_sensitivity.setFixedWidth(80)
        self.mouse_sensitivity.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.mouse_sensitivity.setStyleSheet(Styles.get_spinbox_style())

        self.mouse_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mouse_sensitivity_slider.setMinimum(1)
        self.mouse_sensitivity_slider.setMaximum(100)
        self.mouse_sensitivity_slider.setSingleStep(1)
        self.mouse_sensitivity_slider.valueChanged.connect(lambda value: self.mouse_sensitivity.setValue(value / 10))
        self.mouse_sensitivity.valueChanged.connect(
            lambda value: self.mouse_sensitivity_slider.setValue(int(value * 10)))
        self.mouse_sensitivity_slider.setStyleSheet(Styles.get_slider_style())

        sensitivity_layout.addWidget(self.mouse_sensitivity_slider)
        sensitivity_layout.addWidget(self.mouse_sensitivity)
        mouse_layout.addLayout(sensitivity_layout, row, 1, 1, 1)
        row += 1

        # Mouse FOV Width
        label = QLabel("FOV宽度:")
        label.setStyleSheet("color: #666666;")
        mouse_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        fov_width_layout = QHBoxLayout()
        self.mouse_fov_width = QSpinBox()
        self.mouse_fov_width.setMinimum(10)
        self.mouse_fov_width.setMaximum(180)
        self.mouse_fov_width.setFixedWidth(80)
        self.mouse_fov_width.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.mouse_fov_width.setStyleSheet(Styles.get_spinbox_style())

        self.mouse_fov_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.mouse_fov_width_slider.setMinimum(10)
        self.mouse_fov_width_slider.setMaximum(180)
        self.mouse_fov_width_slider.setSingleStep(5)
        self.mouse_fov_width_slider.valueChanged.connect(lambda value: self.mouse_fov_width.setValue(value))
        self.mouse_fov_width.valueChanged.connect(lambda value: self.mouse_fov_width_slider.setValue(value))
        self.mouse_fov_width_slider.setStyleSheet(Styles.get_slider_style())

        fov_width_layout.addWidget(self.mouse_fov_width_slider)
        fov_width_layout.addWidget(self.mouse_fov_width)
        mouse_layout.addLayout(fov_width_layout, row, 1, 1, 1)
        row += 1

        # Mouse FOV Height
        label = QLabel("FOV高度:")
        label.setStyleSheet("color: #666666;")
        mouse_layout.addWidget(label, row, 0, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        fov_height_layout = QHBoxLayout()
        self.mouse_fov_height = QSpinBox()
        self.mouse_fov_height.setMinimum(10)
        self.mouse_fov_height.setMaximum(180)
        self.mouse_fov_height.setFixedWidth(80)
        self.mouse_fov_height.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.mouse_fov_height.setStyleSheet(Styles.get_spinbox_style())

        self.mouse_fov_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.mouse_fov_height_slider.setMinimum(10)
        self.mouse_fov_height_slider.setMaximum(180)
        self.mouse_fov_height_slider.setSingleStep(5)
        self.mouse_fov_height_slider.valueChanged.connect(lambda value: self.mouse_fov_height.setValue(value))
        self.mouse_fov_height.valueChanged.connect(lambda value: self.mouse_fov_height_slider.setValue(value))
        self.mouse_fov_height_slider.setStyleSheet(Styles.get_slider_style())

        fov_height_layout.addWidget(self.mouse_fov_height_slider)
        fov_height_layout.addWidget(self.mouse_fov_height)
        mouse_layout.addLayout(fov_height_layout, row, 1, 1, 1)

        return mouse_layout

    def _on_offset_changed(self):
        """当偏移输入框值变化时，更新可视化组件"""
        x_offset = self.body_x_offset.value()
        y_offset = self.body_y_offset.value()
        self.body_offset_visualizer.setOffset(x_offset, y_offset)
        self.offset_changed.emit(x_offset, y_offset)

    def _on_visualizer_offset_changed(self, x, y):
        """当可视化组件偏移值变化时，更新输入框"""
        self.body_x_offset.setValue(x)
        self.body_y_offset.setValue(y)
        self.offset_changed.emit(x, y)
