from .theme import Theme


class Styles:
    """集中管理所有UI样式"""

    @staticmethod
    def get_button_style(primary=False):
        """获取按钮样式"""
        colors = Theme.colors
        spacing = Theme.spacing
        typography = Theme.typography

        if primary:
            return f"""
                QPushButton {{
                    background-color: {colors.primary};
                    color: white;
                    border: none;
                    border-radius: {spacing.border_radius_medium};
                    padding: {spacing.padding_medium};
                    font-family: {typography.family};
                    font-size: {typography.size_medium};
                }}
                QPushButton:hover {{
                    background-color: {colors.primary_hover};
                }}
                QPushButton:pressed {{
                    background-color: {colors.primary_pressed};
                }}
                QPushButton:focus {{
                    outline: none;
                    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.3);
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: {colors.background_tertiary};
                    color: {colors.text_primary};
                    border: 1px solid {colors.border};
                    border-radius: {spacing.border_radius_medium};
                    padding: {spacing.padding_medium};
                    font-family: {typography.family};
                    font-size: {typography.size_medium};
                }}
                QPushButton:hover {{
                    background-color: {colors.background_secondary};
                    border-color: {colors.border_hover};
                }}
                QPushButton:pressed {{
                    background-color: {colors.border};
                }}
                QPushButton:focus {{
                    outline: none;
                    border-color: #626681;
                }}
            """

    @staticmethod
    def get_spinbox_style():
        """获取数字输入框样式"""
        colors = Theme.colors
        spacing = Theme.spacing
        typography = Theme.typography

        return f"""
            QSpinBox, QDoubleSpinBox {{
                border: 1px solid {colors.border};
                border-radius: {spacing.border_radius_medium};
                padding: {spacing.padding_small};
                background: {colors.background};
                font-family: {typography.family};
                font-size: {typography.size_medium};
                color: {colors.text_primary};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {colors.border_focus};
                outline: none;
                box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
            }}
            QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {colors.primary_light};
            }}
        """

    @staticmethod
    def get_combobox_style():
        """获取下拉框样式"""
        colors = Theme.colors
        spacing = Theme.spacing
        typography = Theme.typography

        return f"""
            QComboBox {{
                border: 1px solid {colors.border};
                border-radius: {spacing.border_radius_medium};
                padding: {spacing.padding_small};
                background: {colors.background};
                font-family: {typography.family};
                font-size: {typography.size_medium};
                color: {colors.text_primary};
                min-width: 140px;
            }}
            QComboBox:focus {{
                border: 1px solid {colors.border_focus};
                outline: none;
                background: {colors.background};
            }}
            QComboBox:hover {{
                border-color: {colors.primary_light};
            }}
            QComboBox::drop-down {{
                border-left: 1px solid {colors.border};
                border-top-right-radius: {spacing.border_radius_medium};
                border-bottom-right-radius: {spacing.border_radius_medium};
                background: {colors.background_secondary};
            }}
            QComboBox::down-arrow {{
                image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12"><path fill="%23666666" d="M6 9L1 4h10z"/></svg>');
                width: 12px;
                height: 12px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #c0c0c0;
                border-radius: {spacing.border_radius_medium};
                padding: 2px;
                background: {colors.background};
                selection-background-color: {colors.primary_light};
                selection-color: {colors.text_primary};
                font-family: {typography.family};
                font-size: {typography.size_medium};
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: {spacing.padding_small};
                margin: 1px 0;
                border-radius: {spacing.border_radius_small};
                border: none;
                outline: none;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {colors.primary_light};
                border: none;
                outline: none;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {colors.primary_light};
                border: none;
                outline: none;
            }}
        """

    @staticmethod
    def get_slider_style():
        """获取滑块样式"""
        colors = Theme.colors
        spacing = Theme.spacing

        return f"""
            QSlider::groove:horizontal {{
                border: 1px solid {colors.border};
                height: 6px;
                background: {colors.background_tertiary};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {colors.primary};
                border: 1px solid {colors.primary_hover};
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }}
            QSlider::handle:horizontal:hover {{
                background: {colors.primary_hover};
            }}
            QSlider::handle:horizontal:pressed {{
                background: {colors.primary_pressed};
            }}
        """

    @staticmethod
    def get_group_box_style():
        """获取分组框样式"""
        colors = Theme.colors
        spacing = Theme.spacing

        return f"""
            QWidget {{
                border: 1px solid {colors.border};
                border-radius: {spacing.border_radius_xlarge};
                background-color: {colors.background};
            }}
        """

    @staticmethod
    def get_tab_widget_style():
        """获取标签页样式"""
        colors = Theme.colors
        spacing = Theme.spacing

        return f"""
            QTabWidget::pane {{
                border: 1px solid {colors.border};
                border-top: none;
                background: {colors.background};
            }}
            QTabBar::tab {{
                background: {colors.background_secondary};
                border: 1px solid {colors.border};
                border-bottom: none;
                padding: {spacing.padding_medium};
                margin-right: 2px;
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background: {colors.background};
                border-top: 2px solid {colors.primary};
            }}
        """

    @staticmethod
    def get_log_text_style():
        """获取日志文本框样式"""
        colors = Theme.colors
        spacing = Theme.spacing
        typography = Theme.typography

        return f"""
            QTextEdit {{
                border: 1px solid {colors.border};
                border-radius: {spacing.border_radius_medium};
                padding: {spacing.padding_large};
                background-color: {colors.background_quaternary};
                font-family: {typography.family_mono};
                font-size: {typography.size_small};
                color: {colors.text_primary};
                min-height: 200px;
            }}
        """

    @staticmethod
    def get_video_display_style():
        """获取视频显示区域样式"""
        colors = Theme.colors
        spacing = Theme.spacing

        return f"""
            QWidget {{
                border: 1px solid {colors.border};
                border-radius: {spacing.border_radius_medium};
                background-color: {colors.background_secondary};
            }}
        """

    @staticmethod
    def get_title_label_style():
        """获取标题标签样式"""
        colors = Theme.colors
        typography = Theme.typography

        return f"""
            QLabel {{
                font-weight: 600;
                color: {colors.text_primary};
                font-size: {typography.size_large};
                margin-bottom: 8px;
                border: none;
                background: transparent;
            }}
        """

    @staticmethod
    def get_refresh_button_style():
        """获取刷新按钮样式"""
        colors = Theme.colors
        spacing = Theme.spacing
        typography = Theme.typography

        return f"""
            QPushButton {{
                background-color: {colors.background_tertiary};
                color: {colors.text_secondary};
                border: 1px solid {colors.border};
                border-radius: {spacing.border_radius_medium};
                padding: {spacing.padding_small};
                margin-left: 8px;
                font-size: {typography.size_large};
            }}
            QPushButton:hover {{
                background-color: {colors.background_secondary};
                border-color: {colors.border_hover};
            }}
            QPushButton:focus {{
                outline: none;
                border-color: #626681;
            }}
        """

    @staticmethod
    def get_body_visualizer_style():
        """获取身体可视化组件样式"""
        colors = Theme.colors
        spacing = Theme.spacing

        return f"""
            QWidget {{
                border: 1px solid {colors.border};
                border-radius: {spacing.border_radius_large};
                background-color: {colors.background_quaternary};
            }}
        """
