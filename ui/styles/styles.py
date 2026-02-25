class Styles:
    """集中管理所有UI样式"""

    @staticmethod
    def get_button_style(primary=False):
        """获取按钮样式"""
        if primary:
            return """
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
                QPushButton:pressed {
                    background-color: #2c6aa0;
                }
                QPushButton:focus {
                    outline: none;
                    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.3);
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #f0f0f0;
                    color: #333333;
                    border: 1px solid #d0d0d0;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                    border-color: #999999;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
                QPushButton:focus {
                    outline: none;
                    border-color: #626681;
                }
            """

    @staticmethod
    def get_spinbox_style():
        """获取数字输入框样式"""
        return """
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 10px;
                background: #ffffff;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
                color: #333333;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #4a90e2;
                outline: none;
                box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #94c6f8;
            }
        """

    @staticmethod
    def get_combobox_style():
        """获取下拉框样式"""
        return """
            QComboBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 10px;
                background: #ffffff;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
                color: #333333;
                min-width: 140px;
            }
            QComboBox:focus {
                border: 1px solid #4a90e2;
                outline: none;
                background: #ffffff;
            }
            QComboBox:hover {
                border-color: #94c6f8;
            }
            QComboBox::drop-down {
                border-left: 1px solid #d0d0d0;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                background: #f5f5f5;
            }
            QComboBox::down-arrow {
                image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12"><path fill="%23666666" d="M6 9L1 4h10z"/></svg>');
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 2px;
                background: #ffffff;
                selection-background-color: #94c6f8;
                selection-color: #333333;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 10px;
                margin: 1px 0;
                border-radius: 2px;
                border: none;
                outline: none;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #94c6f8;
                border: none;
                outline: none;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #94c6f8;
                border: none;
                outline: none;
            }
        """

    @staticmethod
    def get_slider_style():
        """获取滑块样式"""
        return """
            QSlider::groove:horizontal {
                border: 1px solid #e0e0e0;
                height: 6px;
                background: #f0f0f0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #357abd;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #357abd;
            }
            QSlider::handle:horizontal:pressed {
                background: #2c6aa0;
            }
        """

    @staticmethod
    def get_group_box_style():
        """获取分组框样式"""
        return """
            QWidget {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """

    @staticmethod
    def get_tab_widget_style():
        """获取标签页样式"""
        return """
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-top: none;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-top: 2px solid #4a90e2;
            }
        """

    @staticmethod
    def get_log_text_style():
        """获取日志文本框样式"""
        return """
            QTextEdit {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 8px 12px;
                background-color: #f9f9f9;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                color: #333333;
                min-height: 200px;
            }
        """

    @staticmethod
    def get_video_display_style():
        """获取视频显示区域样式"""
        return """
            QWidget {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
        """

    @staticmethod
    def get_title_label_style():
        """获取标题标签样式"""
        return """
            QLabel {
                font-weight: 600;
                color: #333333;
                font-size: 14px;
                margin-bottom: 8px;
                border: none;
                background: transparent;
            }
        """

    @staticmethod
    def get_refresh_button_style():
        """获取刷新按钮样式"""
        return """
            QPushButton {
                background-color: #f0f0f0;
                color: #666666;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 8px;
                margin-left: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #999999;
            }
            QPushButton:focus {
                outline: none;
                border-color: #626681;
            }
        """

    @staticmethod
    def get_body_visualizer_style():
        """获取身体可视化组件样式"""
        return """
            QWidget {
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                background-color: #f9f9f9;
            }
        """
