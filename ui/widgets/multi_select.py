from PyQt6.QtCore import pyqtSignal, QEvent
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QMenu


class MultiSelectDropDown(QWidget):
    """多选下拉框组件"""
    selectionChanged = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.container = QWidget()
        self.container.setStyleSheet("""
            QWidget {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QWidget:hover {
                border-color: #94c6f8;
            }
        """)
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(8, 4, 4, 4)
        container_layout.setSpacing(4)

        self.tags_container = QWidget()
        self.tags_container.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
        """)
        self.tags_layout = QHBoxLayout(self.tags_container)
        self.tags_layout.setContentsMargins(0, 0, 0, 0)
        self.tags_layout.setSpacing(6)
        self.tags_layout.addStretch()

        self.placeholder_label = QLabel("请选择热键")
        self.placeholder_label.setStyleSheet("""
            QLabel {
                color: #999999;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
                background: transparent;
                border: none;
            }
        """)
        self.tags_layout.insertWidget(0, self.placeholder_label)

        self.dropdown_button = QPushButton("▼")
        self.dropdown_button.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                padding: 4px 6px;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 11px;
                color: #666666;
            }
            QPushButton:hover {
                background: #f0f0f0;
            }
        """)
        self.dropdown_button.setFixedWidth(24)

        container_layout.addWidget(self.tags_container, 1)
        container_layout.addWidget(self.dropdown_button)

        self.layout.addWidget(self.container)

        self.menu = QMenu(self)

        self.dropdown_button.clicked.connect(self.toggle_menu)
        self.container.mousePressEvent = lambda event: self.toggle_menu()
        self.menu.aboutToHide.connect(self.on_menu_hide)

        self.installEventFilter(self)

        self.selected_items = []
        self.checkbox_actions = {}
        self.items = []
        self.tags = {}
        self.menu_is_open = False

    def addItem(self, text, data=None):
        self.items.append(text)
        action = QAction(text, self)
        action.setCheckable(True)
        action.triggered.connect(lambda checked, t=text: self.handle_checkbox_changed(t, checked))
        self.menu.addAction(action)
        self.checkbox_actions[text] = action

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    def handle_checkbox_changed(self, text, checked):
        if checked:
            if text not in self.selected_items:
                self.selected_items.append(text)
                self.add_tag(text)
        else:
            if text in self.selected_items:
                self.selected_items.remove(text)
                self.remove_tag(text)

        self.update_placeholder()
        self.selectionChanged.emit(self.selected_items)

    def add_tag(self, text):
        if self.placeholder_label.isVisible():
            self.placeholder_label.hide()

        tag_widget = QWidget()
        tag_widget.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
        """)
        tag_layout = QHBoxLayout(tag_widget)
        tag_layout.setContentsMargins(0, 0, 0, 0)
        tag_layout.setSpacing(4)

        tag_label = QLabel(text)
        tag_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 12px;
                background-color: #f0f0f0;
                border-radius: 12px;
                padding: 2px 6px;
            }
        """)

        remove_button = QPushButton("×")
        remove_button.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                padding: 0px 2px;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 14px;
                color: #666666;
            }
            QPushButton:hover {
                color: #333333;
            }
        """)
        remove_button.setFixedSize(16, 16)
        remove_button.clicked.connect(lambda _, t=text: self.handle_checkbox_changed(t, False))

        tag_layout.addWidget(tag_label)
        tag_layout.addWidget(remove_button)

        self.tags_layout.insertWidget(len(self.tags), tag_widget)
        self.tags[text] = tag_widget

    def remove_tag(self, text):
        if text in self.tags:
            tag_widget = self.tags[text]
            self.tags_layout.removeWidget(tag_widget)
            tag_widget.deleteLater()
            del self.tags[text]

        self.update_placeholder()

    def update_placeholder(self):
        if len(self.selected_items) == 0:
            self.placeholder_label.show()
        else:
            self.placeholder_label.hide()

    def setSelectedItems(self, items):
        for item in list(self.selected_items):
            self.handle_checkbox_changed(item, False)

        for item in items:
            if item in self.checkbox_actions:
                self.handle_checkbox_changed(item, True)

    def getSelectedItems(self):
        return self.selected_items

    def toggle_menu(self):
        if self.menu_is_open:
            self.menu.close()
            self.on_menu_hide()
        else:
            for text, action in self.checkbox_actions.items():
                action.setChecked(text in self.selected_items)

            pos = self.container.mapToGlobal(self.container.rect().bottomLeft())

            menu_width = self.container.width()
            self.menu.setMinimumWidth(menu_width)
            self.menu.setMaximumWidth(menu_width)

            self.menu.setStyleSheet("""
                QMenu {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: #ffffff;
                    padding: 4px 0;
                }
                QMenu::item {
                    padding: 6px 16px;
                    margin: 0;
                    border: none;
                }
                QMenu::item:selected {
                    background-color: #f0f0f0;
                }
            """)

            self.dropdown_button.setText("▲")
            self.menu_is_open = True

            self.menu.popup(pos)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if self.isAncestorOf(obj) or obj == self:
                if obj == self:
                    global_pos = self.mapToGlobal(event.pos())
                else:
                    global_pos = obj.mapToGlobal(event.pos())

                if self.container.geometry().contains(self.container.mapFromGlobal(global_pos)) or \
                        self.dropdown_button.geometry().contains(self.dropdown_button.mapFromGlobal(global_pos)):
                    if self.menu_is_open:
                        self.menu.close()
                        self.on_menu_hide()
                        return True
        return False

    def on_menu_hide(self):
        self.menu_is_open = False
        self.dropdown_button.setText("▼")
