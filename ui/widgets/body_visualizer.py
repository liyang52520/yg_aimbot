import os

from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap
from PyQt6.QtWidgets import QWidget


class BodyOffsetVisualizer(QWidget):
    """身体偏移可视化组件"""
    offsetChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 300)
        self.setMaximumSize(300, 450)

        self.enemy_image = QPixmap()
        image_path = "ui/resources/enemy.svg"
        if os.path.exists(image_path):
            self.enemy_image.load(image_path)

        self.x_offset = 0.0
        self.y_offset = 0.0
        self.marker_pos = QPointF(0, 0)
        self.is_dragging = False

    def setOffset(self, x, y):
        self.x_offset = x
        self.y_offset = y
        self.update_marker_pos()
        self.update()

    def update_marker_pos(self):
        center_x = self.width() // 2
        center_y = self.height() // 2

        image_height = self.height() * 0.8
        image_width = image_height * 0.4

        pos_x = center_x + self.x_offset * (image_width // 2)
        pos_y = center_y + self.y_offset * (image_height // 2)

        self.marker_pos = QPointF(pos_x, pos_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(245, 245, 245))

        if not self.enemy_image.isNull():
            image_height = self.height() * 0.8
            image_width = image_height * (self.enemy_image.width() / self.enemy_image.height())
            image_rect = painter.window().adjusted(
                (self.width() - int(image_width)) // 2,
                (self.height() - int(image_height)) // 2,
                -(self.width() - int(image_width)) // 2,
                -(self.height() - int(image_height)) // 2
            )
            painter.drawPixmap(image_rect, self.enemy_image)
        else:
            self.draw_simple_body(painter)

        painter.setPen(QPen(QColor(128, 128, 128), 1, Qt.PenStyle.DashLine))
        center_x = self.width() // 2
        center_y = self.height() // 2
        painter.drawLine(center_x, 0, center_x, self.height())
        painter.drawLine(0, center_y, self.width(), center_y)

        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QColor(255, 0, 0))
        marker_size = 8
        painter.drawEllipse(self.marker_pos, marker_size, marker_size)

    def draw_simple_body(self, painter):
        center_x = self.width() // 2
        center_y = self.height() // 2
        body_height = self.height() * 0.8
        body_width = body_height * 0.4

        painter.setPen(QPen(QColor(100, 100, 100), 2))

        head_radius = body_width * 0.3
        painter.drawEllipse(
            center_x - head_radius,
            center_y - body_height // 2,
            head_radius * 2,
            head_radius * 2
        )

        painter.drawRect(
            center_x - body_width * 0.3,
            center_y - body_height // 2 + head_radius * 2,
            body_width * 0.6,
            body_height * 0.5
        )

        painter.drawLine(
            center_x - body_width * 0.3,
            center_y - body_height // 2 + head_radius * 2 + body_height * 0.1,
            center_x - body_width * 0.8,
            center_y - body_height // 4
        )
        painter.drawLine(
            center_x + body_width * 0.3,
            center_y - body_height // 2 + head_radius * 2 + body_height * 0.1,
            center_x + body_width * 0.8,
            center_y - body_height // 4
        )
        painter.drawLine(
            center_x - body_width * 0.2,
            center_y - body_height // 2 + head_radius * 2 + body_height * 0.5,
            center_x - body_width * 0.4,
            center_y + body_height // 2
        )
        painter.drawLine(
            center_x + body_width * 0.2,
            center_y - body_height // 2 + head_radius * 2 + body_height * 0.5,
            center_x + body_width * 0.4,
            center_y + body_height // 2
        )

    def mousePressEvent(self, event):
        self.is_dragging = True
        self.update_offset_from_pos(event.position())

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.update_offset_from_pos(event.position())

    def mouseReleaseEvent(self, event):
        self.is_dragging = False

    def update_offset_from_pos(self, pos):
        center_x = self.width() // 2
        center_y = self.height() // 2

        max_offset_x = self.width() * 0.4
        max_offset_y = self.height() * 0.4

        new_x_offset = (pos.x() - center_x) / max_offset_x
        new_y_offset = (pos.y() - center_y) / max_offset_y

        new_x_offset = max(-1.0, min(1.0, new_x_offset))
        new_y_offset = max(-1.0, min(1.0, new_y_offset))

        if abs(new_x_offset - self.x_offset) > 0.01 or abs(new_y_offset - self.y_offset) > 0.01:
            self.x_offset = new_x_offset
            self.y_offset = new_y_offset
            self.marker_pos = pos
            self.update()
            self.offsetChanged.emit(self.x_offset, self.y_offset)
