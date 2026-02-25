from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor
from PyQt6.QtWidgets import QProxyStyle


class CheckBoxStyle(QProxyStyle):
    """自定义CheckBox样式 - IntelliJ IDEA风格"""

    def drawControl(self, control, option, painter, widget=None):
        if control == self.ControlElement.CE_CheckBox:
            painter.save()

            size = 16
            rect = option.rect.adjusted(
                (option.rect.width() - size) // 2,
                (option.rect.height() - size) // 2,
                -(option.rect.width() - size) // 2,
                -(option.rect.height() - size) // 2
            )

            border_color = QColor(74, 144, 226) if option.state & self.StateFlag.State_On else QColor(160, 160, 160)
            painter.setPen(QPen(border_color, 1.5))

            if option.state & self.StateFlag.State_On:
                painter.setBrush(QColor(74, 144, 226))
            else:
                painter.setBrush(QColor(255, 255, 255))

            painter.drawRoundedRect(rect, 2, 2)

            if option.state & self.StateFlag.State_On:
                painter.setPen(QPen(QColor(255, 255, 255), 1.5, Qt.PenStyle.SolidLine,
                                    Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

                tick_start_x = rect.left() + 4
                tick_start_y = rect.top() + rect.height() // 2
                tick_mid_x = rect.left() + rect.width() // 2 - 1
                tick_mid_y = rect.bottom() - 4
                tick_end_x = rect.right() - 3
                tick_end_y = rect.top() + 3

                painter.drawLine(tick_start_x, tick_start_y, tick_mid_x, tick_mid_y)
                painter.drawLine(tick_mid_x, tick_mid_y, tick_end_x, tick_end_y)

            painter.restore()
        else:
            super().drawControl(control, option, painter, widget)

    def hitTestComplexControl(self, control, option, point, widget=None):
        if control == self.ComplexControl.CC_CheckBox:
            expanded_rect = option.rect.adjusted(-10, -10, 10, 10)
            if expanded_rect.contains(point):
                return self.SubControl.SC_CheckBox
        return super().hitTestComplexControl(control, option, point, widget)
