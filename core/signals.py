from PyQt6.QtCore import QObject, pyqtSignal


# 创建日志信号类
class LogSignal(QObject):
    log = pyqtSignal(str)


# 创建图像信号类
class ImageSignal(QObject):
    image = pyqtSignal(object)
    capture_time = pyqtSignal(float)
    predict_time = pyqtSignal(float)
    clear_predict_fps = pyqtSignal()
    capture_fps = pyqtSignal(float)
    predict_fps = pyqtSignal(float)


# 创建全局日志信号实例
log_signal = LogSignal()
# 创建全局图像信号实例
image_signal = ImageSignal()
