from PyQt6.QtCore import QObject, pyqtSignal


class LogSignal(QObject):
    log = pyqtSignal(str)


class ImageSignal(QObject):
    image = pyqtSignal(object)
    capture_time = pyqtSignal(float)
    predict_time = pyqtSignal(float)
    clear_predict_fps = pyqtSignal()
    capture_fps = pyqtSignal(float)
    predict_fps = pyqtSignal(float)
    detection_result = pyqtSignal(object)


log_signal = LogSignal()
image_signal = ImageSignal()
