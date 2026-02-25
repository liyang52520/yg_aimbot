import io
import sys

from PyQt6.QtCore import QObject, pyqtSignal


class LogCapture(QObject):
    """日志捕获类，用于捕获标准输出和标准错误"""
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.string_io = io.StringIO()
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        """重写write方法，捕获输出"""
        self.stdout.write(text)
        self.string_io.write(text)
        self.log_signal.emit(text)

    def flush(self):
        """重写flush方法"""
        self.stdout.flush()
        self.string_io.flush()

    def close(self):
        """关闭捕获，恢复原始输出"""
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.string_io.close()
