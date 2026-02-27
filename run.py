import asyncio
import logging
import sys
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

from core.aimbot import app
from core.logger import setup_logger
from core.services.config_service import config_service
from ui.main_window import MainWindow
from ui.signals import log_signal

setup_logger()
logger = logging.getLogger(__name__)


class GUILogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_signal.log.emit(msg + '\n')
        except Exception:
            pass


gui_handler = GUILogHandler()
gui_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gui_handler.setFormatter(formatter)
logger.addHandler(gui_handler)


def run_aimbot():
    """运行自瞄"""
    logger.info("YG Aimbot is started! (Version 1.0.0)")
    asyncio.run(app.run())


def start_aimbot_thread():
    """启动自瞄线程"""
    import threading
    thread = threading.Thread(target=run_aimbot, daemon=True)
    thread.start()
    return thread


def run_gui():
    """运行GUI"""
    print("YG Aimbot Configurator is started!")
    application = QApplication(sys.argv)

    class CustomMainWindow(MainWindow):
        def closeEvent(self, event: QCloseEvent):
            logger.info("正在关闭系统...")
            app.stop()
            event.accept()

    window = CustomMainWindow()
    log_signal.log.connect(window.append_log)
    window.show()

    aimbot_thread = start_aimbot_thread()

    try:
        sys.exit(application.exec())
    finally:
        if aimbot_thread.is_alive():
            aimbot_thread.join(timeout=5.0)
        logger.info("程序已退出")


def main():
    """主函数"""
    try:
        run_gui()
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        config_service.save()
        logger.info("程序正在退出，释放所有资源...")


if __name__ == "__main__":
    main()
