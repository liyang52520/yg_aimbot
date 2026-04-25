import asyncio
import logging
import sys
import subprocess
import os
import time

from core.aimbot import app
from core.logger import setup_logger
from core.services.config_service import config_service
from core.ui_bridge import ui_bridge
from core.websocket_server import websocket_server

setup_logger()
logger = logging.getLogger(__name__)


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


def run_electron_ui():
    """运行 Electron UI"""
    print("YG Aimbot Configurator is starting...")
    
    # 启动 WebSocket 服务器
    websocket_server.start()
    ui_bridge.set_websocket(websocket_server)
    
    # 等待 WebSocket 服务器启动
    time.sleep(0.5)
    
    # 启动自瞄核心
    aimbot_thread = start_aimbot_thread()
    
    # 启动 Electron UI
    electron_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'electron_ui')
    
    try:
        # 检查是否安装了 Electron
        if sys.platform == 'win32':
            # Windows
            electron_exe = os.path.join(electron_dir, 'node_modules', '.bin', 'electron.cmd')
            if not os.path.exists(electron_exe):
                # 尝试全局安装
                electron_exe = 'electron'
        else:
            electron_exe = os.path.join(electron_dir, 'node_modules', '.bin', 'electron')
            if not os.path.exists(electron_exe):
                electron_exe = 'electron'
        
        # 启动 Electron
        cmd = [electron_exe, '.']
        logger.info(f"Starting Electron UI from {electron_dir}")
        
        process = subprocess.Popen(
            cmd,
            cwd=electron_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待 Electron 进程结束
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Electron process exited with code {process.returncode}")
            if stderr:
                logger.error(f"Electron stderr: {stderr.decode('utf-8', errors='ignore')}")
        
    except FileNotFoundError:
        logger.error("Electron not found. Please install it first:")
        logger.error("  cd electron_ui && npm install")
        print("\nError: Electron not found!")
        print("Please install it first by running:")
        print("  cd electron_ui")
        print("  npm install")
    except Exception as e:
        logger.error(f"Failed to start Electron UI: {e}")
    finally:
        # 停止 WebSocket 服务器
        websocket_server.stop()
        
        # 停止自瞄
        app.stop()
        if aimbot_thread.is_alive():
            aimbot_thread.join(timeout=5.0)
        logger.info("程序已退出")


def main():
    """主函数"""
    try:
        run_electron_ui()
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        config_service.save()
        logger.info("程序正在退出，释放所有资源...")


if __name__ == "__main__":
    main()
