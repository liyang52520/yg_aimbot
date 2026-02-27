import configparser
import logging
import os
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class Config:
    CONFIG_PATH = "config/config.ini"
    RELOAD_INTERVAL = 1.0

    def __init__(self):
        self.config = configparser.ConfigParser()
        self._last_modified_time = 0.0
        self._lock = threading.RLock()
        self._callbacks: list[Callable] = []
        self._auto_reload = True
        self._last_check_time = 0.0

        self.Read(verbose=False)

    def _get_file_modified_time(self) -> float:
        try:
            return os.path.getmtime(self.CONFIG_PATH)
        except (OSError, FileNotFoundError):
            return 0.0

    def check_reload(self, verbose: bool = False) -> bool:
        if not self._auto_reload:
            return False

        current_time = time.time()
        if current_time - self._last_check_time < self.RELOAD_INTERVAL:
            return False
        self._last_check_time = current_time

        try:
            current_mtime = self._get_file_modified_time()
            if current_mtime > self._last_modified_time:
                with self._lock:
                    self.Read(verbose=verbose)
                    self._last_modified_time = current_mtime
                    for callback in self._callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"配置重载回调错误: {e}")
                return True
        except Exception as e:
            logger.error(f"检查配置重载错误: {e}")
        return False

    def register_reload_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def unregister_reload_callback(self, callback: Callable):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def Read(self, verbose: bool = False):
        try:
            with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config.read_file(f)
            self._last_modified_time = self._get_file_modified_time()
        except FileNotFoundError:
            logger.error(f"[Config] 配置文件未找到: {self.CONFIG_PATH}")
            raise
        except Exception as e:
            logger.error(f"[Config] 读取配置文件异常: {str(e)}")
            raise

        self._load_capture_config()
        self._load_ai_config()
        self._load_aim_config()
        self._load_mouse_config()

        if verbose:
            logger.info("[Config] 配置已重新加载")

    def _load_capture_config(self):
        self.config_Capture = self.config["Capture"]
        self.capture_window_width = self._get_int("Capture", "capture_window_width")
        self.capture_window_height = self._get_int("Capture", "capture_window_height")
        self.capture_circle = self._get_boolean("Capture", "capture_circle")
        self.capture_fps = self._get_int("Capture", "capture_fps")
        self.capture_ai_debug = self._get_boolean("Capture", "capture_ai_debug")

    def _load_ai_config(self):
        self.config_AI = self.config["AI"]
        self.ai_model_name = self._get_str("AI", "ai_model_name")
        self.ai_model_type = "yolov5" if self.ai_model_name.lower().startswith("yolov5") else "ultralytics"
        self.ai_conf = self._get_float("AI", "ai_conf")
        self.ai_device = self._get_str("AI", "ai_device")

    def _load_aim_config(self):
        self.config_Aim = self.config["Aim"]
        self.aim_auto = self._get_boolean("Aim", "auto")
        self.aim_target_cls = self._get_float("Aim", "target_cls")
        self.aim_hotkeys = self._get_str("Aim", "hotkeys").split(",")
        self.aim_body_x_offset = self._get_float("Aim", "body_x_offset")
        self.aim_body_y_offset = self._get_float("Aim", "body_y_offset")
        self.aim_mode = self._get_str("Aim", "mode", fallback="hold")
        self.aim_max_target_distance = self._get_int("Aim", "max_target_distance", fallback=90)

    def _load_mouse_config(self):
        self.config_Mouse = self.config["Mouse"]
        self.mouse_move = self._get_str("Mouse", "mouse_move")
        self.mouse_dpi = self._get_int("Mouse", "mouse_dpi")
        self.mouse_sensitivity = self._get_float("Mouse", "mouse_sensitivity")
        self.mouse_fov_width = self._get_int("Mouse", "mouse_fov_width")
        self.mouse_fov_height = self._get_int("Mouse", "mouse_fov_height")

    def _get_str(self, section: str, key: str, fallback: Optional[str] = None) -> str:
        try:
            return self.config.get(section, key, fallback=fallback) if fallback else self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if fallback is not None:
                return fallback
            logger.error(f"[Config] 配置项缺失 [{section}].{key}: {e}")
            raise

    def _get_int(self, section: str, key: str, fallback: Optional[int] = None) -> int:
        try:
            return self.config.getint(section, key, fallback=fallback) if fallback is not None else self.config.getint(
                section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            if fallback is not None:
                return fallback
            logger.error(f"[Config] 配置项错误或缺失 [{section}].{key}: {e}")
            raise

    def _get_float(self, section: str, key: str, fallback: Optional[float] = None) -> float:
        try:
            return self.config.getfloat(section, key,
                                        fallback=fallback) if fallback is not None else self.config.getfloat(section,
                                                                                                             key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            if fallback is not None:
                return fallback
            logger.error(f"[Config] 配置项错误或缺失 [{section}].{key}: {e}")
            raise

    def _get_boolean(self, section: str, key: str, fallback: Optional[bool] = None) -> bool:
        try:
            return self.config.getboolean(section, key,
                                          fallback=fallback) if fallback is not None else self.config.getboolean(
                section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            if fallback is not None:
                return fallback
            logger.error(f"[Config] 配置项错误或缺失 [{section}].{key}: {e}")
            raise


cfg = Config()
