import configparser

import logging

logger = logging.getLogger(__name__)


class Config():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.Read(verbose=False)

    def Read(self, verbose=False):
        try:
            with open("config/config.ini", "r", encoding="utf-8", ) as f:
                self.config.read_file(f)
        except FileNotFoundError:
            logger.error("[Config] Config file not found!")
            quit()
        except Exception as e:
            logger.error(f"[Config] Unknown exception: {str(e)}")

        # Detection window
        self.config_Capture = self.config["Capture"]
        self.capture_window_width = int(self.config_Capture["capture_window_width"])
        self.capture_window_height = int(self.config_Capture["capture_window_height"])
        self.capture_circle = self.config_Capture.getboolean("capture_circle")
        self.capture_fps = int(self.config_Capture["capture_fps"])
        self.capture_ai_debug = self.config_Capture.getboolean("capture_ai_debug")

        # AI
        self.config_AI = self.config["AI"]
        self.ai_model_name = str(self.config_AI["ai_model_name"])
        self.ai_model_image_size = int(self.config_AI["ai_model_image_size"])
        self.ai_model_type = self.config_AI["ai_model_type"]
        self.ai_conf = float(self.config_AI["ai_conf"])
        self.ai_device = str(self.config_AI["ai_device"])
        self.ai_tracker = self.config_AI.getboolean("ai_tracker")

        # Aim
        self.config_Aim = self.config["Aim"]
        self.aim_auto = self.config_Aim.getboolean("auto")
        self.aim_target_cls = float(self.config_Aim["target_cls"])
        self.aim_hotkeys = str(self.config_Aim["hotkeys"]).split(",")
        self.aim_body_x_offset = float(self.config_Aim["body_x_offset"])
        self.aim_body_y_offset = float(self.config_Aim["body_y_offset"])
        self.aim_mode = self.config_Aim.get("mode", "hold")
        self.aim_max_target_distance = int(self.config_Aim.get("max_target_distance", 150))

        # Mouse
        self.config_Mouse = self.config["Mouse"]
        self.mouse_move = self.config_Mouse["mouse_move"]
        self.mouse_dpi = int(self.config_Mouse["mouse_dpi"])
        self.mouse_sensitivity = float(self.config_Mouse["mouse_sensitivity"])
        self.mouse_fov_width = int(self.config_Mouse["mouse_fov_width"])
        self.mouse_fov_height = int(self.config_Mouse["mouse_fov_height"])

        if verbose:
            logger.info("[Config] Config reloaded")


cfg = Config()
