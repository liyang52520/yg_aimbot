import configparser
import os


class ConfigManager:
    """配置管理器 - 负责配置的加载、保存和应用"""

    CONFIG_PATH = "config/config.ini"

    def __init__(self):
        self.config = configparser.ConfigParser()

    def load_config(self, ui_components):
        """加载配置到UI组件

        Args:
            ui_components: 包含所有UI控件的字典
        """
        if not os.path.exists(self.CONFIG_PATH):
            return False

        self.config.read(self.CONFIG_PATH)

        # 加载AI配置
        if "AI" in self.config:
            ui_components['ai_model_name'].setCurrentText(
                self.config["AI"].get("ai_model_name", "YOLOv8s_apex_teammate_enemy.engine")
            )
            ui_components['ai_conf'].setValue(
                float(self.config["AI"].get("ai_conf", "0.2"))
            )
            ui_components['ai_device'].setValue(
                int(self.config["AI"].get("ai_device", "0"))
            )

        # 加载Capture配置
        if "Capture" in self.config:
            ui_components['capture_window_width'].setValue(
                int(self.config["Capture"].get("capture_window_width", "320"))
            )
            ui_components['capture_window_height'].setValue(
                int(self.config["Capture"].get("capture_window_height", "320"))
            )
            ui_components['capture_fps'].setValue(
                int(self.config["Capture"].get("capture_fps", "60"))
            )
            ui_components['capture_circle'].setChecked(
                self.config["Capture"].getboolean("capture_circle", True)
            )
            ui_components['capture_ai_debug'].setChecked(
                self.config["Capture"].getboolean("capture_ai_debug", False)
            )

        # 加载Aim配置
        if "Aim" in self.config:
            ui_components['auto'].setChecked(
                self.config["Aim"].getboolean("auto", False)
            )
            ui_components['aim_mode'].setCurrentText(
                self.config["Aim"].get("mode", "hold")
            )
            target_cls_value = str(int(float(self.config["Aim"].get("target_cls", "1.0"))))
            ui_components['target_cls'].setCurrentText(target_cls_value)
            ui_components['body_x_offset'].setValue(
                float(self.config["Aim"].get("body_x_offset", "0.1"))
            )
            ui_components['body_y_offset'].setValue(
                float(self.config["Aim"].get("body_y_offset", "0.1"))
            )
            hotkeys_str = self.config["Aim"].get(
                "hotkeys", "X1MouseButton,X2MouseButton,RightMouseButton,LeftMouseButton"
            )
            hotkeys_list = hotkeys_str.split(",")
            ui_components['hotkeys'].setSelectedItems(hotkeys_list)
            ui_components['max_target_distance'].setValue(
                int(self.config["Aim"].get("max_target_distance", "90"))
            )

        # 加载Mouse配置
        if "Mouse" in self.config:
            ui_components['mouse_move'].setCurrentText(
                self.config["Mouse"].get("mouse_move", "makcu")
            )
            ui_components['mouse_dpi'].setValue(
                int(self.config["Mouse"].get("mouse_dpi", "1100"))
            )
            ui_components['mouse_sensitivity'].setValue(
                float(self.config["Mouse"].get("mouse_sensitivity", "3.0"))
            )
            ui_components['mouse_fov_width'].setValue(
                int(self.config["Mouse"].get("mouse_fov_width", "40"))
            )
            ui_components['mouse_fov_height'].setValue(
                int(self.config["Mouse"].get("mouse_fov_height", "40"))
            )

        return True

    def save_config(self, ui_components):
        """保存UI组件的值到配置文件

        Args:
            ui_components: 包含所有UI控件的字典
        """
        # 保存AI配置
        self.config["AI"] = {
            "ai_model_name": ui_components['ai_model_name'].currentText(),
            "ai_conf": str(ui_components['ai_conf'].value()),
            "ai_device": str(ui_components['ai_device'].value())
        }

        # 保存Capture配置
        self.config["Capture"] = {
            "capture_window_width": str(ui_components['capture_window_width'].value()),
            "capture_window_height": str(ui_components['capture_window_height'].value()),
            "capture_fps": str(ui_components['capture_fps'].value()),
            "capture_circle": str(ui_components['capture_circle'].isChecked()),
            "capture_ai_debug": str(ui_components['capture_ai_debug'].isChecked())
        }

        # 保存Aim配置
        self.config["Aim"] = {
            "auto": str(ui_components['auto'].isChecked()),
            "mode": ui_components['aim_mode'].currentText(),
            "target_cls": ui_components['target_cls'].currentText(),
            "body_x_offset": str(ui_components['body_x_offset'].value()),
            "body_y_offset": str(ui_components['body_y_offset'].value()),
            "hotkeys": ",".join(ui_components['hotkeys'].getSelectedItems()),
            "max_target_distance": str(ui_components['max_target_distance'].value())
        }

        # 保存Mouse配置
        self.config["Mouse"] = {
            "mouse_move": ui_components['mouse_move'].currentText(),
            "mouse_dpi": str(ui_components['mouse_dpi'].value()),
            "mouse_sensitivity": str(ui_components['mouse_sensitivity'].value()),
            "mouse_fov_width": str(ui_components['mouse_fov_width'].value()),
            "mouse_fov_height": str(ui_components['mouse_fov_height'].value())
        }

        # 确保config目录存在
        os.makedirs(os.path.dirname(self.CONFIG_PATH), exist_ok=True)

        # 写入配置文件
        with open(self.CONFIG_PATH, "w") as f:
            self.config.write(f)

    def apply_config_to_memory(self, ui_components):
        """将配置应用到内存中的cfg对象

        Args:
            ui_components: 包含所有UI控件的字典
        """
        try:
            from core.config import cfg

            # 应用AI配置
            cfg.ai_model_name = ui_components['ai_model_name'].currentText()
            # 重新计算模型类型
            cfg.ai_model_type = "yolov5" if  cfg.ai_model_name.lower().startswith("yolov5") else "ultralytics"
            cfg.ai_conf = ui_components['ai_conf'].value()
            cfg.ai_device = str(ui_components['ai_device'].value())

            # 应用Capture配置
            cfg.capture_window_width = ui_components['capture_window_width'].value()
            cfg.capture_window_height = ui_components['capture_window_height'].value()
            cfg.capture_fps = ui_components['capture_fps'].value()
            cfg.capture_circle = ui_components['capture_circle'].isChecked()
            cfg.capture_ai_debug = ui_components['capture_ai_debug'].isChecked()

            # 应用Aim配置
            cfg.aim_auto = ui_components['auto'].isChecked()
            cfg.aim_mode = ui_components['aim_mode'].currentText()
            cfg.aim_target_cls = float(ui_components['target_cls'].currentText())
            cfg.aim_body_x_offset = ui_components['body_x_offset'].value()
            cfg.aim_body_y_offset = ui_components['body_y_offset'].value()
            cfg.aim_hotkeys = ui_components['hotkeys'].getSelectedItems()
            cfg.aim_max_target_distance = ui_components['max_target_distance'].value()

            # 应用Mouse配置
            cfg.mouse_move = ui_components['mouse_move'].currentText()
            cfg.mouse_dpi = ui_components['mouse_dpi'].value()
            cfg.mouse_sensitivity = ui_components['mouse_sensitivity'].value()
            cfg.mouse_fov_width = ui_components['mouse_fov_width'].value()
            cfg.mouse_fov_height = ui_components['mouse_fov_height'].value()

            return True
        except Exception as e:
            print(f"应用配置失败: {e}")
            return False
