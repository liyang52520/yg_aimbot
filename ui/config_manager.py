import os
from typing import Dict, Any

from core.services.config_service import config_service


class UIConfigManager:
    """UI配置管理器 - 适配新的配置服务"""

    def load_config(self, ui_components: Dict[str, Any]) -> bool:
        """加载配置到UI组件"""
        try:
            capture_cfg = config_service.get_section('capture')
            ai_cfg = config_service.get_section('ai')
            aim_cfg = config_service.get_section('aim')
            mouse_cfg = config_service.get_section('mouse')

            ui_components['ai_model_name'].setCurrentText(
                ai_cfg.get('model_name', 'YOLOv8s_apex_teammate_enemy.engine')
            )
            ui_components['ai_conf'].setValue(ai_cfg.get('conf', 0.2))
            ui_components['ai_device'].setValue(int(ai_cfg.get('device', '0')))

            ui_components['capture_window_width'].setValue(capture_cfg.get('window_width', 320))
            ui_components['capture_window_height'].setValue(capture_cfg.get('window_height', 320))
            ui_components['capture_fps'].setValue(capture_cfg.get('fps', 60))
            ui_components['capture_circle'].setChecked(capture_cfg.get('circle', True))
            ui_components['capture_ai_debug'].setChecked(capture_cfg.get('ai_debug', False))

            ui_components['auto'].setChecked(aim_cfg.get('auto', False))
            ui_components['aim_mode'].setCurrentText(aim_cfg.get('mode', 'hold'))
            ui_components['target_cls'].setCurrentText(str(int(aim_cfg.get('target_cls', 1.0))))
            ui_components['body_x_offset'].setValue(aim_cfg.get('body_x_offset', 0.1))
            ui_components['body_y_offset'].setValue(aim_cfg.get('body_y_offset', 0.1))

            hotkeys_str = aim_cfg.get('hotkeys', 'X1MouseButton,X2MouseButton')
            hotkeys_list = hotkeys_str.split(',')
            ui_components['hotkeys'].setSelectedItems(hotkeys_list)
            ui_components['max_target_distance'].setValue(aim_cfg.get('max_target_distance', 90))

            ui_components['mouse_move'].setCurrentText(mouse_cfg.get('move', 'makcu'))
            ui_components['mouse_dpi'].setValue(mouse_cfg.get('dpi', 1100))
            ui_components['mouse_sensitivity'].setValue(mouse_cfg.get('sensitivity', 3.0))
            ui_components['mouse_fov_width'].setValue(mouse_cfg.get('fov_width', 40))
            ui_components['mouse_fov_height'].setValue(mouse_cfg.get('fov_height', 40))

            return True
        except Exception as e:
            print(f"加载配置失败: {e}")
            return False

    def save_config(self, ui_components: Dict[str, Any]):
        """保存UI组件的值到配置"""
        try:
            config_service.update_section('ai', {
                'model_name': ui_components['ai_model_name'].currentText(),
                'conf': ui_components['ai_conf'].value(),
                'device': str(ui_components['ai_device'].value())
            })

            config_service.update_section('capture', {
                'window_width': ui_components['capture_window_width'].value(),
                'window_height': ui_components['capture_window_height'].value(),
                'fps': ui_components['capture_fps'].value(),
                'circle': ui_components['capture_circle'].isChecked(),
                'ai_debug': ui_components['capture_ai_debug'].isChecked()
            })

            config_service.update_section('aim', {
                'auto': ui_components['auto'].isChecked(),
                'mode': ui_components['aim_mode'].currentText(),
                'target_cls': float(ui_components['target_cls'].currentText()),
                'body_x_offset': ui_components['body_x_offset'].value(),
                'body_y_offset': ui_components['body_y_offset'].value(),
                'hotkeys': ','.join(ui_components['hotkeys'].getSelectedItems()),
                'max_target_distance': ui_components['max_target_distance'].value()
            })

            config_service.update_section('mouse', {
                'move': ui_components['mouse_move'].currentText(),
                'dpi': ui_components['mouse_dpi'].value(),
                'sensitivity': ui_components['mouse_sensitivity'].value(),
                'fov_width': ui_components['mouse_fov_width'].value(),
                'fov_height': ui_components['mouse_fov_height'].value()
            })

            config_service.save()
        except Exception as e:
            print(f"保存配置失败: {e}")

    def apply_config_to_memory(self, ui_components: Dict[str, Any]) -> bool:
        """将配置应用到内存"""
        try:
            config_service.update_section('ai', {
                'model_name': ui_components['ai_model_name'].currentText(),
                'conf': ui_components['ai_conf'].value(),
                'device': str(ui_components['ai_device'].value())
            })

            config_service.update_section('capture', {
                'window_width': ui_components['capture_window_width'].value(),
                'window_height': ui_components['capture_window_height'].value(),
                'fps': ui_components['capture_fps'].value(),
                'circle': ui_components['capture_circle'].isChecked(),
                'ai_debug': ui_components['capture_ai_debug'].isChecked()
            })

            config_service.update_section('aim', {
                'auto': ui_components['auto'].isChecked(),
                'mode': ui_components['aim_mode'].currentText(),
                'target_cls': float(ui_components['target_cls'].currentText()),
                'body_x_offset': ui_components['body_x_offset'].value(),
                'body_y_offset': ui_components['body_y_offset'].value(),
                'hotkeys': ','.join(ui_components['hotkeys'].getSelectedItems()),
                'max_target_distance': ui_components['max_target_distance'].value()
            })

            config_service.update_section('mouse', {
                'move': ui_components['mouse_move'].currentText(),
                'dpi': ui_components['mouse_dpi'].value(),
                'sensitivity': ui_components['mouse_sensitivity'].value(),
                'fov_width': ui_components['mouse_fov_width'].value(),
                'fov_height': ui_components['mouse_fov_height'].value()
            })

            return True
        except Exception as e:
            print(f"应用配置失败: {e}")
            return False
