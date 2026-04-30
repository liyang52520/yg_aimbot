import configparser
import logging
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Any, Callable, List

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """配置节"""
    capture: Dict[str, Any] = field(default_factory=dict)
    ai: Dict[str, Any] = field(default_factory=dict)
    aim: Dict[str, Any] = field(default_factory=dict)
    mouse: Dict[str, Any] = field(default_factory=dict)


class ConfigService:
    """配置服务 - 统一配置管理"""

    DEFAULT_CONFIG = {
        'capture': {
            'window_width': 320,
            'window_height': 320,
            'fps': 60,
            'circle': True,
            'ai_debug': False
        },
        'ai': {
            'model_name': 'YOLOv8s_apex_teammate_enemy.engine',
            'conf': 0.2,
            'device': '0'
        },
        'aim': {
            'auto': False,
            'mode': 'hold',
            'target_cls': 1.0,
            'body_x_offset': 0.1,
            'body_y_offset': 0.1,
            'hotkeys': 'X1MouseButton,X2MouseButton',
            'max_target_distance': 90,
            'max_miss_time': 0.15,
            'max_miss_distance': 120,
            'prediction_history_size': 5
        },
        'mouse': {
            'move': 'makcu',
            'dpi': 1100,
            'sensitivity': 3.0,
            'fov_width': 40,
            'fov_height': 40
        }
    }

    def __init__(self, config_path: str = "./config.ini"):
        self.config_path = config_path
        self._config = configparser.ConfigParser()
        self._cache: ConfigSection = ConfigSection()
        self._lock = Lock()
        self._callbacks: List[Callable] = []
        self._loading = False
        self._load()

    def _load(self):
        """加载配置"""
        with self._lock:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在，使用默认配置: {self.config_path}")
                self._apply_defaults()
                return

            try:
                self._config.read(self.config_path)
                self._parse_config()
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
                self._apply_defaults()

    def _parse_config(self):
        """解析配置到缓存"""
        self._cache.capture = self._parse_section('Capture', self.DEFAULT_CONFIG['capture'])
        self._cache.ai = self._parse_section('AI', self.DEFAULT_CONFIG['ai'])
        self._cache.aim = self._parse_section('Aim', self.DEFAULT_CONFIG['aim'])
        self._cache.mouse = self._parse_section('Mouse', self.DEFAULT_CONFIG['mouse'])

    def _parse_section(self, section: str, defaults: dict) -> Dict[str, Any]:
        """解析配置节"""
        if section not in self._config:
            return defaults.copy()

        result = defaults.copy()
        for key, value in self._config[section].items():
            result[key] = self._convert_value(value, defaults.get(key))
        return result

    def _convert_value(self, value: str, default: Any) -> Any:
        """转换配置值类型"""
        if default is None:
            return value

        if isinstance(default, bool):
            return value.lower() in ('true', 'yes', '1', 'on')
        elif isinstance(default, int):
            return int(value)
        elif isinstance(default, float):
            return float(value)
        return value

    def _apply_defaults(self):
        """应用默认配置"""
        self._cache.capture = self.DEFAULT_CONFIG['capture'].copy()
        self._cache.ai = self.DEFAULT_CONFIG['ai'].copy()
        self._cache.aim = self.DEFAULT_CONFIG['aim'].copy()
        self._cache.mouse = self.DEFAULT_CONFIG['mouse'].copy()

    def save(self):
        """保存配置"""
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

                if 'Capture' not in self._config:
                    self._config['Capture'] = {}
                if 'AI' not in self._config:
                    self._config['AI'] = {}
                if 'Aim' not in self._config:
                    self._config['Aim'] = {}
                if 'Mouse' not in self._config:
                    self._config['Mouse'] = {}

                for key, value in self._cache.capture.items():
                    self._config['Capture'][key] = str(value)
                for key, value in self._cache.ai.items():
                    self._config['AI'][key] = str(value)
                for key, value in self._cache.aim.items():
                    self._config['Aim'][key] = str(value)
                for key, value in self._cache.mouse.items():
                    self._config['Mouse'][key] = str(value)

                with open(self.config_path, 'w', encoding='utf-8') as f:
                    self._config.write(f)

                logger.info("配置已保存")
            except Exception as e:
                logger.error(f"保存配置失败: {e}")

    def get(self, section: str, key: str, default=None):
        """获取配置值"""
        with self._lock:
            section_dict = getattr(self._cache, section.lower(), {})
            return section_dict.get(key, default)

    def set(self, section: str, key: str, value: Any):
        """设置配置值"""
        with self._lock:
            section_dict = getattr(self._cache, section.lower(), {})
            section_dict[key] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取整个配置节"""
        with self._lock:
            return getattr(self._cache, section.lower(), {}).copy()

    def update_section(self, section: str, updates: Dict[str, Any], notify: bool = True):
        """批量更新配置节"""
        with self._lock:
            section_dict = getattr(self._cache, section.lower(), {})
            section_dict.update(updates)
        if notify and not self._loading:
            self._notify_callbacks(section, updates)

    def begin_loading(self):
        """开始批量加载配置，禁用回调"""
        self._loading = True

    def end_loading(self):
        """结束批量加载配置，启用回调"""
        self._loading = False

    def register_callback(self, callback: Callable):
        """注册配置变更回调"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable):
        """注销配置变更回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, section: str, updates: Dict[str, Any]):
        """通知所有回调"""
        for callback in self._callbacks:
            try:
                callback(section, updates)
            except Exception as e:
                logger.error(f"配置回调执行失败: {e}")


config_service = ConfigService()
