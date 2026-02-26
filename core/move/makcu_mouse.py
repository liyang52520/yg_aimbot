import logging

logger = logging.getLogger(__name__)

# 尝试导入makcu库，如果失败则使用模拟实现
try:
    from makcu import create_controller
    _MAKCU_AVAILABLE = True
except ImportError:
    logger.warning("makcu库未安装，使用模拟鼠标控制")
    _MAKCU_AVAILABLE = False
    
    # 模拟实现
    class MockController:
        def move(self, x, y):
            logger.debug(f"[模拟] 鼠标移动: ({x}, {y})")
    
    def create_controller(auto_reconnect=True):
        return MockController()


class MakcuMouse:
    """
    Makcu鼠标控制器封装类
    提供线程安全的鼠标移动控制
    """
    _device = None
    _scope = 20
    _lock = None
    _initialized = False

    @classmethod
    def _get_lock(cls):
        """延迟初始化锁对象"""
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def get_device(cls):
        """获取或创建控制器实例（线程安全）"""
        if cls._device is None:
            with cls._get_lock():
                # 双重检查锁定
                if cls._device is None:
                    try:
                        cls._device = create_controller(auto_reconnect=True)
                        cls._initialized = True
                        logger.info("Makcu鼠标控制器初始化成功")
                    except Exception as e:
                        logger.error(f"Makcu鼠标控制器初始化失败: {e}")
                        # 使用模拟控制器作为fallback
                        cls._device = create_controller(auto_reconnect=True)
                        cls._initialized = False
        return cls._device

    @classmethod
    def move(cls, x: int, y: int) -> bool:
        """
        移动鼠标
        
        Args:
            x: X轴移动距离
            y: Y轴移动距离
            
        Returns:
            bool: 移动是否成功
        """
        try:
            # 限制移动范围
            x = max(-cls._scope, min(x, cls._scope))
            y = max(-cls._scope, min(y, cls._scope))
            
            device = cls.get_device()
            if device is None:
                logger.error("鼠标控制器未初始化")
                return False
                
            device.move(x, y)
            return True
        except Exception as e:
            logger.error(f"鼠标移动失败: {e}")
            # 重置设备，下次调用会重新初始化
            cls._device = None
            return False

    @classmethod
    def is_initialized(cls) -> bool:
        """检查控制器是否已初始化"""
        return cls._initialized

    @classmethod
    def reset(cls):
        """重置控制器状态"""
        with cls._get_lock():
            cls._device = None
            cls._initialized = False
            logger.info("Makcu鼠标控制器已重置")
