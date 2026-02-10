from makcu import create_controller


class Makcu:
    _device = None
    _scope = 20

    @classmethod
    def get_device(cls):
        if cls._device is None:
            cls._device = create_controller(auto_reconnect=True)
        return cls._device

    @classmethod
    def move(cls, x, y):
        x = max(-1 * cls._scope, min(x, cls._scope))
        y = max(-1 * cls._scope, min(y, cls._scope))
        cls.get_device().move_smooth(x, y, segments=0.5)
