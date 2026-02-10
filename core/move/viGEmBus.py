import time

import vgamepad


class ViGEmBus:
    _device = None

    @classmethod
    def get_device(cls):
        if cls._device is None:
            cls._device = vgamepad.VX360Gamepad()
            cls._device.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)
            cls._device.update()
        return cls._device

    @classmethod
    def move(cls, x, y, move_scope=250, move_sleep=0.05):
        cls.get_device().right_joystick_float(x / move_scope, -y / move_scope)
        cls.get_device().update()
        time.sleep(move_sleep)
        cls.get_device().reset()
        cls.get_device().update()
