from dataclasses import dataclass
from typing import Final

COLOR_PRIMARY: Final = "#4a90e2"
COLOR_PRIMARY_HOVER: Final = "#357abd"
COLOR_PRIMARY_PRESSED: Final = "#2c6aa0"
COLOR_PRIMARY_LIGHT: Final = "#94c6f8"

COLOR_TEXT_PRIMARY: Final = "#333333"
COLOR_TEXT_SECONDARY: Final = "#666666"
COLOR_TEXT_DISABLED: Final = "#999999"

COLOR_BORDER: Final = "#d0d0d0"
COLOR_BORDER_HOVER: Final = "#999999"
COLOR_BORDER_FOCUS: Final = "#4a90e2"

COLOR_BACKGROUND: Final = "#ffffff"
COLOR_BACKGROUND_SECONDARY: Final = "#f5f5f5"
COLOR_BACKGROUND_TERTIARY: Final = "#f0f0f0"
COLOR_BACKGROUND_QUATERNARY: Final = "#f9f9f9"

COLOR_SUCCESS: Final = "#4CAF50"
COLOR_INFO: Final = "#2196F3"

BORDER_RADIUS_SMALL: Final = "2px"
BORDER_RADIUS_MEDIUM: Final = "4px"
BORDER_RADIUS_LARGE: Final = "6px"
BORDER_RADIUS_XLARGE: Final = "8px"

PADDING_SMALL: Final = "6px 8px"
PADDING_MEDIUM: Final = "8px 16px"
PADDING_LARGE: Final = "16px"

FONT_FAMILY: Final = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
FONT_FAMILY_MONO: Final = "'Consolas', 'Monaco', 'Courier New', monospace"
FONT_SIZE_SMALL: Final = "12px"
FONT_SIZE_MEDIUM: Final = "13px"
FONT_SIZE_LARGE: Final = "14px"


@dataclass
class ColorScheme:
    primary: str = COLOR_PRIMARY
    primary_hover: str = COLOR_PRIMARY_HOVER
    primary_pressed: str = COLOR_PRIMARY_PRESSED
    primary_light: str = COLOR_PRIMARY_LIGHT
    text_primary: str = COLOR_TEXT_PRIMARY
    text_secondary: str = COLOR_TEXT_SECONDARY
    text_disabled: str = COLOR_TEXT_DISABLED
    border: str = COLOR_BORDER
    border_hover: str = COLOR_BORDER_HOVER
    border_focus: str = COLOR_BORDER_FOCUS
    background: str = COLOR_BACKGROUND
    background_secondary: str = COLOR_BACKGROUND_SECONDARY
    background_tertiary: str = COLOR_BACKGROUND_TERTIARY
    background_quaternary: str = COLOR_BACKGROUND_QUATERNARY
    success: str = COLOR_SUCCESS
    info: str = COLOR_INFO


@dataclass
class Spacing:
    padding_small: str = PADDING_SMALL
    padding_medium: str = PADDING_MEDIUM
    padding_large: str = PADDING_LARGE
    border_radius_small: str = BORDER_RADIUS_SMALL
    border_radius_medium: str = BORDER_RADIUS_MEDIUM
    border_radius_large: str = BORDER_RADIUS_LARGE
    border_radius_xlarge: str = BORDER_RADIUS_XLARGE


@dataclass
class Typography:
    family: str = FONT_FAMILY
    family_mono: str = FONT_FAMILY_MONO
    size_small: str = FONT_SIZE_SMALL
    size_medium: str = FONT_SIZE_MEDIUM
    size_large: str = FONT_SIZE_LARGE


class Theme:
    colors = ColorScheme()
    spacing = Spacing()
    typography = Typography()
