import logging


class ColoredFormatter(logging.Formatter):
    """彩色日志格式"""
    COLORS = {
        logging.DEBUG: '\033[94m',  # 蓝色
        logging.INFO: '\033[92m',  # 绿色
        logging.WARNING: '\033[93m',  # 黄色
        logging.ERROR: '\033[91m',  # 红色
        logging.CRITICAL: '\033[95m'  # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        log_format = f'{color}[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s{self.RESET}'
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


# 配置日志
def setup_logger():
    """设置日志配置"""
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除现有处理器
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # 添加新处理器
    root_logger.addHandler(console_handler)


# 自动配置日志
setup_logger()
