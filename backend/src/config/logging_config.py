import logging
import sys


def setup_logging():
    """
    配置全局日志记录器。

    该函数为应用程序设置了一个标准化的日志系统。
    - 日志级别设置为 INFO，意味着 INFO, WARNING, ERROR, CRITICAL 级别的日志都将被记录。
    - 日志格式包含时间戳、日志记录器名称、日志级别和消息本身，便于追踪。
    - 日志直接输出到控制台 (stdout)。
    """
    # 创建一个格式化器，定义日志的输出格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建一个处理器，用于将日志记录发送到标准输出（控制台）
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # 获取根日志记录器，并进行配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 防止重复添加处理器
    if not root_logger.handlers:
        root_logger.addHandler(handler)


# 在模块加载时执行一次日志配置
setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    获取一个指定名称的日志记录器实例。

    参数:
        name (str): 通常是当前模块的名称 (__name__)。

    返回:
        logging.Logger: 配置好的日志记录器实例。
    """
    return logging.getLogger(name)
