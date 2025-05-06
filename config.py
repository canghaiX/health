import logging

# config.py

# 数据库配置
DB_HOST = 'localhost'
DB_PORT = 5432
DB_NAME = 'mydatabase'
DB_USER = 'myuser'
DB_PASSWORD = 'mypassword'

# API 密钥
API_KEY = 'your_api_key'

# 调试模式
DEBUG = True

#日志记录
def get_logger(name, log_file=None):
    # 创建一个日志记录器
    logger = logging.getLogger(name)
    # 设置日志级别
    logger.setLevel(logging.DEBUG)
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    # 设置控制台处理器的日志级别
    console_handler.setLevel(logging.DEBUG)
    # 将格式化器添加到控制台处理器
    console_handler.setFormatter(formatter)
    # 将控制台处理器添加到日志记录器
    logger.addHandler(console_handler)

    if log_file:
        # 创建一个文件处理器
        file_handler = logging.FileHandler(log_file)
        # 设置文件处理器的日志级别
        file_handler.setLevel(logging.DEBUG)
        # 将格式化器添加到文件处理器
        file_handler.setFormatter(formatter)
        # 将文件处理器添加到日志记录器
        logger.addHandler(file_handler)

    return logger