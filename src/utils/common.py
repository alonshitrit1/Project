import logging


def get_logger(name: str, file: str = None, level: int = logging.DEBUG):
    logger = logging.getLogger(name=name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if file:
        file_handler = logging.FileHandler(filename=file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
