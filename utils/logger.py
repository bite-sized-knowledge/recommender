import logging

def get_logger(name: str = "bite") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 핸들러가 중복 설정되지 않도록 방지
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
