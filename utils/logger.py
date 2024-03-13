import logging
import colorlog
import os
from datetime import datetime

def cast_level(level='info'):
    # level to lowercase 
    level = level.lower()
    if level == 'debug':
        return logging.DEBUG
    elif level == 'info':
        return logging.INFO
    elif level == 'warning':
        return logging.WARNING
    elif level == 'error':
        return logging.ERROR
    elif level == 'critical':
        return logging.CRITICAL
    else:
        raise ValueError('level must be in [debug, info, warning, error, critical]')

LOGGER = None

def get_logger():
    global LOGGER
    if LOGGER is None:
        now = datetime.now()
        name = now.strftime("%Y-%m-%d_%H:%M:%S")
        level = os.environ.get("LOGGING_LEVEL", "info")
        level = cast_level(level)
        logger = logging.getLogger(name)
        logger.setLevel(level)

        log_file_name = 'log/{}.log'.format(name)
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(level)

        sh = logging.StreamHandler()
        sh.setLevel(level)

        log_format = '[%(levelname)s]%(asctime)s: %(message)s'
        formatter = logging.Formatter(log_format)
        
        log_colors = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
        colored_formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)s]%(asctime)s: %(message)s',
            log_colors=log_colors
        )

        fh.setFormatter(formatter)
        sh.setFormatter(colored_formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)

        LOGGER = logger
    return LOGGER


if __name__ == '__main__':
    import time
    logger = get_logger()
    logger.info('test_info')
    logger.warning('test_warning')
    try:
        1 / 0 
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        print('aaa')