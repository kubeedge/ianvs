import logging
import colorlog


class Logger:
    """
    Deafult logger in ianvs
    Args:
        name(str) : Logger name, default is 'ianvs'
    """

    def __init__(self, name: str = "ianvs"):
        self.logger = logging.getLogger(name)

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] %(filename)s(%(lineno)d)'
            ' [%(levelname)s]%(reset)s - %(message)s', )

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = 'INFO'
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False


LOGGER = Logger().logger
