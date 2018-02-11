import os
import time
import logging


class PrinterLogger:
    new = lambda *args, **kwargs: 1
    info = print
    warning = print
    error = print


class Logger:
    __logger__ = PrinterLogger

    @staticmethod
    def new(logPath, filename=time.strftime("%Y-%m-%d")):
        if Logger.__logger__ is PrinterLogger:
            if not os.path.exists(logPath):
                os.makedirs(logPath)

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            path = os.path.join(logPath, filename + '.log')
            handler = logging.FileHandler(path)
            handler.setFormatter(formatter)

            logger = logging.getLogger('EARL')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

            Logger.__logger__ = logger
    
    @staticmethod
    def info(msg):
        Logger.__logger__.info(msg)

    @staticmethod
    def warning(msg):
        Logger.__logger__.warning(msg)

    @staticmethod
    def error( msg):
        Logger.__logger__.error(msg)


