import logging
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

class dlogger:
    '''
    Custom logger class.
    
    usage:
    import dlogging
    from dlogging import dlogger
    
    dlog = dlogger("logger_name", dlogging.DEBUG, True)

    dlog.debug('test: {}', 'test_string')
    dlog.debug('test: {}', 'test_string', flag=True)

    channels_first flag is only used for nd array of shape size 3 or 4.
    for 3: 0 and 2 are swapped
    for 4: 1 and 3 are swapped
    '''
    def __init__(self, name, level = DEBUG, enabled=False):
        logging.basicConfig()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.enabled = enabled

    def __handle_chanels_first(self, fmt, channels_first):
        if not channels_first:
            return fmt
        
        if not isinstance(fmt, np.ndarray):
            return fmt
        
        if len(fmt.shape) == 3:
            fmt = fmt.transpose(2, 0, 1)

        if len(fmt.shape) == 4:
            fmt = fmt.transpose(0, 3, 1, 2)

        return fmt


    def debug(self, fmt, *args, flag=False, channels_first=False):
        fmt = self.__handle_chanels_first(fmt, channels_first)
        if self.enabled or flag:
            if len(args) == 0:
                self.logger.debug(fmt)
            else:
                self.logger.debug(fmt.format(*args))
    
    def error(self, fmt, *args, flag=False, channels_first=False):
        self.__handle_chanels_first(fmt, channels_first)
        if self.enabled or flag:
            if len(args) == 0:
                self.logger.error(fmt)
            else:    
                self.logger.error(fmt.format(*args))

    def info(self, fmt, *args, flag=False, channels_first=False):
        self.__handle_chanels_first(fmt, channels_first)
        if self.enabled or flag:
            if len(args) == 0:
                self.logger.info(fmt)
            else:
                self.logger.info(fmt.format(*args))

    def warn(self, fmt, *args, flag=False, channels_first=False):
        self.__handle_chanels_first(fmt, channels_first)
        if self.enabled or flag:
            if len(args) == 0:
                self.logger.info(fmt)
            else:    
                self.logger.info(fmt.format(*args))

    def imshow(self, img, flag=False, save=False, path=None):
        if self.enabled or flag:
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
                img = Image.fromarray(img, 'RGB')

            plt.imshow(img)
            plt.show()

        if save and path != None:
            img.save(path)
