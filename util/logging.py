import logging
import os.path
import time

from util.args import hargs

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_name = os.path.dirname(os.getcwd()) + '/logs/' + rq + '.log'
# File log
fh = logging.FileHandler(log_name, mode='w')
# File log fix to DEBUG -- save all logs
fh.setLevel(logging.DEBUG)
# Console log
ch = logging.StreamHandler()
if hargs.DEBUG:
    ch.setLevel(logging.DEBUG)
elif hargs.INFO:
    ch.setLevel(logging.INFO)
else:
    ch.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")  # noqa
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
