import logging
import time
import os

from util.args import hargs

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

date = time.strftime('%Y%m%d', time.localtime(time.time()))
log_path = 'logs/' + date
if not os.path.exists(log_path):
    os.mkdir(log_path)
rq = time.strftime('%H:%M', time.localtime(time.time()))
log_path = log_path + '/' + rq + '.log'
# File log
fh = logging.FileHandler(log_path, mode='w+')
# File log fix to DEBUG -- save all logs
fh.setLevel(logging.DEBUG)
# Console log
ch = logging.StreamHandler()
hargs.DEBUG = True
if hargs.DEBUG:
    ch.setLevel(logging.DEBUG)
elif hargs.INFO:
    ch.setLevel(logging.INFO)
else:
    ch.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s() - %(levelname)s: %(message)s")  # noqa
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
