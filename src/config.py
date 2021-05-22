import os
from pickle import FALSE
import datetime

curr_dir = os.getcwd()

COMMAND = curr_dir + '/../bin/nethack'
OPTIONS_FILE = curr_dir + '/../quixote.nethackrc'

WIDTH, HEIGHT = 80, 24
READ_TIMEOUT = 0.001
MOVE_DELAY = 0
Batch_size = 4
SAVE_EPOCH = 10
PATH = curr_dir + '/../checkpoint'
LOAD_MODEL = True
REWARD_FILE = curr_dir + '/../scores' + str(datetime.datetime.now())
TARGET_UPDATE = 5
