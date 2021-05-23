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
PATH = curr_dir + '/../logs/checkpoint'
LOAD_MODEL = False
REWARD_FILE = curr_dir + '/../logs/scores' + str(datetime.datetime.now())
EPSILON_FILE = curr_dir + '/../logs/epsilon.txt'
TARGET_UPDATE = 5

#############################   COLAB   ########################################

# PATH = '/content/drive/MyDrive/RL_model/checkpoint'
# REWARD_FILE = '/content/drive/MyDrive/RL_model/scores' + \
#     str(datetime.datetime.now())
# EPSILON_FILE = '/content/drive/MyDrive/RL_model/epsilon.txt'
