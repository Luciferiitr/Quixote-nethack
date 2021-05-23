import torch
import action
import torch.nn as nn
import numpy as np
from config import SAVE_EPOCH, PATH, REWARD_FILE, EPSILON_FILE


def calculate_input_tensor(binary_state, device):
    state = decimal_to_binary_state(binary_state)
    state = torch.tensor(state)
    # print('calculate_input_tensor ', state.shape)
    return state.to(device).float()


def decimal_to_binary_state(a, num_elements=72):
    b = []
    while(a > 0):
        if(a % 2 != 0):
            b.append(1)
        else:
            b.append(0)
        a = int(a/2)
    curr = len(b)
    required = num_elements - curr
    while(required > 0):
        b.append(0)
        required -= 1
    c = b[::-1]
    final_state = np.array(c, dtype=np.float32)
    final_state = np.expand_dims(final_state, axis=0)
    return final_state


def get_model_name(root_dir_path, epoch):
    curr_name = "saved_checkpoint_" + str(epoch)
    return curr_name


def save_model(model, epoch, epsilon, path=PATH, save_dict_only=True):
    if epoch % SAVE_EPOCH == 0:
        # print("saving the model")
        f = open(EPSILON_FILE, 'w')
        f.write(str(epsilon))
        if not save_dict_only:
            torch.save(model, path)
        else:
            torch.save(model.state_dict(), path)


def load_model(model_1, path=PATH, save_dict_only=True):
    print('Loading model ................... ')
    if not save_dict_only:
        model = torch.load(path)
        return model
    else:
        model_1.load_state_dict(torch.load(path))
        curr_epsilon_file = open(EPSILON_FILE)
        str_epsilon_list = curr_epsilon_file.readlines()
        epsilon = float(str_epsilon_list[0])
        return model_1, epsilon


def save_rewards(reward_list, file_name=REWARD_FILE):
    curr_new = np.array(reward_list)
    curr_file = file_name + ".npy"
    np.save(curr_file, curr_new)
