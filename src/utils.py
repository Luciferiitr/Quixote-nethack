import torch
import action
import numpy as np


def calculate_input_tensor(binary_state, act, device):
    state = decimal_to_binary_state(binary_state)
    state = torch.tensor(state)
    act_num = action.map_act_int[act]
    act_num = torch.tensor([[act_num]], dtype=float)
    state_action_tensor = torch.cat([state, act_num], axis=1)
    # print(state_action_tensor.to(device))
    return state_action_tensor.to(device).float()


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


# print(type(decimal_to_binary_state(52)))