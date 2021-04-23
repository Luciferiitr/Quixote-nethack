import random
import string
import collections

import action
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import calculate_input_tensor, decimal_to_binary_state


class DeepQNetwork(nn.Module):
    def __init__(self, action_space_shape, num_in_features, intermediate_features_shape=256):
        super(DeepQNetwork, self).__init__()
        self.fc_1 = nn.Linear(action_space_shape +
                              num_in_features, intermediate_features_shape)
        self.fc_2 = nn.Linear(intermediate_features_shape,
                              intermediate_features_shape)
        self.fc_3 = nn.Linear(intermediate_features_shape,
                              intermediate_features_shape)
        self.fc_4 = nn.Linear(intermediate_features_shape,
                              intermediate_features_shape)
        self.dropout_layer = nn.Dropout(0.5)
        self.output_layer = nn.Linear(intermediate_features_shape, 1)

    def forward(self, x):
        out = self.fc_1(x)
        out = F.relu(out)
        out = self.fc_2(out)
        # out = self.dropout_layer(x)
        out = F.relu(out)
        out = self.fc_3(out)
        out = F.relu(out)
        # out = self.dropout_layer(x)
        out = self.fc_4(out)
        out = F.relu(out)
        out = self.output_layer(out)
        return out


class QLearningBot:
    PATTERNS = [string.ascii_letters, '+', '>', '-', '|', ' ', '#']

    def __init__(self, lr=0.2, epsilon=0.1, discount=0.6):
        self.prev_state = None
        self.prev_act = None
        self.prev_reward = None
        self.prev_map = None
        self.prev_poses = []
        self.prev_level = None
        self.prev_Q = None
        self.beneath = None
        self.prev_discovered = False
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.state_act_counts = collections.defaultdict(int)
        self.Q = collections.defaultdict(float)

    def find_self(self, state_map):
        for y in range(len(state_map)):
            for x in range(len(state_map[0])):
                if state_map[y][x] == '@':
                    return x, y
        return None

    def get_neighbors(self, state_map, x, y):
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                try:
                    neighbors.append(state_map[y + dy][x + dx])
                except IndexError:
                    neighbors.append(' ')
        return neighbors

    def update_prev_map(self, new_map):
        replaced_map = self.prev_map
        self.prev_map = new_map
        for line, row in enumerate(self.prev_map):
            if '@' in row:
                if replaced_map is None:
                    beneath = '.'
                else:
                    beneath = replaced_map[line][row.index('@')]
                self.prev_map[line] = row.replace('@', beneath)
                break
        if replaced_map != self.prev_map:
            self.prev_discovered = True

    def parse_state(self, state):
        pos = self.find_self(state['map'])
        if pos is None or self.prev_map is None:
            parsed = None
        else:
            parsed = []
            x, y = pos
            self.beneath = self.prev_map[y][x]
            neighbors = self.get_neighbors(state['map'], x, y)
            for pattern in self.PATTERNS:
                for neighbor in neighbors:
                    parsed.append(neighbor in pattern)
                parsed.append(self.beneath in pattern)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    parsed.append((x + dx, y + dy) in self.prev_poses)

        self.update_prev_map(state['map'])
        if parsed is None:
            return None
        binary_rep = ''.join(['1' if part else '0' for part in parsed])
        return int(binary_rep, 2)

    def update_Q(self, parsed_state):
        state_act_pair = (self.prev_state, self.prev_act)
        self.state_act_counts[state_act_pair] += 1
        # / float(self.state_act_counts[state_act_pair])
        state_act_lr = self.lr
        if self.prev_state is not None:
            self.prev_Q = self.Q[state_act_pair]
            max_Q = max([self.Q[(parsed_state, act)]
                         for act in action.MOVE_ACTIONS])
            new_Q = (1 - state_act_lr) * self.prev_Q
            new_Q += state_act_lr * (self.prev_reward + self.discount * max_Q)
            self.Q[state_act_pair] = new_Q

    def modify_reward(self, reward, pos, level):
        if pos in self.prev_poses:
            reward -= 0.5
        if self.prev_discovered:
            reward += 5
            self.prev_discovered = False
        if self.prev_level is not None and level is not None:
            reward += 50 * (level - self.prev_level)
        return reward - 0.1

    def choose_action(self, state):
        pos = self.find_self(state['map'])
        parsed_state = self.parse_state(state)
        self.update_Q(parsed_state)

        if state['message']['is_more']:
            act = action.Action.MORE
        elif state['message']['is_yn']:
            act = (action.Action.NO if 'Beware' in state['message']['text']
                   else action.Action.YES)

        else:
            if random.random() < self.epsilon:
                act = random.choice(action.MOVE_ACTIONS)
            else:
                best_actions = None
                best_Q = None
                for new_act in action.MOVE_ACTIONS:
                    new_Q = self.Q[(parsed_state, new_act)]
                    if best_Q is None or new_Q > best_Q:
                        best_actions = [new_act]
                        best_Q = new_Q
                    elif new_Q == best_Q:
                        best_actions.append(new_act)
                act = random.choice(best_actions)
        self.prev_state = parsed_state
        self.prev_act = act
        level = state['Dlvl'] if 'Dlvl' in state else None
        self.prev_reward = self.modify_reward(state['reward'], pos, level)
        self.prev_poses.append(pos)
        self.prev_level = level
        return act

    def get_status(self):
        train_string = 'TRAIN' if self.train else 'TEST'
        status = '{}\tEP:{}'.format(train_string, self.epoch)
        if self.prev_state is not None and self.prev_Q is not None:
            status += '\tQ:{:.3f}\tR:{:.3f}\n\tST:{:018x}'.format(
                self.Q[(self.prev_state, self.prev_act.name)],
                self.prev_reward, self.prev_state)
        status += '\n'
        if self.beneath is not None:
            status += '\tBN:{}'.format(self.beneath)
        if self.prev_act is not None:
            status += '\t{}'.format(self.prev_act)
        status += '\n'
        for act in action.MOVE_ACTIONS:
            status += '\n\t{}:{:.3f}'.format(act.name,
                                             self.Q[(self.prev_state, act)])
        return status


class DQNLearningBot:
    PATTERNS = [string.ascii_letters, '+', '>', '-', '|', ' ', '#']

    def __init__(self, lr=0.001, epsilon=0.1, discount=0.6):
        self.prev_state = None
        self.prev_act = None
        self.prev_reward = None
        self.prev_map = None
        self.prev_poses = []
        self.prev_level = None
        self.prev_Q = None
        self.beneath = None
        self.prev_discovered = False
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.state_act_counts = collections.defaultdict(int)
        #self.Q = collections.defaultdict(float)
        self.Q = DeepQNetwork(1, 72)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.Q = self.Q.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

    def find_self(self, state_map):
        for y in range(len(state_map)):
            for x in range(len(state_map[0])):
                if state_map[y][x] == '@':
                    return x, y
        return None

    def get_neighbors(self, state_map, x, y):
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                try:
                    neighbors.append(state_map[y + dy][x + dx])
                except IndexError:
                    neighbors.append(' ')
        return neighbors

    def update_prev_map(self, new_map):
        replaced_map = self.prev_map
        self.prev_map = new_map
        for line, row in enumerate(self.prev_map):
            if '@' in row:
                if replaced_map is None:
                    beneath = '.'
                else:
                    beneath = replaced_map[line][row.index('@')]
                self.prev_map[line] = row.replace('@', beneath)
                break
        if replaced_map != self.prev_map:
            self.prev_discovered = True

    def parse_state(self, state):
        pos = self.find_self(state['map'])
        if pos is None or self.prev_map is None:
            parsed = None
        else:
            parsed = []
            x, y = pos
            self.beneath = self.prev_map[y][x]
            neighbors = self.get_neighbors(state['map'], x, y)
            for pattern in self.PATTERNS:
                for neighbor in neighbors:
                    parsed.append(neighbor in pattern)
                parsed.append(self.beneath in pattern)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    parsed.append((x + dx, y + dy) in self.prev_poses)

        self.update_prev_map(state['map'])
        if parsed is None:
            return None
        binary_rep = ''.join(['1' if part else '0' for part in parsed])
        return int(binary_rep, 2)

    def update_Q(self, parsed_state):
        state_act_pair = (self.prev_state, self.prev_act)
        self.state_act_counts[state_act_pair] += 1
        # / float(self.state_act_counts[state_act_pair])
        state_act_lr = self.lr
        if self.prev_state is not None and parsed_state is not None:
            #self.prev_Q = self.Q[state_act_pair]
            self.optimizer.zero_grad()

            self.prev_Q = self.Q(calculate_input_tensor(
                self.prev_state, self.prev_act, self.device))  # Q(s,a)

            max_Q = max([self.Q(calculate_input_tensor(parsed_state, act, self.device))
                         for act in action.MOVE_ACTIONS])
            # new_Q = (1 - state_act_lr) * self.prev_Q
            # new_Q += state_act_lr * (self.prev_reward + self.discount * max_Q)
            # print('hiiiiiiiiiiii ', self.prev_Q, max_Q)
            loss = self.criterion(
                self.prev_Q, self.prev_reward + self.discount * max_Q)
            loss.backward()
            self.optimizer.step()
            #self.Q[state_act_pair] = new_Q

    def modify_reward(self, reward, pos, level):
        if pos in self.prev_poses:
            reward -= 0.5
        if self.prev_discovered:
            reward += 5
            self.prev_discovered = False
        if self.prev_level is not None and level is not None:
            reward += 50 * (level - self.prev_level)
        return reward - 0.1

    def choose_action(self, state):
        pos = self.find_self(state['map'])
        parsed_state = self.parse_state(state)
        self.update_Q(parsed_state)

        if state['message']['is_more']:
            act = action.Action.MORE
        elif state['message']['is_yn']:
            act = (action.Action.NO if 'Beware' in state['message']['text']
                   else action.Action.YES)

        else:
            if parsed_state is None:
                print(self.prev_map)
                print(state['map'])
            if random.random() < self.epsilon:
                act = random.choice(action.MOVE_ACTIONS)
            else:
                best_actions = None
                best_Q = None
                # print(action.MOVE_ACTIONS)
                for new_act in action.MOVE_ACTIONS:
                    new_Q = self.Q(calculate_input_tensor(
                        parsed_state, new_act, self.device))

                    if best_Q is None or new_Q > best_Q:
                        best_actions = [new_act]
                        best_Q = new_Q
                    elif new_Q == best_Q:
                        best_actions.append(new_act)
                act = random.choice(best_actions)
        self.prev_state = parsed_state
        self.prev_act = act
        level = state['Dlvl'] if 'Dlvl' in state else None
        self.prev_reward = self.modify_reward(state['reward'], pos, level)
        self.prev_poses.append(pos)
        self.prev_level = level
        return act

    def get_status(self):
        train_string = 'TRAIN' if self.train else 'TEST'
        status = '{}\tEP:{}'.format(train_string, self.epoch)

        if self.prev_state is not None and self.prev_Q is not None:
            status += '\tQ:{:.3f}\tR:{:.3f}\n\tST:{:018x}'.format(
                self.Q(calculate_input_tensor(self.prev_state,
                                              self.prev_act, self.device)).data.cpu().numpy(),
                self.prev_reward, self.prev_state)
        status += '\n'
        if self.beneath is not None:
            status += '\tBN:{}'.format(self.beneath)
        if self.prev_act is not None:
            status += '\t{}'.format(self.prev_act)
        status += '\n'
        for act in action.MOVE_ACTIONS:
            status += '\n\t{}:{:.3f}'.format(act.name,
                                             self.Q(calculate_input_tensor(self.prev_state, act, self.device)))
        return status