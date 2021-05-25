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
        self.fc_1 = nn.Linear(num_in_features, intermediate_features_shape)
        self.fc_2 = nn.Linear(intermediate_features_shape,
                              intermediate_features_shape)
        self.fc_3 = nn.Linear(intermediate_features_shape,
                              intermediate_features_shape)
        self.fc_4 = nn.Linear(intermediate_features_shape,
                              intermediate_features_shape)
        self.dropout_layer = nn.Dropout(0.5)
        self.output_layer = nn.Linear(
            intermediate_features_shape, action_space_shape)

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

    def __init__(self, lr=0.001, epsilon=0.9, discount=0.6):
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
        # self.Q = collections.defaultdict(float)
        self.Q = DeepQNetwork(20, 72)
        self.Q_target = DeepQNetwork(20, 72)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.Q = self.Q.to(self.device)
        self.Q_target = self.Q_target.to(self.device)
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

    def parse_state(self, state, update=True):
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
        if update:
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
            # self.prev_Q = self.Q[sTrue
            self.prev_Q = self.Q(calculate_input_tensor(
                self.prev_state, self.device))[0, action.map_act_int[self.prev_act]].view(1, 1)  # Q(s,a)
            # print("self.Q(calculate_input_tensor(next_states, self.device)).max(1) ", self.Q(calculate_input_tensor(parsed_state, self.device)).max(1)[0].shape)
            max_Q = self.Q(calculate_input_tensor(parsed_state, self.device)).max(1)[
                0].detach().view(1, 1)
            # new_Q = (1 - state_act_lr) * self.prev_Q
            # new_Q += state_act_lr * (self.prev_reward + self.discount * max_Q)
            # print('hiiiiiiiiiiii ', self.prev_Q, max_Q)
            loss = self.criterion(
                self.prev_Q, self.prev_reward + self.discount * max_Q)
            loss.backward()
            self.optimizer.step()
            #self.Q[state_act_pair] = new_Q

    def update_Q_replay_buffer(self, replay_buffer):
        if len(replay_buffer) < replay_buffer.batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        state_action_values = self.Q(states).gather(
            1, actions).view(4, 1)  # (4,1)
        # print('state_action_values ' ,state_action_values.shape)
        next_state_values = self.Q_target(next_states).max(1)[
            0].detach().view(4, 1)
        # print('next state values  ', next_state_values.shape)
        expected_state_action_values = (
            next_state_values * self.discount) + rewards.float()
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def modify_reward(self, reward, pos, level):
        if pos in self.prev_poses:
            reward -= 0.5
        if self.prev_discovered:
            reward += 5
            self.prev_discovered = False
        if self.prev_level is not None and level is not None:
            reward += 50 * (level - self.prev_level)
        return reward - 0.1

    def choose_action(self, state, replay_buffer):
        pos = self.find_self(state['map'])
        parsed_state = self.parse_state(state)
        # self.update_Q(parsed_state)
        if len(replay_buffer) >= replay_buffer.batch_size:
            self.update_Q_replay_buffer(replay_buffer)
        else:
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
                return -1
            if random.random() < self.epsilon:
                act = random.choice(action.MOVE_ACTIONS)
            else:
                best_actions = None
                best_Q = None
                # print(action.MOVE_ACTIONS)
                for new_act in action.MOVE_ACTIONS:
                    # print(calculate_input_tensor(parsed_state, self.device).shape)
                    new_Q = self.Q(calculate_input_tensor(
                        parsed_state, self.device))
                    # print("new Q shape", new_Q.shape)
                    new_Q = new_Q[0, action.map_act_int[new_act]].view(1)

                    if best_Q is None or new_Q > best_Q:
                        best_actions = [new_act]
                        best_Q = new_Q
                    elif new_Q == best_Q:
                        best_actions.append(new_act)
                act = random.choice(best_actions)
                # act = self.Q(calculate_input_tensor(parsed_state, self.device)).argmax()
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
            print('both not none')
            status += '\tQ:{:.3f}\tR:{:.3f}\n\tST:{:018x}'.format(
                self.Q(calculate_input_tensor(self.prev_state, self.device))[
                    0, action.map_act_int[self.prev_act]].data.cpu().numpy(),
                self.prev_reward, self.prev_state)
        if self.prev_state is not None:
            status += '\n'
            if self.beneath is not None:
                status += '\tBN:{}'.format(self.beneath)
            if self.prev_act is not None:
                status += '\t{}'.format(self.prev_act)
            status += '\n'
            for act in action.MOVE_ACTIONS:
                status += '\n\t{}:{:.3f}'.format(act.name,
                                                 self.Q(calculate_input_tensor(self.prev_state, self.device))[
                                                     0, action.map_act_int[act]].data.cpu().numpy())
        return status
