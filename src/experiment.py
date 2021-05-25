import time
from torch.serialization import load

from tqdm import tqdm
from replay_buffer import ReplayBuffer

import config
import action
from utils import save_model, load_model, save_rewards


class Experiment:
    def __init__(self, exp_bot, exp_game, exp_display):
        self.exp_bot = exp_bot
        self.exp_game = exp_game
        self.exp_display = exp_display
        self.history = []
        self.replay_buffer = ReplayBuffer(
            action_size=20, buffer_size=2000, batch_size=config.Batch_size, seed=2323)
        self.num_iters = 0

    def run(self, verbose=False, show=False, epochs=10, train=True, scheduling=True):
        if verbose and show:
            raise ValueError(
                'Experiment can either be run in verbose or show mode')

        end_states = []
        try:
            epoch_iter = tqdm(range(epochs)) if verbose else range(epochs)
            if config.LOAD_MODEL:
                self.exp_bot.Q, self.exp_bot.epsilon = load_model(
                    self.exp_bot.Q)
            for epoch in epoch_iter:
                self.exp_bot.epoch = epoch
                self.exp_bot.train = train
                self.exp_game.start()
                if show:
                    self.exp_display.start()
                if verbose:
                    pbar = tqdm()
                save_model(self.exp_bot.Q, epoch, self.exp_bot.epsilon)
                if epoch % config.TARGET_UPDATE == 0:
                    self.exp_bot.Q_target.load_state_dict(
                        self.exp_bot.Q.state_dict())
                count = 0
                while True:
                    count += 1
                    self.num_iters += 1
                    if self.num_iters % 50000 == 0 and scheduling:
                        self.exp_bot.epsilon = max(
                            0.1, self.exp_bot.epsilon - (self.num_iters // 50000) * 0.1)
                    start_time = time.time()
                    if show:
                        game_screen = self.exp_game.get_screen()
                        status = self.exp_bot.get_status()
                        self.exp_display.update(game_screen, status)
                        if not self.exp_display.running:
                            self.exp_game.quit()
                            break
                        if self.exp_display.paused:
                            continue
                    state = self.exp_game.get_state()
                    if not self.exp_game.running:
                        break
                    if count == 15000:
                        self.exp_game.quit()
                        break
                    act = self.exp_bot.choose_action(state, self.replay_buffer)
                    # act = self.exp_bot.choose_action(state)
                    if act == -1:
                        self.exp_game.quit()
                        break
                    self.exp_game.do_action(act)
                    new_state = self.exp_bot.parse_state(
                        self.exp_game.get_state(), update=False)
                    if verbose:
                        pbar.update(1)
                    remaining = config.MOVE_DELAY - (time.time() - start_time)
                    if remaining > 0:
                        time.sleep(remaining)
                    # if self.exp_bot.prev_state != new_state:
                    #     print(self.exp_bot.prev_state, new_state)
                    if self.exp_bot.prev_state and new_state:
                        self.replay_buffer.add(self.exp_bot.prev_state, action.map_act_int[self.exp_bot.prev_act],
                                               self.exp_bot.prev_reward, new_state, False)
                if self.exp_bot.prev_state and new_state:
                    self.replay_buffer.add(self.exp_bot.prev_state, action.map_act_int[self.exp_bot.prev_act],
                                           self.exp_bot.prev_reward, new_state, True)
                if verbose:
                    tqdm.write('Epoch {}: {}'.format(epoch, state['score']))
                    pbar.close()
                self.history.append(state['score'])
                save_rewards(self.history)
                end_states.append(state)
        except Exception as e:
            raise e
        finally:
            if show:
                self.exp_display.stop()
        return end_states
