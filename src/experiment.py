import time

from tqdm import tqdm
from replay_buffer import ReplayBuffer

import config


class Experiment:
    def __init__(self, exp_bot, exp_game, exp_display):
        self.exp_bot = exp_bot
        self.exp_game = exp_game
        self.exp_display = exp_display
        self.history = []
        self.replay_buffer = ReplayBuffer(
            action_size=20, buffer_size=2000, batch_size=config.Batch_size, seed=2323)

    def run(self, verbose=False, show=False, epochs=10, train=True):
        if verbose and show:
            raise ValueError(
                'Experiment can either be run in verbose or show mode')

        end_states = []
        try:
            epoch_iter = tqdm(range(epochs)) if verbose else range(epochs)
            for epoch in epoch_iter:
                self.exp_bot.epoch = epoch
                self.exp_bot.train = train
                self.exp_game.start()
                if show:
                    self.exp_display.start()
                if verbose:
                    pbar = tqdm()
                while True:
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
                    act = self.exp_bot.choose_action(state, self.replay_buffer)
                    self.exp_game.do_action(act)
                    if verbose:
                        pbar.update(1)
                    remaining = config.MOVE_DELAY - (time.time() - start_time)
                    if remaining > 0:
                        time.sleep(remaining)
                    self.replay_buffer.add(self.exp_bot.prev_state, self.exp_bot.prev_act,
                                           self.exp_bot.prev_reward, self.exp_bot.parse_state(state), False)
                self.replay_buffer.add(self.exp_bot.prev_state, self.exp_bot.prev_act,
                                       self.exp_bot.prev_reward, self.exp_bot.parse_state(state), True)
                if verbose:
                    tqdm.write('Epoch {}: {}'.format(epoch, state['score']))
                    pbar.close()
                self.history.append(state['score'])
                end_states.append(state)
        except Exception as e:
            raise e
        finally:
            if show:
                self.exp_display.stop()
        return end_states
