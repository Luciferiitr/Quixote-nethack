import game
import bot
import display
import experiment
import matplotlib.pyplot as plt

exp_game = game.Game()
exp_display = display.Display()
exp_bot = bot.QLearningBot(discount=0.4)
exp = experiment.Experiment(exp_bot, exp_game, exp_display)
exp.run(verbose=False, show=True, epochs=50)
x_values = range(50)
plt.plot(x_values, exp.history)
plt.show()
