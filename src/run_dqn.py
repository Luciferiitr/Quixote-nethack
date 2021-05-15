import game
import bot
import display
import experiment
import matplotlib.pyplot as plt

NUM_EPOCHS = 50

exp_game = game.Game()
exp_display = display.Display()
exp_bot = bot.DQNLearningBot(discount=0.4)
exp = experiment.Experiment(exp_bot, exp_game, exp_display)
# exp.run(show=True, verbose=False, epochs=NUM_EPOCHS)
exp.run(verbose=False, epochs=NUM_EPOCHS)
x_values = range(NUM_EPOCHS)
print("history")
print(exp.history)
plt.plot(x_values, exp.history)
plt.show()
