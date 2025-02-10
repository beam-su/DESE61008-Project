# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from GameTheoryModel import CournotGame, StackelbergGame

# Run simulation for both models
cournot_game = CournotGame(n_players=2, cost_params=[0.5, 0.1], initial_guess=[10, 10], bounds=[(0, 50), (0, 50)])
quantities, price, profits = cournot_game.equilibrium()
print(f"Cournot Equilibrium Quantities: {quantities}, Price: {price}, Profits: {profits}")

stackelberg_game = StackelbergGame(n_players=2, cost_params=[0.5, 0.1], initial_guess=[10, 10], bounds=[(0, 50), (0, 50)])
quantities, price, profits = stackelberg_game.equilibrium()
print(f"Stackelberg Equilibrium Quantities: {quantities}, Price: {price}, Profits: {profits}")