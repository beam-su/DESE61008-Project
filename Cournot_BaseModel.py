'''
Base Model for Cournot Game

This script contains the base model for a Cournot game. It defines the CournotGame class, which is a subclass of the GameTheoryModel class. The CournotGame class implements the demand function, cost function, best response function, and equilibrium computation for a Cournot game with n players.

This script uses the initial parameters given in the project brief to show that our script does work in finding the Nash equilibrium for a Cournot game with two players. 

'''

import numpy as np
import matplotlib.pyplot as plt
from GameTheoryModel import CournotGame

# Define our parameters
alpha = 100
beta = 0.5
beta1, beta2 = 0.5, 0.1  # Cost coefficients
players = 2

cournot = CournotGame(alpha=alpha, beta=beta, cost_params=[beta1, beta2], n=players)

q1 = np.linspace(0, 80, 200)
q2 = np.linspace(0, 80, 200)
br1 = [cournot.best_response([q], beta1) for q in q2]
br2 = [cournot.best_response([q], beta2) for q in q1]

# Find Nash Equilibrium
guess = np.array([10, 10])
tol = 1e-4
error = 1

while error > tol:
    q1_new = cournot.best_response([guess[1]], beta1)
    q2_new = cournot.best_response([guess[0]], beta2)
    new_guess = np.array([q1_new, q2_new])
    error = np.linalg.norm(new_guess - guess)
    guess = new_guess

Q = np.sum(guess)
p = cournot.demand_function(Q)
profits = [p * guess[i] - cournot.cost_function(guess[i], cournot.cost_params[i]) for i in range(players)]

# Plot results
plt.plot(q2, br1, label="Firm 1 Best Response")
plt.plot(br2, q1, label="Firm 2 Best Response")
plt.scatter(guess[1], guess[0], color='red', label="Nash Equilibrium", marker='x')

plt.xlabel("Firm 2 Quantity (q2)")
plt.ylabel("Firm 1 Quantity (q1)")
plt.title("Best Response Functions & Nash Equilibrium")
plt.legend()
plt.grid()
plt.show()

# Print results
print(f"Nash Equilibrium Quantities: Firm 1: {guess[0]:.2f}, Firm 2: {guess[1]:.2f}")
print(f"Market Price: {p:.2f}")
print(f"Profits: Firm 1: {profits[0]:.2f}, Firm 2: {profits[1]:.2f}")
