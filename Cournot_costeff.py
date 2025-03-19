'''
2-Player Cournot Game with Cost Efficiency

This script is similar to the Cournot_BaseModel.py script, but this script focuses on analysing the impact of cost efficiency on the Cournot game. We will vary the cost parameter of Firm 1 and observe how it affects the market quantity, price, and profits of both firms.

'''

import numpy as np
import matplotlib.pyplot as plt
from GameTheoryModel import CournotGame

# Define base parameters
alpha = 100
beta = 0.5
players = 2


cost1 = np.linspace(0.1, 5, 30)
cost2 = 0.1

q1_list = []
q2_list = []
prices = []
profits1 = []
profits2 = []

for cost_firm1 in cost1:
    cournot = CournotGame(alpha=alpha, beta=beta, cost_params=[cost_firm1, cost2], n=players)

    # Compute Nash Equilibrium
    guess = np.array([10, 10])
    tol = 1e-4
    error = 1

    while error > tol:
        q1_new = cournot.best_response([guess[1]], cost_firm1)
        q2_new = cournot.best_response([guess[0]], cost2)
        new_guess = np.array([q1_new, q2_new])
        error = np.linalg.norm(new_guess - guess)
        guess = new_guess

    # Compute market outcomes
    Q = np.sum(guess)
    p = cournot.demand_function(Q)
    profit1 = p * guess[0] - cournot.cost_function(guess[0], cost_firm1)
    profit2 = p * guess[1] - cournot.cost_function(guess[1], cost2)

    # Store results
    q1_list.append(guess[0])
    q2_list.append(guess[1])
    prices.append(p)
    profits1.append(profit1)
    profits2.append(profit2)

# Plot Firm 1 Quantity vs Cost
plt.figure(figsize=(8, 6))
plt.plot(cost1, q1_list, label="Firm 1 Quantity", color="blue", marker='x')
plt.plot(cost1, q2_list, label="Firm 2 Quantity", color="orange", marker='x')
plt.xlabel("Firm 1 Cost")
plt.ylabel("Equilibrium Quantity")
plt.title("Effect of Firm 1 Cost on Market Quantity")
plt.legend()
plt.grid()
plt.show()

# Plot Market Price vs Cost
plt.plot(cost1, prices, label="Market Price", color="red", linestyle="dashed")
plt.xlabel("Firm 1 Cost")
plt.ylabel("Market Price")
plt.title("Effect of Firm 1 Cost on Market Price")
plt.legend()
plt.grid()
plt.show()

# Plot Firm 1 & Firm 2 Profits vs Cost
plt.plot(cost1, profits1, label="Firm 1 Profit", color="blue", marker='x')
plt.plot(cost1, profits2, label="Firm 2 Profit", color="orange", marker='x')
plt.xlabel("Firm 1 Cost")
plt.ylabel("Profit")
plt.title("Effect of Firm 1 Cost on Firm Profits")
plt.legend()
plt.grid()
plt.show()

# Display results in tabular format
import pandas as pd

df_results = pd.DataFrame({
    "Firm 1 Cost": cost1,
    "Firm 1 Quantity": q1_list,
    "Firm 2 Quantity": q2_list,
    "Market Price": prices,
    "Firm 1 Profit": profits1,
    "Firm 2 Profit": profits2
})

df_results