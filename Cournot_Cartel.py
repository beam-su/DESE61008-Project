'''
Cartel Formation in Cournot Oligopoly

From the Cournot game we know that as the number of player increases, the market price decreases and the quantity produced by each firm decreases. However, firms can collude to form a cartel to increase their profits. 

In this script, we want to see how firms can form a cartel to maximise their profit and how would the total number of players in the market would impact their decision of joining a cartel.

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GameTheoryModel import CournotGame

# Game Setup
alpha = 100  # Demand intercept
beta = 0.7   # Demand slope
cost_per_firm = 0.5  # Constant marginal cost for all firms
max_firms = 20  # Maximum number of firms

num_firms_list = np.arange(2, max_firms + 1)
cartel_profits = []
competitive_profits = []
non_cartel_profits = []
cartel_price = []
competitive_price = []

# Loop through n firms
for n in num_firms_list:
    cost_params = [cost_per_firm] * n  # Assume all firms have the same cost

    cournot_competition = CournotGame(alpha=alpha, beta=beta, cost_params=cost_params, n=n)
    guess = np.ones(n) * 10
    tol = 1e-4
    error = 1

    while error > tol:
        new_guess = np.array([cournot_competition.best_response(np.delete(guess, i), cost_params[i]) for i in range(n)])
        error = np.linalg.norm(new_guess - guess)
        guess = new_guess

    Q_competition = np.sum(guess)
    P_competition = cournot_competition.demand_function(Q_competition)
    profits_competition = [P_competition * guess[i] - cournot_competition.cost_function(guess[i], cost_params[i]) for i in range(n)]

    # Cartel Scenario
    cartel_size = max(1, n // 2)  # Half of the firms attempt to collude
    competitive_firms = n - cartel_size  # Remaining firms act independently

    # Cartel firms act as a single monopolist, but share profits
    Q_cartel = (alpha - cost_per_firm) / (2 * beta)  # Monopoly quantity split among cartel members
    q_cartel_per_firm = Q_cartel / cartel_size
    P_cartel = cournot_competition.demand_function(Q_cartel)
    profits_cartel = [(P_cartel * q_cartel_per_firm) - (cost_per_firm * q_cartel_per_firm)] * cartel_size

    # Non-cartel firms behave like Cournot competitors
    if competitive_firms > 0:
        non_cartel_cournot = CournotGame(alpha=alpha, beta=beta, cost_params=[cost_per_firm] * competitive_firms, n=competitive_firms)
        guess_non_cartel = np.ones(competitive_firms) * 10  # Initial guess for non-cartel firms
        error = 1

        while error > tol:
            new_guess_non_cartel = np.array([
                non_cartel_cournot.best_response(np.delete(guess_non_cartel, i), cost_per_firm) for i in range(competitive_firms)
            ])
            error = np.linalg.norm(new_guess_non_cartel - guess_non_cartel)
            guess_non_cartel = new_guess_non_cartel

        Q_non_cartel = np.sum(guess_non_cartel)
        P_non_cartel = non_cartel_cournot.demand_function(Q_non_cartel)
        profits_non_cartel = [P_non_cartel * guess_non_cartel[i] - non_cartel_cournot.cost_function(guess_non_cartel[i], cost_per_firm) for i in range(competitive_firms)]
    else:
        profits_non_cartel = [0]  # If no non-cartel firms exist

    # Store results
    cartel_profits.append(np.mean(profits_cartel))
    competitive_profits.append(np.mean(profits_competition))
    non_cartel_profits.append(np.mean(profits_non_cartel))
    cartel_price.append(P_cartel)
    competitive_price.append(P_competition)

# Plot Competitive vs Cartel vs Non-Cartel Profits
plt.figure(figsize=(8, 6))
plt.plot(num_firms_list, cartel_profits, label="Cartel Profit per Firm", linestyle="dashed")
plt.plot(num_firms_list, competitive_profits, label="Competitive Profit per Firm")
plt.plot(num_firms_list, non_cartel_profits, label="Non-Cartel Profit per Firm", linestyle="dotted")
plt.xticks(num_firms_list)
plt.xlabel("Number of Firms (n)")
plt.ylabel("Profit per Firm")
plt.title("Firm Profits with Varying n")
plt.legend()
plt.grid()
plt.show()

# Plot Competitive vs Cartel Prices
plt.figure(figsize=(8, 6))
plt.plot(num_firms_list, cartel_price, label="Cartel Price", linestyle="dashed")
plt.plot(num_firms_list, competitive_price, label="Competitive Market Price")
plt.xlabel("Number of Firms (n)")
plt.ylabel("Market Price")
plt.xticks(num_firms_list)
plt.title("Market Price with Varying n")
plt.legend()
plt.grid()
plt.show()

# Export results to DataFrame
df_results = pd.DataFrame({
    "Number of Firms": num_firms_list,
    "Cartel Profit per Firm": cartel_profits,
    "Competitive Profit per Firm": competitive_profits,
    "Non-Cartel Profit per Firm": non_cartel_profits,
    "Cartel Price": cartel_price,
    "Competitive Market Price": competitive_price
})

df_results.to_csv("cartel_vs_competition_results.csv", index=False)