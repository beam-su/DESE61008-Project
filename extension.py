import numpy as np
from scipy.optimize import minimize
from GameTheoryModel import CournotGame

# Initialize a standard Cournot game
base_game = CournotGame(n_players=3, cost_params=[0.5, 0.8, 0.6])

# Compute the standard Cournot-Nash equilibrium
q_cournot, p_cournot, profits_cournot = base_game.equilibrium()

# Define cartel firms (e.g., firms 0 and 1 cooperate)
cartel_firms = [0, 1]

# Optimize joint profit for cartel firms
initial_guess = np.full(len(cartel_firms), np.mean(base_game.initial_guess))
result = minimize(
    lambda q_cartel: -sum(
        base_game.demand_function(np.sum(np.concatenate(
            (q_cartel, np.delete(base_game.equilibrium(only_quantities=True), cartel_firms))
        ))) * q_cartel[i] - base_game.cost_function(q_cartel[i], base_game.cost_params[cartel_firms[i]])
        for i in range(len(cartel_firms))
    ),
    initial_guess,
    bounds=[base_game.bounds[i] for i in cartel_firms]
)

# Get the optimal cartel output
q_cartel_optimal = result.x

# Compute non-cartel firms' best responses
q_non_cartel = [
    base_game.best_response(
        np.concatenate((q_cartel_optimal, np.delete(base_game.equilibrium(only_quantities=True), cartel_firms))),
        base_game.cost_params[i]
    ) for i in range(base_game.n_players) if i not in cartel_firms
]

# Merge cartel and non-cartel quantities
quantities = np.zeros(base_game.n_players)
quantities[cartel_firms] = q_cartel_optimal
non_cartel_indices = [i for i in range(base_game.n_players) if i not in cartel_firms]
for i, idx in enumerate(non_cartel_indices):
    quantities[idx] = q_non_cartel[i]

# Compute final market price and profits
Q = np.sum(quantities)
p_coop = base_game.demand_function(Q)
profits_coop = [p_coop * quantities[i] - base_game.cost_function(quantities[i], base_game.cost_params[i]) for i in range(base_game.n_players)]

# Print results
print("\nStandard Cournot Equilibrium:")
print(f"Quantities: {q_cournot}, Price: {p_cournot}, Profits: {profits_cournot}")

print("\nCooperative Cournot Equilibrium (Cartel: Firms 0 & 1):")
print(f"Quantities: {quantities}, Price: {p_coop}, Profits: {profits_coop}")
