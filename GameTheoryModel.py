import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GameTheoryModel:
    def __init__(self, alpha=100, beta=0.5, cost_params=None, initial_guess=None, bounds=None, n=2):
        self.alpha = alpha  # Demand intercept
        self.beta = beta    # Demand slope
        self.n = n          # Number of firms
        self.cost_params = cost_params if cost_params else [0.5] * n
        self.initial_guess = initial_guess if initial_guess else [10] * n
        self.bounds = bounds if bounds else [(0, 50)] * n

    def demand_function(self, Q):
        """Computes price given total quantity Q"""
        return max(self.alpha - self.beta * Q, 0)

    def cost_function(self, q, cost_param):
        """Computes cost for a firm"""
        return cost_param * q

class CournotGame(GameTheoryModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def best_response(self, q_others, cost_param):
        """Computes the best response function (n firm)"""
        return max((self.alpha - self.beta * sum(q_others) - cost_param) / (2 * self.beta), 0)  # Adjusted for n players

    def equilibrium(self, tol=1e-4):
        """Iterative best response computation for Cournot equilibrium"""
        quantities = np.array(self.initial_guess)
        error = 1

        while error > tol:
            new_quantities = np.array([
                self.best_response(np.delete(quantities, i), self.cost_params[i])
                for i in range(self.n)
            ])
            error = np.linalg.norm(new_quantities - quantities)
            quantities = new_quantities

        Q = np.sum(quantities)
        p = self.demand_function(Q)
        profits = [p * quantities[i] - self.cost_function(quantities[i], self.cost_params[i]) for i in range(self.n)]

        return quantities, p, profits
