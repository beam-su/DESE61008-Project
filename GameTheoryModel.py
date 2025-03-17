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
        return max((self.alpha - Q) / self.beta, 0)

    def cost_function(self, q, beta):
        """Computes cost for a firm"""
        return beta * q

class CournotGame(GameTheoryModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def best_response(self, q_others, beta):
        """Computes the best response function for a firm in n-player Cournot competition"""
        return max((self.alpha - sum(q_others) - self.beta * beta) / self.n, 0)  # Generalized for n players

    def equilibrium(self, tol=1e-4):
        """Finds Nash equilibrium using an iterative best response method for any number of firms"""
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

class StackelbergGame(GameTheoryModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def follower_best_response(self, q_leaders, beta_follower):
        return max((self.alpha - sum(q_leaders) - self.beta * beta_follower) / (self.n - len(q_leaders) + 1), 0)
    
    def leader_objective(self, q_leaders):
        q_followers = [self.follower_best_response(q_leaders, beta) for beta in self.cost_params[len(q_leaders):]]
        Q = sum(q_leaders) + sum(q_followers)
        p = self.demand_function(Q)
        profit_leaders = [float(p * q_leaders[i] - self.cost_function(q_leaders[i], self.cost_params[i])) for i in range(len(q_leaders))]
        return -sum(profit_leaders)
    
    def equilibrium(self):
        result = minimize(self.leader_objective, self.initial_guess[:1], bounds=self.bounds[:1])
        q_leaders = result.x
        q_followers = [self.follower_best_response(q_leaders, beta) for beta in self.cost_params[len(q_leaders):]]
        quantities = np.concatenate((q_leaders, q_followers))
        Q = np.sum(quantities)
        p = self.demand_function(Q)
        profits = [float(p * quantities[i] - self.cost_function(quantities[i], self.cost_params[i])) for i in range(self.n)]
        
        return quantities, float(p), profits