import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GameTheoryModel:
    def __init__(self, alpha_d=100, beta_d=0.5, cost_params=None, initial_guess=None, bounds=None, n_players=2):
        self.alpha_d = alpha_d                                      # Demand Intercept
        self.beta_d = beta_d                                        # Demand Slope
        self.n_players = n_players                                  # Number of players
        self.cost_params = cost_params or [0.5] * n_players         # Check cost parameter input
        self.initial_guess = initial_guess or [10] * n_players      # Check initial guess input
        self.bounds = bounds or [(0, 50)] * n_players               # Check boundary input
    
    def demand_function(self, Q):
        return max((self.alpha_d - Q) / self.beta_d, 0)
    
    def cost_function(self, q, beta):
        return beta * q

class CournotGame(GameTheoryModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def best_response(self, q_others, beta):
        return max((self.alpha_d - sum(q_others) - self.beta_d * beta) / (self.n_players + 1), 0)
    
    def equilibrium(self):
        quantities = np.zeros(self.n_players)
        for i in range(self.n_players):
            q_others = np.delete(quantities, i)
            quantities[i] = self.best_response(q_others, self.cost_params[i])
        
        Q = np.sum(quantities)
        p = self.demand_function(Q)
        profits = [float(p * quantities[i] - self.cost_function(quantities[i], self.cost_params[i])) for i in range(self.n_players)]
        
        return quantities, float(p), profits

class StackelbergGame(GameTheoryModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def follower_best_response(self, q_leaders, beta_follower):
        return max((self.alpha_d - sum(q_leaders) - self.beta_d * beta_follower) / (self.n_players - len(q_leaders) + 1), 0)
    
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
        profits = [float(p * quantities[i] - self.cost_function(quantities[i], self.cost_params[i])) for i in range(self.n_players)]
        
        return quantities, float(p), profits