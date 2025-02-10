import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GameTheoryModel:
    def __init__(self, alpha_d=100, beta_d=0.5, beta1=0.5, beta2=0.1, initial_guess=[10,10], bounds=[(0,50),(0,50)]):
        self.alpha_d = alpha_d  # Demand Intercept
        self.beta_d = beta_d    # Demand Slope
        self.beta1 = beta1      # Cost coefficient for firm A
        self.beta2 = beta2      # Cost coefficient for firm B
        self.initial_guess = initial_guess
        self.bounds = bounds

    def demand_function(self, Q):
        return (self.alpha_d - Q) / self.beta_d
    
    def cost_function(self, q, beta):
        return beta * q
    
class CournotGame(GameTheoryModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def best_response_q1(self, q2):
        return (self.alpha_d - q2 - self.beta_d * self.beta1) / 2

    def best_response_q2(self, q1):
        return (self.alpha_d - q1 - self.beta_d * self.beta2) / 2

    def equilibrium(self):
        # Solve for Nash equilibrium using best response functions
        q1 = (self.alpha_d - self.beta_d * (self.beta1 + 2 * self.beta2)) / 3
        q2 = (self.alpha_d - self.beta_d * (self.beta2 + 2 * self.beta1)) / 3
        Q = q1 + q2
        p = self.demand_function(Q)
        profit1 = p * q1 - self.cost_function(q1, self.beta1)
        profit2 = p * q2 - self.cost_function(q2, self.beta2)
        
        return q1, q2, p, profit1, profit2
    
class StackelbergGame(GameTheoryModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def follower_best_response(self, q1):
        return max((self.alpha_d - q1 - self.beta_d * self.beta2) / 2, 0)
    
    def leader_objective(self, q1):
        q2 = self.follower_best_response(q1)
        Q = q1 + q2
        p = self.demand_function(Q)
        profit1 = p * q1 - self.cost_function(q1, self.beta1)
        return -profit1
    
    def equilibrium(self):
        result = minimize(self.leader_objective, [self.initial_guess[0]], bounds=[self.bounds[0]])
        q1 = result.x[0]
        q2 = self.follower_best_response(q1)
        Q = q1 + q2
        p = self.demand_function(Q)
        profit1 = p * q1 - self.cost_function(q1, self.beta1)
        profit2 = p * q2 - self.cost_function(q2, self.beta2)
        
        return q1, q2, p, profit1, profit2