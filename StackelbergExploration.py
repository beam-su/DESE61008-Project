import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def StackelbergDuopoly(alpha_d, beta_d, c_1, c_2):
    """
    Computes the Stackelberg equilibrium quantities and profits for a leader-follower duopoly, derived in the report.
    Also plots profit curves for both firms as a function of the leader's quantity.
    
    Parameters:
    alpha_d (float): Demand intercept
    beta_d (float): Demand slope
    c_1 (float): Marginal cost of the leader firm
    c_2 (float): Marginal cost of the follower firm
    
    Returns:
    q_1_star (float): Equilibrium quantity of the leader firm
    q_2_star (float): Equilibrium quantity of the follower firm
    profit_1 (float): Profit of the leader firm
    profit_2 (float): Profit of the follower firm
    """
    # Compute equilibrium quantities
    q_1_star = (alpha_d + c_2 - 2 * c_1) / (2 * beta_d)
    q_2_star = (alpha_d - 3 * c_2 + 2 * c_1) / (4 * beta_d)
    
    # Ensure non-negative production
    q_1_star = max(q_1_star, 0)
    q_2_star = max(q_2_star, 0)
    
    # Generate varying q1 values
    q1_vals = np.linspace(0, q_1_star * 2, 100)
    q2_vals = (alpha_d - beta_d * q1_vals - c_2) / (2 * beta_d)  # Follower's best response
    
    # Compute profits
    profit_1_vals = (alpha_d - beta_d * (q1_vals + q2_vals)) * q1_vals - c_1 * q1_vals
    profit_2_vals = (alpha_d - beta_d * (q1_vals + q2_vals)) * q2_vals - c_2 * q2_vals
    
    # Plot profit functions
    # plt.figure(figsize=(10, 6))
    # plt.plot(q1_vals, profit_1_vals, label="Leader's Profit", color='blue')
    # plt.plot(q1_vals, profit_2_vals, label="Follower's Profit", color='orange')
    # plt.axvline(q_1_star, color='red', linestyle='--', label="Equilibrium q1")
    # plt.scatter(q_1_star, (alpha_d - beta_d * (q_1_star + q_2_star)) * q_1_star - c_1 * q_1_star, color='red', marker='x')
    # plt.xlabel("Leader's Quantity (q1)")
    # plt.ylabel("Profit")
    # plt.title("Profit Functions of Leader and Follower")
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    # Return quantities and profits.
    return q_1_star, q_2_star, (alpha_d - beta_d * (q_1_star + q_2_star)) * q_1_star - c_1 * q_1_star, (alpha_d - beta_d * (q_1_star + q_2_star)) * q_2_star - c_2 * q_2_star

def followerProfitGraph(alpha_d, beta_d, c_1, c_2, q_1_fixed):
    # Compute equilibrium q2
    q_2_star = (alpha_d - 3 * c_2 + 2 * c_1) / (4 * beta_d)
    
    # Generate varying q2 values
    q2_vals = np.linspace(0, q_2_star * 2, 100)
    
    # Compute profits for follower
    profit_2_vals = (alpha_d - beta_d * (q_1_fixed + q2_vals)) * q2_vals - c_2 * q2_vals
    
    # Plot follower profit function
    plt.figure(figsize=(10, 6))
    plt.plot(q2_vals, profit_2_vals, label="Follower's Profit", color='orange')
    plt.axvline(q_2_star, color='red', linestyle='--', label="Equilibrium q2")
    plt.scatter(q_2_star, (alpha_d - beta_d * (q_1_fixed + q_2_star)) * q_2_star - c_2 * q_2_star, color='red', marker='x')
    plt.xlabel("Follower's Quantity (q2)")
    plt.ylabel("Profit")
    plt.title("Follower's Profit vs. Varying q2")
    plt.legend()
    plt.grid()
    plt.show()

def plot_profit_vs_c1_ratio(alpha_d, beta_d, c_2, c1_range):
    """
    Plots the profitability of both firms as a function of the c1:c2 ratio.
    Also identifies and marks the intersection point where profits are equal.
    
    Parameters:
    alpha_d (float): Demand intercept
    beta_d (float): Demand slope
    c_2 (float): Fixed marginal cost of the follower firm
    c1_range (array): Range of c1 values to iterate over
    """
    c1_c2_ratios = c1_range / c_2  # Compute the ratio c1:c2
    profit_1_vals = []
    profit_2_vals = []
    
    for c_1 in c1_range:
        _, _, profit_1, profit_2 = StackelbergDuopoly(alpha_d, beta_d, c_1, c_2)
        profit_1_vals.append(profit_1)
        profit_2_vals.append(profit_2)
    
    # Find intersection where profits are equal
    profit_diffs = np.abs(np.array(profit_1_vals) - np.array(profit_2_vals))
    intersection_idx = np.argmin(profit_diffs)
    intersection_ratio = c1_c2_ratios[intersection_idx]
    intersection_profit = profit_1_vals[intersection_idx]  # Same for both firms at this point
    
    # Print intersection coordinates
    print(f"Intersection Point: c1:c2 Ratio = {intersection_ratio:.2f}, Profit = {intersection_profit:.2f}")
    
    # Plot profit functions
    plt.figure(figsize=(10, 6))
    plt.plot(c1_c2_ratios, profit_1_vals, label="Leader's Profit", color='blue')
    plt.plot(c1_c2_ratios, profit_2_vals, label="Follower's Profit", color='orange')
    plt.scatter(intersection_ratio, intersection_profit, color='red', marker='x', label="Profit Intersection")
    plt.xlabel("c1:c2 Ratio")
    plt.ylabel("Profit")
    plt.title("Profit vs. c1:c2 Ratio in Stackelberg Competition")
    plt.legend()
    plt.grid()
    plt.show()

# Call function with a range of c1 values
alpha_d = 100
beta_d = 0.5
c_1 = 0.5
c_2 = 0.1
c1_range = np.linspace(0.01, 10, 1000)  # Adequate resolution between 0.01 and 3

plot_profit_vs_c1_ratio(alpha_d, beta_d, c_2, c1_range)

basecase = StackelbergDuopoly(alpha_d, beta_d, c_1, c_2)
graph = followerProfit(alpha_d,beta_d,c_1,c_2,basecase[0])
print(basecase)