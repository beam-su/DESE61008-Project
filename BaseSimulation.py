# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from GameTheoryModel import CournotGame, StackelbergGame

# Run simulation for both models
cournot = CournotGame()
stackelberg = StackelbergGame()

q1_cournot, q2_cournot, p_cournot, profit1_cournot, profit2_cournot = cournot.equilibrium()
q1_stackelberg, q2_stackelberg, p_stackelberg, profit1_stackelberg, profit2_stackelberg = stackelberg.equilibrium()

print("\nCournot Equilibrium:")
print(f"Firm A Quantity: {q1_cournot}, Firm B Quantity: {q2_cournot}")
print(f"Market Price: {p_cournot}")
print(f"Firm A Profit: {profit1_cournot}, Firm B Profit: {profit2_cournot}")

print("\nStackelberg Equilibrium:")
print(f"Firm A Quantity (Leader): {q1_stackelberg}, Firm B Quantity (Follower): {q2_stackelberg}")
print(f"Market Price: {p_stackelberg}")
print(f"Firm A Profit: {profit1_stackelberg}, Firm B Profit: {profit2_stackelberg}")

#Plots here?