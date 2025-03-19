'''
Cartel Formation in Cournot Oligopoly

From the Cournot game we know that as the number of player increases, the market price decreases and the quantity produced by each firm decreases. However, firms can collude to form a cartel to increase their profits. 

In this script, we want to see how firms can form a cartel to maximise their profit and how would the total number of players in the market would impact the profit of the cartel.

'''

import numpy as np
import matplotlib.pyplot as plt

alpha = 100
beta = 0.5
c = 10
n = range(3, 21)
cartel_size = 2

P_ct = [] # Total cartel profit
P_cpf = [] # Cartel profit per firm
P_nc = [] # Non-cartel profit
P_ch = [] # Cheater profit

for i in n:
    nc_size = i - cartel_size
    
    # Cartel as a joint monopolist: Maximizing total profit
    Q_ct = (alpha - c) / (2 * beta) # Total quantity for cartel
    Q_cartel_individual = Q_ct / cartel_size
    
    # Cournot competition for non-cartel firms
    Q_nc = (alpha - c - beta * Q_ct) / (beta * (1 + nc_size))
    Q_t = cartel_size * Q_cartel_individual + nc_size * Q_nc

    P_market = alpha - beta * Q_t
    
    profit_cartel_total = cartel_size * ((P_market - c) * Q_cartel_individual)
    profit_cartel_per_firm = (P_market - c) * Q_cartel_individual
    profit_non_cartel = (P_market - c) * Q_nc
    
    # Cheating scenario (1 defector)
    Q_ch = (alpha - c - beta * ((cartel_size - 1) * Q_cartel_individual + nc_size * Q_nc)) / (2 * beta)
    Q_tch = (cartel_size - 1) * Q_cartel_individual + Q_ch + nc_size * Q_nc
    P_cheat = alpha - beta * Q_tch
    profit_cheater = (P_cheat - c) * Q_ch
    
    P_ct.append(profit_cartel_total)
    P_cpf.append(profit_cartel_per_firm)
    P_nc.append(profit_non_cartel)
    P_ch.append(profit_cheater)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(n, P_cpf, label='Cartel Profit (Per Firm)', marker='x')
plt.plot(n, P_nc, label='Non-Cartel Firm Profit', marker='x')
plt.plot(n, P_ch, label='Cheater Profit', linestyle='dashed', marker='x')
plt.xlabel('Total Number of Firms')
plt.ylabel('Profit')
plt.title('Profits and Cartels')
plt.xticks(range(4, 21, 1))
plt.xlim(4, 20)
plt.legend()
plt.grid()
plt.show()