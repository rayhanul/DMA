import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.stats import norm

# Given parameters
T_values = [100, 10000]
delta_values = np.logspace(-7, -14, 8)
epsilon_values = np.linspace(0.0001, 0.001, 5)
beta = 0.001
epsilon_values = np.log(1 + beta * (np.exp(epsilon_values) - 1))
delta_values = beta * delta_values

# Initialize arrays to store results
Delta = np.zeros((len(T_values), len(epsilon_values), len(delta_values)))
X = np.zeros((len(T_values), len(epsilon_values), len(delta_values)))
Y = np.zeros((len(T_values), len(epsilon_values), len(delta_values)))
U_R2 = np.zeros((len(T_values), len(epsilon_values), len(delta_values)))
sigma = np.zeros((len(T_values), len(epsilon_values), len(delta_values)))
U_G = np.zeros((len(T_values), len(epsilon_values), len(delta_values)))
c = 1

# Loop through each combination of T, epsilon, and delta
for i in range(len(T_values)):
    print(i)
    T = T_values[i]

    print(f"epsilon {len(epsilon_values)}")
    for j in range(len(epsilon_values)):
        epsilon = epsilon_values[j]
        print(f" len: {len(delta_values)}")
        for k in range(len(delta_values)):
            delta = delta_values[k]

            sigma[i, j, k] = (2 * np.sqrt(2 * T * np.log(1 / delta)) + np.sqrt(8 * T * np.log(1 / delta) + 8 * T * epsilon)) / (4 * epsilon)
            print(f"sigma : {sigma[i, j, k]}")
            U_G[i, j, k] = norm.cdf(c / sigma[i, j, k]) - norm.cdf(-c / sigma[i, j, k])
            Delta[i, j, k] = epsilon - np.log(epsilon)

            x = np.linspace(0.0001, 1, 200)
            g = 0
            for t in range(100):
                if delta ** x[t] < np.exp(-1):
                    KK = -lambertw(-delta ** x[t]).real / (T * Delta[i, j, k] * x[t])
                    R2 = 1 - 1 / (c * x[t] + 1) ** KK
                    if R2 > g:
                        g = R2
                        X[i, j, k] = x[t]
                        Y[i, j, k] = KK

            U_R2[i, j, k] = 1 - 1 / (c * X[i, j, k] + 1) ** Y[i, j, k]

# Plotting
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for k in range(len(delta_values)):
    ax = axs[k // 4, k % 4]
    for i in range(len(T_values)):
        ax.plot(epsilon_values, U_R2[i, :, k], '--', label=f'R2, T = {T_values[i]}')
        ax.plot(epsilon_values, U_G[i, :, k], '-', label=f'G, T = {T_values[i]}')
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.set_xlabel('epsilon')
    ax.set_ylabel('P(|noise|<1)')
    ax.set_title(f'Delta = {delta_values[k]}')
    ax.legend(loc='best')
plt.tight_layout()
plt.show()
