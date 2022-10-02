# Imports
import numpy as np
from matplotlib import pyplot as plt

# Variance process assumed to follow CIR,
# d(V_t) = k * (th - V_t) * dt + xi * sqrt(V_t) * d(W_t^(1))
# where W^(1) is SBM.
# V_0 : initial variance
V_0 = 0.3
# th : theta, long-run expectation of variance
th = 0.2
# k : kappa, "variance reversion" parameter
k = 1
# xi : vol of vol
xi = 0.3

# Feller condition: 2 * k * th > xi^2. In theory this is enough to prevent the 
print(2 * k * th - xi * xi)
# In practise however, it isn't always so simple, one (crude) way to avoid
# nonpositive solutions is to replace sqrt(V_t) with sqrt(max{0, V_t}).

# Use Euler-Maruyama method to simulate V
T = 1
N = 5000
dt = T / N
tt = np.linspace(0, T, N + 1)
dW1 = np.random.normal(0, np.sqrt(dt), N)
W1 = np.cumsum(dW1)

V = np.zeros(N + 1)
V[0] = V_0

for i in range(1, N + 1):
    a = k * (th - V[i - 1])
    b = xi * np.sqrt(np.maximum(0, V[i - 1]))
    V[i] = V[i - 1] + a * dt + b * dW1[i - 1]

# Asset price is then assumed to follow the following SDE
# d(S_t) = r * S_t * dt + sqrt(V_t) * S_t * d(W_t^(2))
# where W^(2) is BM correlated with W^(1) with corr. rho.
rho = -0.6
# r : risk-free interest rate (or mean drift rate in terms of P-meas)
r = 0

# Generate correlated BM
dW2 = rho * dW1 + np.sqrt(1 - rho * rho) * np.random.normal(0, np.sqrt(dt), N)
W2 = np.cumsum(dW2)

# Euler-Maruyama for S
S = np.zeros(N + 1)
S[0] = 50

for i in range(1, N + 1):
    a = r * S[i - 1]
    b = S[i - 1] * np.sqrt(np.maximum(0, V[i - 1]))
    S[i] = S[i - 1] + a * dt + b * dW2[i - 1]

# plotting
plt.style.use("bmh")

fig, ax1 = plt.subplots(figsize = (12, 8))
plt.title("Heston Model")

#ax1.plot(W1)
#ax1.plot(W2)

ax1.set_xlabel("Time")
ax1.set_ylabel("Asset Price")
ax1.plot(tt, S, color = "green")
# New y-axis with shared x-axis
ax2 = ax1.twinx()
ax2.set_ylabel("Volatility")
ax2.plot(tt, V, color = "red")

ax1.yaxis.label.set_color("green")
ax2.yaxis.label.set_color("red")

plt.show()