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
#print(2 * k * th - xi * xi)
# In practise however, it isn't always so simple; one (crude) way to avoid
# nonpositive solutions is to replace sqrt(V_t) with sqrt(max{0, V_t}).

# Use Euler-Maruyama method to simulate V
T = 1 / 12
N = 31
dt = T / N
#tt = np.linspace(0, T, N + 1)

# Asset price is then assumed to follow the following SDE
# d(S_t) = r * S_t * dt + sqrt(V_t) * S_t * d(W_t^(2))
# where W^(2) is BM correlated with W^(1) with corr. rho.
S_0 = 50
rho = -0.6
# r : risk-free interest rate (or mean drift rate in terms of P-meas)
r = 0.01

def heston(m): # m = no. samples for Monte Carlo
    # Initialise
    S_t = np.zeros(m) + S_0
    V_t = np.zeros(m) + V_0
    
    for j in range(N):
        # Generate correlated BMs
        dW1s = np.random.normal(0, np.sqrt(dt), m)
        dW2s = rho * dW1s + np.sqrt(1 - rho * rho) * np.random.normal(0, np.sqrt(dt), m)
        
        # Euler-Maruyama for V
        a = k * (th - V_t)
        b = xi * np.sqrt(np.maximum(0, V_t))
        V_t = V_t + a * dt + b * dW1s
        
        # Euler-Maruyama for S
        a = r * S_t
        b = S_t * np.sqrt(np.maximum(0, V_t))
        S_t = S_t + a * dt + b * dW2s
    
    return S_t

# Defaults as in above:
# m, V_0 = 0.3, th = 0.2, k = 1, xi = 0.3, T = 1, N = 5000, rho = -0.6, r = 0.01, S_0 = 50
# 0.3, 0.2, 1, 0.3, 1, 5000, -0.6, 0.01, 50

# Returns a single call option value with strike K
def hestonCallStkK(m, K):
    # See above def for code comments
    S_t = np.zeros(m) + S_0
    V_t = np.zeros(m) + V_0
    
    for j in range(N):
        dW1s = np.random.normal(0, np.sqrt(dt), m)
        dW2s = rho * dW1s + np.sqrt(1 - rho * rho) * np.random.normal(0, np.sqrt(dt), m)
        
        a = k * (th - V_t)
        b = xi * np.sqrt(np.maximum(0, V_t))
        V_t = V_t + a * dt + b * dW1s
        
        a = r * S_t
        b = S_t * np.sqrt(np.maximum(0, V_t))
        S_t = S_t + a * dt + b * dW2s
    
    return np.mean(np.maximum(S_t - K, 0)) * np.exp(-r * T)

# Returns a single call option value with strike K
def hestonPutStkK(m, K):
    # See above def for code comments
    S_t = np.zeros(m) + S_0
    V_t = np.zeros(m) + V_0
    
    for j in range(N):
        dW1s = np.random.normal(0, np.sqrt(dt), m)
        dW2s = rho * dW1s + np.sqrt(1 - rho * rho) * np.random.normal(0, np.sqrt(dt), m)
        
        a = k * (th - V_t)
        b = xi * np.sqrt(np.maximum(0, V_t))
        V_t = V_t + a * dt + b * dW1s
        
        a = r * S_t
        b = S_t * np.sqrt(np.maximum(0, V_t))
        S_t = S_t + a * dt + b * dW2s
    
    return np.mean(np.maximum(S_t - K, 0)) * np.exp(-r * T)

# Same as heston(m) but with parameter control
def hestonP(m, V_0, th, k, xi, T, N, rho, r, S_0):
    S_t = np.zeros(m) + S_0
    V_t = np.zeros(m) + V_0
    
    for j in range(N):
        dW1s = np.random.normal(0, np.sqrt(dt), m)
        dW2s = rho * dW1s + np.sqrt(1 - rho * rho) * np.random.normal(0, np.sqrt(dt), m)
        
        a = k * (th - V_t)
        b = xi * np.sqrt(np.maximum(0, V_t))
        V_t = V_t + a * dt + b * dW1s
        
        a = r * S_t
        b = S_t * np.sqrt(np.maximum(0, V_t))
        S_t = S_t + a * dt + b * dW2s
        
    return S_t

# Run Monte Carlo
K = np.linspace(30, 70)
S = heston(100000)
C = np.zeros(50)
P = np.zeros(50)
for i in range(50):
    C[i] = np.mean(np.maximum(S - K[i], 0)) * np.exp(-r * T)
    P[i] = np.mean(np.maximum(K[i] - S, 0)) * np.exp(-r * T)

# Plotting
plt.style.use("bmh")

fig, ax1 = plt.subplots(figsize = (12, 8))
plt.title("Heston model Monte Carlo options")

ax1.set_xlabel("Strike")
ax1.set_ylabel("Call price")
ax1.plot(K, C, color = "green")
ax1.yaxis.label.set_color("green")
# New y-axis with shared x-axis
ax2 = ax1.twinx()
ax2.set_ylabel("Put price")
ax2.plot(K, P, color = "red")
ax2.yaxis.label.set_color("red")

plt.show()






