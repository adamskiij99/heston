# Imports
import numpy as np
from matplotlib import pyplot as plt

# Variance process assumed to follow CIR,
# d(V_t) = k * (th - V_t) * dt + xi * sqrt(V_t) * d(W_t^(1))
# where W^(1) is SBM.
# initial variance
V_0 = 0.3
# initial stock value
S_0 = 50
# theta, long-run expectation of variance
th = 0.2
# kappa, "variance reversion" parameter
k = 1
# vol of vol
xi = 0.3

# Asset price is then assumed to follow the following SDE
# d(S_t) = r * S_t * dt + sqrt(V_t) * S_t * d(W_t^(2))
# where W^(2) is BM correlated with W^(1) with corr. rho.
rho = -0.6
# risk-free rate
r = 0

# Use Euler-Maruyama method to simulate V with terminal time T and N steps
T = 30/365
N = 5000

def heston_paths(T=T, N=N, V_0=V_0, S_0=S_0, th=th, k=k, xi=xi, rho=rho, r=r,
           method="Maruyama", terminal_S=False):
    
    # Generate correlated BMs
    dt = T / N
    dW1 = np.random.normal(0, np.sqrt(dt), N)
    dW2 = rho*dW1 + np.sqrt(1-rho*rho)*np.random.normal(0, np.sqrt(dt), N)
    
    # 
    V = np.zeros(N+1)
    V[0] = V_0    
    for i in range(1, N+1):
        a = k*(th-V[i-1])
        b = xi*np.sqrt(np.maximum(0, V[i-1]))
        V[i] = V[i-1] + a*dt + b*dW1[i-1]
    
    # Initialise and generate S using EM
    S = np.zeros(N+1)
    S[0] = S_0
    for i in range(1, N+1):
        a = r*S[i-1]
        b = S[i-1]*np.sqrt(np.maximum(0, V[i-1]))
        S[i] = S[i-1] + a*dt + b*dW2[i-1]
    
    if terminal_S:
        return S[-1]
    else:
        return S, V

if __name__ == "__main__":
    
    print("Running...")
    
    S, V = heston_paths()
    
    # plotting
    plt.style.use("bmh")
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=250)
    plt.title("Heston Model")
    
    tt = np.linspace(0, T, N+1)
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
    
    print("Done.")