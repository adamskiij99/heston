import numpy as np
from matplotlib import pyplot as plt

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
# BM correlation
rho = -0.6
# risk-free rate
r = 0
# terminal time
T = 30/365
# no. time steps
N = 5000

# Generate samples of the stock's value at time T under the Heston model with
# the specified parameters.
def heston_terminal_values(sample_paths=100, T=T, N=N, V_0=V_0, S_0=S_0,
                           th=th, k=k, xi=xi, rho=rho, r=r):
    dt = T / N    
    # Initialise
    S_t = np.zeros(sample_paths) + S_0
    V_t = np.zeros(sample_paths) + V_0
    
    for j in range(N):
        # Generate correlated BMs (increments)
        dW1s = np.random.normal(0, np.sqrt(dt), sample_paths)
        dW2s = rho*dW1s + np.sqrt(1-rho*rho)*np.random.normal(0, np.sqrt(dt),
                                                              sample_paths)
        
        # Euler-Maruyama for V
        a = k*(th-V_t)
        b = xi*np.sqrt(np.maximum(0, V_t))
        V_t = V_t + a*dt + b*dW1s
        
        # Euler-Maruyama for S
        a = r*S_t
        b = S_t*np.sqrt(np.maximum(0, V_t))
        S_t = S_t + a*dt + b*dW2s
    
    return S_t

# The following two functions generate call and put prices from scratch.
# DO NOT use these if you already have samples of the stock's value at time T;
# instead you should compute the value of a call with the command
#       np.mean(np.maximum(S-K, 0))*np.exp(-r*T)
# or a put with the command
#       np.mean(np.maximum(K-S, 0))*np.exp(-r*T)
# where S is your array of samples of S_T.
def hestonCall(K, sample_paths=100, T=T, N=N, V_0=V_0, S_0=S_0,
               th=th, k=k, xi=xi, rho=rho, r=r):
    terminal_values = heston_terminal_values(sample_paths, T, N)
    return np.mean(np.maximum(terminal_values - K, 0)) * np.exp(-r*T)

def hestonPut(K, sample_paths=100, T=T, N=N, V_0=V_0, S_0=S_0,
              th=th, k=k, xi=xi, rho=rho, r=r):
    terminal_values = heston_terminal_values(sample_paths, T, N)
    return np.mean(np.maximum(K - terminal_values, 0)) * np.exp(-r*T)

if __name__ == "__main__":
    
    print("Running...")
    
    # Run Monte Carlo
    K = np.linspace(30, 70)
    S = heston_terminal_values(20000)
    C = np.zeros(50)
    P = np.zeros(50)
    for i in range(50):
        C[i] = np.mean(np.maximum(S-K[i], 0))*np.exp(-r*T)
        P[i] = np.mean(np.maximum(K[i]-S, 0))*np.exp(-r*T)
    
    # Plotting
    plt.style.use("bmh")
    
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=250)
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
    
    print("Done.")