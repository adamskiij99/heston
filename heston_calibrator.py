# Imports
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from heston_options import hestonAnalyticalCall, hestonAnalyticalPut
from heston import heston_paths

# params_0 : tuple of the form (V_0, th, k, xi, rho)
# market_(calls/puts) : tuples of the form (strike, price, time_to_maturity)
def calibrate(params_0, market_calls, market_puts, S_0):
    def loss(params, market_calls, market_puts, S_0):
        tot = 0
        V_0, th, k, xi, rho = params
        for (strike, price, maturity) in market_calls:
            analytical_price = hestonAnalyticalCall(strike, V_0=V_0, th=th,
                                                    k=k, xi=xi, rho=rho,
                                                    T=maturity, S_0=S_0)
            tot += (price - analytical_price)**2
        
        for (strike, price, maturity) in market_puts:
            analytical_price = hestonAnalyticalPut(strike, V_0=V_0, th=th,
                                                    k=k, xi=xi, rho=rho,
                                                    T=maturity, S_0=S_0)
            tot += (price - analytical_price)**2
        return tot
    res = minimize(loss, params_0, args=(market_calls, market_puts, S_0),
                    method="Nelder-Mead")
    return res

if __name__ == "__main__":
    print("Running...")
    (V_0, th, k, xi, rho) = (0.5, 0.5, 0.5, 0.5, -0.5)
    market_calls = []
    market_puts = []
    Ks = [37, 52, 69, 31, 42, 56, 50, 30, 38]
    Ts = [2, 3, 0.1, 1, 1.5, 2.5, 0.5, 0.7, 1.7]
    for i in range(len(Ks)):
        market_call = (Ks[i], hestonAnalyticalCall(K=Ks[i], T=Ts[i]), Ts[i])
        market_calls += [market_call]
        market_put = (Ks[i], hestonAnalyticalPut(K=Ks[i], T=Ts[i]), Ts[i])
        market_puts += [market_put]
    
    res = calibrate((V_0, th, k, xi, rho), market_calls, market_puts, 50)
    if res.success == True:
        calibrated_params = res.x
        print("\n\nCalibration successful. Calibrated parameters are:\n"+
              "[V_0, th, k, xi, rho] =", calibrated_params)
    else:
        sys.exit("Calibration unsuccessful. Exiting program...")
    
    print("Plotting simulated market under true params and calibrated params",
          "using the same source of randomness.")
    plt.figure(dpi=250)
    seed = 0
    np.random.seed(seed)
    true_path, _ = heston_paths(1)
    (V_0, th, k, xi, rho) = calibrated_params
    np.random.seed(seed)
    calibrated_path, _ = heston_paths(1, V_0=V_0, th=th, k=k, xi=xi, rho=rho)
    tt = np.linspace(0, 1, 5001)
    plt.plot(tt, true_path, label="True path")
    plt.plot(tt, calibrated_path, label="Calibrated path")
    plt.title("Comparison of true dynamics and calibrated dynamics")
    plt.legend()
    plt.show()