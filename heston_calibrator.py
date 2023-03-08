# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#from heston import heston_paths
from heston_options import hestonAnalyticalCall, hestonAnalyticalPut

#DATA, _ = heston_paths()

# params_0 : tuple of the form (V_0, th, k, xi, rho)
# market_(calls/puts) : tuples of the form (strike, price, time_to_maturity)
def calibrate(params_0, market_calls, market_puts):
    def loss(params, market_calls, market_puts):
        tot = 0
        V_0, th, k, xi, rho = params
        for (strike, price, maturity) in market_calls:
            analytical_price = hestonAnalyticalCall(strike, V_0=V_0, th=th,
                                                    k=k, xi=xi, rho=rho,
                                                    T=maturity)
            tot += (price - analytical_price)**2
        
        for (strike, price, maturity) in market_puts:
            analytical_price = hestonAnalyticalPut(strike, V_0=V_0, th=th,
                                                    k=k, xi=xi, rho=rho,
                                                    T=maturity)
            tot += (price - analytical_price)**2
        print(tot)
        
        return tot
    return minimize(loss, params_0, args=(market_calls, market_puts),
                    method="Nelder-Mead")

if __name__ == "__main__":
    print("Running...")
    params_0 = (0.5, 0.5, 0.5, 0.5, -0.5)
    market_calls = []
    market_puts = []
    Ks = [37, 52, 69, 31, 42, 56, 50, 30, 38]
    Ts = [2, 3, 0.1, 1, 1.5, 2.5, 0.5, 0.7, 1.7]
    for i in range(len(Ks)):
        market_call = (Ks[i], hestonAnalyticalCall(K=Ks[i], T=Ts[i]), Ts[i])
        market_calls += [market_call]
        market_put = (Ks[i], hestonAnalyticalPut(K=Ks[i], T=Ts[i]), Ts[i])
        market_puts += [market_put]
    
    print(calibrate(params_0, market_calls, market_puts))
    print("Done.")