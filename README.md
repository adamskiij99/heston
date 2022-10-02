# Heston model options pricer
Monte Carlo options pricer under the Heston model.

Volatility $V_t$ and asset price $S_t$ follow:

$$ dV_t = \kappa (\theta - V_t) dt + \xi \sqrt{V_t} dW_t^{(1)}, $$

$$ dS_t = r S_t dt + S_t \sqrt{V_t} dW_t^{(2)}, $$

$$ dW_t^{(1)}dW_t^{(2)} = \rho dt. $$

Default parameters:
- `V_0`$= V_0 = 0.3$: initial volatility.
- `th`$=\theta = 0.2$: long-run expectation of variance.
- `k`$=\kappa = 1$: variance reversion parameter.
- `xi`$=\xi = 0.3$: volatility of volatility.
- `T`$=T = 1/12$: time to maturity in years.
- `N`$=N = 31$: time steps.
- `rho`$=\rho = -0.6$: correlation of BMs. Typically negative.
- `r`$=r = 0.1$: risk-free rate / drift.
- `S_0`$=S_0 = 50$: initial asset price.

Running `heston.py` will generate the CIR process (stochastic volatility process) and the corresponding asset price process and plot them.

![image](https://user-images.githubusercontent.com/62266775/193455318-aa78114f-c59d-4e71-838f-05ea02ea83fd.png)

Running `monte carlo heston model options pricing.py` will generate call and put prices over a specified strike range (default m = 100 000 for Monte Carlo) and plot them.

![image](https://user-images.githubusercontent.com/62266775/193455471-c1afa572-a4e6-4cf8-b064-2f503ba535da.png)

