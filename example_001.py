"""
A simulation of the statistical properties for the motion of
a lysozyme molecule in water is presented using `yupi` API.
The simulation shows cualitatively the classical scaling laws of
the Langevin theory to explain Brownian Motion (those for Mean
Square Displacement or Velocity Autocorrelation Function).

The example is structured as follows:
- Definition of parameters
- Dimenssionless equation
- Data analysis and plotting
- References
"""

import numpy as np
import matplotlib.pyplot as plt
from yupi.generators import LangevinGenerator
from yupi.stats import (
    msd,
    speed_ensemble,
    collect_from_ensemble,
    collect_from_lag,
    vacf,
    turning_angles_ensemble,
    kurtosis,
    kurtosis_reference
)
from yupi.graphics import (
    plot_2D,
    plot_angles_hist,
    plot_kurtosis,
    plot_msd,
    plot_vacf,
    plot_velocity_hist
)

np.random.seed(0)

## 1. Simulation and model parameters

# simulation parameters
tt_adim = 30     # dimensionless total time
dim = 2          # trajectory dimension
N = 1000         # number of trajectories
dt_adim = 1e-1   # dimensionaless time step

# deterministic model parameters
N0 = 6.02e23     # Avogadro's constant [1/mol]
k = 1.38e-23     # Boltzmann's constant [J/mol.K]
T = 300          # absolute temperature [K]
eta = 1.002e-3   # water viscosity [Pa.s]
M = 14.1         # lysozyme molar mass [kg/mol] [1]
d1 = 90e-10      # semi-major axis [m] [2]
d2 = 18e-10      # semi-minor axis [m] [2]

m = M / N0                   # mass of one molecule
a = np.sqrt(d1/2 * d2/2)     # radius of the molecule
alpha = 6 * np.pi * eta * a  # Stoke's coefficient
tau = (alpha / m)**-1        # relaxation time
v_eq = np.sqrt(k * T / m)    # equilibrium thermal velocity

# intrinsic reference quantities
vr = v_eq       # intrinsic reference velocity
tr = tau        # intrinsic reference time
lr = vr * tr    # intrinsic reference length

# statistical model parameters
dt = dt_adim * tr                        # real time step
noise_pdf = 'normal'                     # noise pdf
noise_scale_adim = np.sqrt(2 * dt_adim)  # scale parameter of noise pdf
v0_adim = np.random.randn(dim, N)        # initial dimensionaless speeds


## 2. Simulating the process

lg = LangevinGenerator(tt_adim, dim, N, dt_adim, v0=v0_adim)
# lg.set_scale(v_scale=vr, r_scale=lr, t_scale=tr)
trajs = lg.generate()


## 3. Data analysis and plots

plt.figure(figsize=(9,5))

# Spacial trajectories
ax1 = plt.subplot(231)
plot_2D(trajs[:5], legend=False, show=False)

#  velocity histogram
# v = speed_ensemble(trajs, step=1)
v = collect_from_lag(trajs, key='vn', tau=1, time_in_samples=True)
ax2 = plt.subplot(232)
plot_velocity_hist(v, bins=20, show=False)

#  turning angles
theta = turning_angles_ensemble(trajs)
ax3 = plt.subplot(233, projection='polar')
plot_angles_hist(theta, bins=50, show=False)

#  mean square displacement
lag_msd = 30
msd, msd_std = msd(trajs, time_avg=True, lag=lag_msd)
ax4 = plt.subplot(234)
plot_msd(msd, msd_std, dt, lag=lag_msd, show=False)

#  kurtosis
kurtosis, _ = kurtosis(trajs, time_avg=False, lag=30)
kurt_ref = kurtosis_reference(trajs)
ax5 = plt.subplot(235)
plot_kurtosis(kurtosis, kurtosis_ref=kurt_ref, dt=dt, show=False)

#  velocity autocorrelation function
lag_vacf = 50
vacf, _ = vacf(trajs, time_avg=True, lag=lag_vacf)
ax6 = plt.subplot(236)
plot_vacf(vacf, dt, lag_vacf, show=False)

# Generate plot
plt.tight_layout()
plt.show()

## References
# [1] Berg, Howard C. Random walks in biology. Princeton University Press, 1993.
# [2] Colvin, J. Ross. "The size and shape of lysozyme." Canadian Journal of Chemistry 30.11 (1952): 831-834.
