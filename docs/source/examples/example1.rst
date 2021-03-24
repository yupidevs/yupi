Example 1
=========

A simulation of the statistical properties for the motion of 
a lysozyme molecule in water is presented using `yupi` API. 
The simulation shows cualitatively the classical scaling laws of 
the Langevin theory to explain Brownian Motion (those for Mean 
Square Displacement or Velocity Autocorrelation Function). 

The example is structured as follows:
1. Setup dependencies
2. Definition of parameters
3. Generating trajectories
4. Data analysis and plotting
5. References


1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from yupi.generating import LangevinGenerator
   import yupi.analyzing as ypa

Fix the random generator seed to make results reproducible:

.. code-block:: python

   np.random.seed(0)


2. Definition of parameters
---------------------------

Simulation parameters

.. code-block:: python

   tt_adim = 30     # dimensionless total time
   dim = 2          # trajectory dimension
   N = 1000         # number of trajectories
   dt_adim = 1e-1   # dimensionaless time step

Deterministic model parameters

.. code-block:: python

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

Intrinsic reference quantities

.. code-block:: python

   vr = v_eq       # intrinsic reference velocity
   tr = tau        # intrinsic reference time
   lr = vr * tr    # intrinsic reference length

Statistical model parameters

.. code-block:: python

   dt = dt_adim * tr                        # real time step
   noise_pdf = 'normal'                     # noise pdf
   noise_scale_adim = np.sqrt(2 * dt_adim)  # scale parameter of noise pdf
   v0_adim = np.random.randn(dim, N)        # initial dimensionaless speeds


3. Generating trajectories
--------------------------

.. code-block:: python

   lg = LangevinGenerator(tt_adim, dim, N, dt_adim, v0=v0_adim)
   lg.set_scale(v_scale=vr, r_scale=lr, t_scale=tr)
   trajs = lg.generate()


4. Data analysis and plots
--------------------------

Initialize empty figure for plot all the results:

.. code-block:: python

   plt.figure(figsize=(9,5))

Plot spacial trajectories

.. code-block:: python

   ax1 = plt.subplot(231)
   ypa.plot_trajectories(trajs, max_trajectories=5, legend=False, plot=False)

Plot velocity histogram 

.. code-block:: python

   v = ypa.estimate_velocity_samples(trajs, step=1)
   ax2 = plt.subplot(232)
   ypa.plot_velocity_hist(v, bins=20, plot=False)

Plot turning angles 

.. code-block:: python

   theta = ypa.estimate_turning_angles(trajs)
   ax3 = plt.subplot(233, projection='polar')
   ypa.plot_angle_distribution(theta, plot=False)

Plot Mean Square Displacement 

.. code-block:: python

   lag_msd = 30
   msd, msd_std = ypa.estimate_msd(trajs, time_avg=True, lag=lag_msd)
   ax4 = plt.subplot(234)
   ypa.plot_msd(msd, msd_std, dt, lag=lag_msd, plot=False)

Plot Kurtosis

.. code-block:: python

   kurtosis = ypa.estimate_kurtosis(trajs, time_avg=False, lag=30)
   ax5 = plt.subplot(235)
   ypa.plot_kurtosis(kurtosis, dt=dt, plot=False)

Plot Velocity autocorrelation function 

.. code-block:: python

   lag_vacf = 50
   vacf, _ = ypa.estimate_vacf(trajs, time_avg=True, lag=lag_vacf)
   ax6 = plt.subplot(236)
   ypa.plot_vacf(vacf, dt, lag_vacf, plot=False)

Generate plot

.. code-block:: python

   plt.tight_layout()
   plt.show()

.. figure:: /images/example1.png
   :alt: Output of example1
   :align: center


5. References
-------------
| [1] Berg, Howard C. Random walks in biology. Princeton University Press, 1993.
| [2] Colvin, J. Ross. "The size and shape of lysozyme." Canadian Journal of Chemistry 30.11 (1952): 831-834.
