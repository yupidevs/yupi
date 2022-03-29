.. _Example 1:

Example 1
=========

We use a Langevin Generator tuned to generate the trajectories
of a lysozyme molecule in water. After generating a significant
amount of trajectories, we analyze the statistics of them and
observe the classical scaling laws of the Langevin theory to
explain Brownian Motion.

The example is structured as follows:
  | :ref:`Setup dependencies 1`
  | :ref:`Definition of parameters 1`
  | :ref:`Generating trajectories 1`
  | :ref:`Data analysis and plots 1`
  | :ref:`References 1`

.. note::
   You can access `the script of this example <https://github.com/yupidevs/yupi_examples/blob/master/example_001.py>`_ on the `yupi examples repository <https://github.com/yupidevs/yupi_examples>`_.

.. _Setup dependencies 1:

1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from yupi.generators import LangevinGenerator
   from yupi.stats import (
      msd,
      speed_ensemble,
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
      plot_speed_hist
   )

.. _Definition of parameters 1:

2. Definition of parameters
---------------------------

First, we define some physical constants:


.. code-block:: python

   N0 = 6.02e23     # Avogadro's constant [1/mol]
   k = 1.38e-23     # Boltzmann's constant [J/mol.K]
   T = 300          # absolute temperature [K]
   eta = 1.002e-3   # water viscosity [Pa.s]
   M = 14.1         # lysozyme molar mass [kg/mol] [1]
   d1 = 90e-10      # semi-major axis [m] [2]
   d2 = 18e-10      # semi-minor axis [m] [2]


Then, we can indirectly measure quantities that are 
related with the physical model:

.. code-block:: python

   m = M / N0                   # mass of one molecule
   a = np.sqrt(d1/2 * d2/2)     # radius of the molecule
   alpha = 6 * np.pi * eta * a  # Stoke's coefficient
   v_eq = np.sqrt(k * T / m)    # equilibrium thermal velocity
   tau = m / alpha              # relaxation time


Next, we compute actual statistical model parameters for the
Langevin Generator:

.. code-block:: python

   gamma = 1 / tau                   # drag parameter
   sigma = np.sqrt(2 / tau) * v_eq   # scale parameter of noise pdf


Finally, we define general simulation parameters:

.. code-block:: python

   dim = 2                # trajectory dimension
   N = 1000               # number of trajectories
   dt = 1e-1 * tau        # time step
   tt = 50 * tau          # total time

.. _Generating trajectories 1:

3. Generating trajectories
--------------------------

Once we have all the parameters required to tune the Langevin Generator,
we just need to instantiate the class and generate the Trajectories:

.. code-block:: python

   lg = LangevinGenerator(tt, dim, N, dt, gamma, sigma, seed=0)
   trajs = lg.generate()


.. _Data analysis and plots 1:

4. Data analysis and plots
--------------------------

Let us initialize an empty figure for plot all the results:

.. code-block:: python

   plt.figure(figsize=(9,5))

Plot spacial trajectories

.. code-block:: python

   plt.subplot(231)
   plot_2D(trajs[:5], legend=False, show=False)

Plot speed histogram

.. code-block:: python

   v_norm = speed_ensemble(trajs)
   plt.subplot(232)
   plot_speed_hist(v_norm, bins=20, show=False)

Plot turning angles

.. code-block:: python

   theta = turning_angles_ensemble(trajs)
   ax3 = plt.subplot(233, projection='polar')
   plot_angles_hist(theta, bins=60, ax=ax3, show=False)


Plot Velocity autocorrelation function

.. code-block:: python

   lag_vacf = 50
   vacf, _ = vacf(trajs, time_avg=True, lag=lag_vacf)
   plt.subplot(234)
   plot_vacf(vacf, dt, lag_vacf, show=False)


Plot Mean Square Displacement

.. code-block:: python

   lag_msd = 30
   msd, msd_std = msd(trajs, time_avg=True, lag=lag_msd)
   plt.subplot(235)
   plot_msd(msd, msd_std, dt, lag=lag_msd, show=False)

Plot Kurtosis

.. code-block:: python

   kurt, _ = kurtosis(trajs, time_avg=False, lag=30)
   kurt_ref = kurtosis_reference(trajs)
   plt.subplot(236)
   plot_kurtosis(kurt, kurtosis_ref=kurt_ref, dt=dt, show=False)


Generate plot

.. code-block:: python

   plt.tight_layout()
   plt.show()

.. figure:: /images/example1.png
   :alt: Output of example1
   :align: center

.. _References 1:

5. References
-------------
| [1] Berg, Howard C. Random walks in biology. Princeton University Press, 1993.
| [2] Colvin, J. Ross. "The size and shape of lysozyme." Canadian Journal of Chemistry 30.11 (1952): 831-834.
