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

   import matplotlib.pyplot as plt
   import numpy as np

   from yupi.generators import LangevinGenerator
   from yupi.graphics import plot_2d
   from yupi.stats.kurtosis import KurtosisStat
   from yupi.stats.msd import MsdTimeAvgStat
   from yupi.stats.speed import SpeedStat
   from yupi.stats.turning_angles import TurningAngleStat
   from yupi.stats.vacf import VacfTimeAvgStat

.. _Definition of parameters 1:

2. Definition of parameters
---------------------------

First, we define some physical constants:


.. code-block:: python

   N0 = 6.02e23      # Avogadro's constant [1/mol]
   k = 1.38e-23      # Boltzmann's constant [J/mol.K]
   T = 300           # absolute temperature [K]
   eta = 1.002e-3    # water viscosity [Pa.s]
   M = 14.1          # lysozyme molar mass [kg/mol] [1]
   d1 = 90e-10       # semi-major axis [m] [2]
   d2 = 18e-10       # semi-minor axis [m] [2]


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

   lg = LangevinGenerator(T=tt, dim=dim, dt=dt, gamma=gamma, sigma=sigma, seed=0)
   trajs = lg.generate(N)


.. _Data analysis and plots 1:

4. Data analysis and plots
--------------------------

Let us initialize an empty figure for plot all the results:

.. code-block:: python

   plt.figure(figsize=(9,5))


Plot spacial trajectories

.. code-block:: python

   plot_2d(trajs[:5], legend=False, ax=plt.subplot(231), show=False)


Plot speed histogram

.. code-block:: python

   SpeedStat(trajs).plot(bins=20, ax=plt.subplot(232), show=False)


Plot turning angles

.. code-block:: python

   TurningAngleStat(trajs).plot(
      bins=60, ax=plt.subplot(233, projection="polar"), show=False
   )


Plot Velocity autocorrelation function

.. code-block:: python

   VacfTimeAvgStat(trajs, lag=50).plot(ax=plt.subplot(234), show=False)


Plot Mean Square Displacement

.. code-block:: python

   MsdTimeAvgStat(trajs, lag=50).plot(ax=plt.subplot(235), show=False)


Plot Kurtosis

.. code-block:: python

   KurtosisStat(trajs).plot(ax=plt.subplot(236), show=False)


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
