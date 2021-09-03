.. _Example 6:

Example 6
=========

A model framework of a diffusion process with fluctuating diffusivity 
is presented. A Brownian but non-Gaussian diffusion by means of a coupled 
set of stochastic differential equations is predicted. Position is 
described by an overdamped Langevin equation and the diffusion coefficient 
as the square of an Ornstein-Uhlenbeck process.

The example is focused in computing the probability density function for 
displacements at different time instants for the case of a one-dimensional 
process, as shown analitically by Chechkin et al. in [1].

The example is structured as follows:
  | :ref:`Setup dependencies 6`
  | :ref:`Definition of parameters 6`
  | :ref:`Generating trajectories 6`
  | :ref:`Data analysis and plots 6`
  | :ref:`References 6`

.. note::
   You can access `the script of this example <https://github.com/yupidevs/yupi_examples/blob/master/example_006.py>`_ on the `yupi examples repository <https://github.com/yupidevs/yupi_examples>`_.

.. _Setup dependencies 6:

1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   import numpy as np
   from yupi.stats import collect_at
   from yupi.graphics import plot_hists
   from yupi.generators import DiffDiffGenerator

Fix the random generator seed to make results reproducible:

.. code-block:: python

   np.random.seed(0)


.. _Definition of parameters 6:

2. Definition of parameters
---------------------------

Simulation parameters:


.. code-block:: python

   T = 1000   # Total time of the simulation
   N = 5000   # Number of trajectories
   dt = .1    # Time step


Definition of time instants:

.. code-block:: python

   time_instants = np.array([1, 10, 100])   # Time instants


.. _Generating trajectories 6:

3. Generating trajectories
--------------------------

Once we have all the parameters required,
we just need to instantiate the class and generate the Trajectories:

.. code-block:: python

   dd = DiffDiffGenerator(T, N=N, dt=dt)
   trajs = dd.generate()


.. _Data analysis and plots 6:

4. Data analysis and plots
--------------------------

Let us obtain the position of all the trajectories in the key
time instants:

.. code-block:: python

   r = [collect_at(trajs, 'rx', t, step_as_time=True) for t in time_instants]


Then, we can plot the results:

.. code-block:: python

   plot_hists(r, bins=30, density=True,
      labels=[f't = {dt}' for dt in time_instants],
      xlabel='x',
      ylabel='PDF',
      legend=True,
      grid=True,
      yscale='log',
      ylim=(1e-3, 1),
      xlim=(-20, 20),
      filled=True
   )

.. figure:: /images/example6.png
   :alt: Output of example6
   :align: center

   
.. _References 6:

5. References
-------------

| [1] Chechkin, Aleksei V., et al. "Brownian yet non-Gaussian diffusion: from superstatistics to subordination of diffusing diffusivities." Physical Review X 7.2 (2017): 021002.
| [2] Thapa, Samudrajit, et al. "Bayesian analysis of single-particle tracking data using the nested-sampling algorithm: maximum-likelihood model selection applied to stochastic-diffusivity data." Physical Chemistry Chemical Physics 20.46 (2018): 29018-29037.