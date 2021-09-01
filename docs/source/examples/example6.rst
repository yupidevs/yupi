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
   import matplotlib.pyplot as plt
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


Definition of key time instants:

.. code-block:: python

   delta_t = np.array([1, 10, 100])   # Time instants
   steps = np.int32(delta_t / dt)     # Same but in steps of the simulation


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

   r = [[traj.r.x[step] for traj in trajs] for step in steps]


Then, we can plot the results:

.. code-block:: python

   for r_, delta_t_ in zip(r, delta_t):
      plt.hist(r_, bins=30, density=True, histtype='step',
               label=f't = {delta_t_}')
   plt.legend()
   plt.grid(True)
   plt.ylim(1e-3, 1)
   plt.xlim(-20, 20)
   plt.yscale('log')
   plt.xlabel('x')
   plt.ylabel('PDF')
   plt.show()

.. figure:: /images/example6.png
   :alt: Output of example6
   :align: center

   
.. _References 6:

5. References
-------------

| [1] Chechkin, Aleksei V., et al. "Brownian yet non-Gaussian diffusion: from superstatistics to subordination of diffusing diffusivities." Physical Review X 7.2 (2017): 021002.
| [2] Thapa, Samudrajit, et al. "Bayesian analysis of single-particle tracking data using the nested-sampling algorithm: maximum-likelihood model selection applied to stochastic-diffusivity data." Physical Chemistry Chemical Physics 20.46 (2018): 29018-29037.