Generating artificial Trajectory objects
----------------------------------------

If you want to generate :py:class:`~yupi.Trajectory` objects based on some statistical constrains, you can use one of the available :py:class:`~generating.Generator` to construct a list of :py:class:`~yupi.Trajectory` objects.


Random Walk Generator
=====================

The :py:class:`~yupi.RandomWalkGenerator` is able to simulate the trajectories of a walker, in an arbitrary number of dimensions, according some probabilistic constrains.

We can import it from generating module as:

.. code-block:: python

   from yupi.generators import RandomWalkGenerator

As any other :py:class:`~generating.Generator` in yupi, you can specify the parameters that define the shape of the trajectories, as well as the number of trajectories to generate:

.. code-block:: python

   T = 500     # Total time (number of time steps if dt==1)
   dim = 2     # Dimension of the walker trajectories
   N = 3       # Number of random walkers
   dt = 1      # Time step


The :py:class:`~yupi.RandomWalkGenerator` starts the generation of every trajectory in the origin of the reference frame. Then, iteratively, it computes an increment on each dimension. The increment (also called actions) can be -1, 0 or 1, and it is taken independently on each dimension for each iteration. Additionally, the user can define a list to establish the probabilities of taking each of the available actions:

.. code-block:: python 

   prob = [[.5, .1, .4],   # x-axis
           [.5,  0, .5]]   # y-axis

Notice that the size of this list should coincide with the desired dimensions of the trajectories being generated, and each element of a list should be a 3-element list describing the probability vector of taking the actions [-1, 0, 1] in that dimension.

Then, we can construct a :py:class:`~yupi.RandomWalkGenerator` with the given variables and call its generate method:

.. code-block:: python

   rw = RandomWalkGenerator(T, dim, N, dt, prob)
   tr = rw.generate()

In the variable **tr** we will have a list of **N** :py:class:`~yupi.Trajectory` objects generated using the given configuration.

The generated trajectories can be inspected using the plot_2D function:


.. code-block:: python

   from yupi.graphics import plot_2D
   plot_2D(tr, legend=None)


.. figure:: /images/tutorial007.png
   :alt: Distribution in submodules
   :align: center

Langevin Generator
==================

The :py:class:`~yupi.LangevinGenerator` simulates trajectories governed by the
Langevin Equation. It allows to produce :py:class:`~yupi.Trajectory` objects that quantitatively emulate several systems.

To use it, we first need to define the general parameters for a generator:

.. code-block:: python

    T = 500     # Total time (number of time steps if dt==1)
    dim = 2     # Dimension of the walker trajectories
    N = 3       # Number of random walkers
    dt = 0.5    # Time step


Then, some specific parameters can be set before the generator initialization:

.. code-block:: python

    tau = 1               # Relaxation time
    noise_pdf = 'normal'  # Noise probabilistic distribution function
    noise_scale = 0.1     # Scale of the noise pdf


Finally, the generator is created and the trajectories can be generated:

.. code-block:: python

   from yupi.generators import LangevinGenerator
   lg = LangevinGenerator(T, dim, N, dt, tau, noise_pdf, noise_scale)
   trajectories = lg.generate()

The generated trajectories can be inspected using the plot_2D function:

.. code-block:: python

   from yupi.graphics import plot_2D
   plot_2D(tr, legend=None)

.. figure:: /images/tutorial008.png
   :alt: Distribution in submodules
   :align: center

Although not illustrated in this example, the initial
velocities and positions can be specified in the :py:class:`~yupi.LangevinGenerator`
creation using the **v0** and **r0** parameters respectively.


A more complex application of this :py:class:`~generating.Generator` can be seen in the Example 1.


Defining a Custom Generator
===========================

A user-defined generator can be easily added by building on top of an abstract class :py:class:`~generating.Generator` (which is the base of the already implemented generators).
