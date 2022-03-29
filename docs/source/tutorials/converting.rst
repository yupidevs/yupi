Converting your data into Trajectory objects
--------------------------------------------

Yupi stores data using an efficient internal representation based on numpy arrays. If you already have some trajectory data, convert it is possible to convert it into a yupi :py:class:`~yupi.Trajectory`. Next, we show how to create the same trajectory in different ways.


Creating trajectories with x, y and z information
=================================================

When the data belongs to trajectories of dimensions within 1 and 3, it is possible to create a :py:class:`~yupi.Trajectory` by assigning each axis data directly:

.. code-block:: python

   from yupi import Trajectory

   x = [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85]
   y = [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]

   track = Trajectory(x=x, y=y, traj_id="Spiral")

For the three-dimensional case, you can pass a variable ``z`` to the constructor. If the trajectory has more than 4 dimensions, check the next way of creating the object.


Creating trajectories with independent axis information
=======================================================

An extension to the previous case, that can be useful for trajectories of higher dimensions, is to provide all the axis information in a single variable. The following example shows how to proceed in this case:

.. code-block:: python

   from yupi import Trajectory

   axes = [
      [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85],
      [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]
   ]

   track = Trajectory(axes=axes, traj_id="Spiral")


Creating trajectories with independent samples
==============================================

There is also a third way of convert your data into a :py:class:`~yupi.Trajectory`. It requires to pass a list of d-dimensional data points:

.. code-block:: python

   from yupi import Trajectory

   points = [[0, 0], [1.0, 0], [0.63, 0.98], [-0.37, 1.24], [-1.24, 0.69],
             [-1.5, -0.3], [-1.08, -1.23], [-0.19, -1.72], [0.82, -1.63],
             [1.63, -1.01], [1.99, -0.06], [1.85, 0.94]]

   track = Trajectory(points=points, traj_id="Spiral")


Note that the dimension of each point must be equal, and it will define the dimension of the trajectory.

A brief comment on time
=======================

By default, the data will be assumed to be uniformly sampled in time, at a sampling time of 1. If you have the corresponding sequence of time data, you can pass it to the constructor using ``t`` parameter. Alternatively, if the data is uniformly sampled, you can only pass the value of ``dt``.
