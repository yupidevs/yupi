Converting your data into Trajectory objects
--------------------------------------------

Yupi stores the data using an efficient internal representation based on Numpy arrays. If you already have some trajectory data, convert it to a yupi :py:class:`~yupi.Trajectory` can be done by creating an instance of the class using the expected parameters. All of the examples that we will show create the same trajectory but in a diferent way.

One of the ways to create a :py:class:`~yupi.Trajectory` is assigning each axis data directly:

.. code-block:: python

   from yupi import Trajectory

   x = [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85]
   y = [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]

   track = Trajectory(x=x, y=y, traj_id="Spiral")

On the other hand, if you have all the axis information in a single variable you don't need to extract each axis. The following example shows how to proceed in this case:

.. code-block:: python

   from yupi import Trajectory

   dims = [
      [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85],
      [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]
   ]

   track = Trajectory(dimensions=dims, traj_id="Spiral")

This way its not only more clean but allows the creation of trajectories with more than 3 dimensions.

There is also a third way of convert your data into a :py:class:`~yupi.Trajectory` and it is by giving a list of points:

.. code-block:: python

   from yupi import Trajectory

   points = [[0, 0], [1.0, 0], [0.63, 0.98], [-0.37, 1.24], [-1.24, 0.69],
             [-1.5, -0.3], [-1.08, -1.23], [-0.19, -1.72], [0.82, -1.63],
             [1.63, -1.01], [1.99, -0.06], [1.85, 0.94]]

   track = Trajectory(points=points, traj_id="Spiral")

