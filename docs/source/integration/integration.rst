Third-party integration
-----------------------

The structure of yupi aims to standardize the usage and storage of general purpose trajectories independently of its dimensions. We believe it is useful to be able to convert, when possible, yupi trajectories to the data structures used by other libraries to
empower our users with the tools offered by third parties. With the same spirit, we offer the possibility of converting data from other libraries to yupi trajectories.

As an extension of yupi, we offer yupiwrap, a collection of functions to simplify the conversion of Trajectory data among `yupi <https://yupi.readthedocs.io/en/latest/>`_ and other useful software libraries designed for analyzing trajectories.

Installation
============

Current recommended installation method is via the pypi package:

.. code-block:: bash

  pip install yupiwrap

It will install required dependencies such as `yupi package <https://pypi.org/project/yupi/>`_ from pypi.

Compatible libraries
====================

traja
+++++

The `Traja Python package <https://traja.readthedocs.io/en/latest/index.html>`_ is a toolkit for the numerical characterization and analysis of the trajectories of moving animals. It provides several machine learning tools that are not yet implemented in yupi. Even when it is limited to two-dimensional trajectories, there are many resources that traja can offer when dealing with 2D Trajectories in `yupi`_.

Converting a yupi :py:class:`~yupi.Trajectory` into a *traja DataFrame*
***********************************************************************

Let's create a trajectory with yupi:

.. code-block:: python

  from yupi import Trajectory

  x = [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85]
  y = [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]

  track = Trajectory(x=x, y=y, traj_id="Spiral")


We can convert it to a traja DataFrame simply by:

.. code-block:: python

  from yupiwrap import yupi2traja
  traja_track = yupi2traja(track)


⚠️ Only yupi :py:class:`~yupi.Trajectory` objects with two dimensions can be converted to *traja DataFrame* due to traja limitations.

Converting a *traja DataFrame* into a yupi :py:class:`~yupi.Trajectory`
***********************************************************************

If you have a *traja DataFrame* you can always convert it to a yupi :py:class:`~yupi.Trajectory` by using:

.. code-block:: python

  from yupiwrap import traja2yupi
  yupi_track = traja2yupi(traja_track)


Tracktable
++++++++++

`Tracktable <https://github.com/sandialabs/tracktable>`_ provides a set of tools for handling 2D and 3D trajectories as well as Terrain trajectories. The core data structures and algorithms on this package are implemented in C++ for speed and more efficient memory use.

Converting a yupi :py:class:`~yupi.Trajectory` into a tracktable trajectory
***************************************************************************

Let's create a trajectory with yupi:

.. code-block:: python

  from yupiwrap.tracktable import yupi2tracktable, tracktable2yupi
  from yupi import Trajectory

  # Creating a yupi trajectory representing terrain coordinates
  points = [[-82.359415, 23.135012],[-82.382116, 23.136252]]
  track_1 = Trajectory(points=points, traj_id="ter_track")

  # Creating a 2D yupi trajectory
  points = [[0, 0], [1.0, 0], [0.63, 0.98], [-0.37, 1.24], [-1.24, 0.69],
            [-1.5, -0.3], [-1.08, -1.23], [-0.19, -1.72], [0.82, -1.63],
            [1.63, -1.01], [1.99, -0.06], [1.85, 0.94]]
  track_2 = Trajectory(points=points, traj_id="2d_track")

  # Creating a 3D yupi trajectory
  points = [[0,0,0], [1,1,3], [3,2,5]]
  track_3 = Trajectory(points=points, traj_id="3d_track")


We can convert these tracks to tracktable trajectories simply by:

.. code-block:: python

  tracktable_track_1 = yupi2tracktable(track_1, is_terrestrial=True)
  tracktable_track_2 = yupi2tracktable(track_2)
  tracktable_track_3 = yupi2tracktable(track_3)
 

⚠️ If a 3D yupi :py:class:`~yupi.Trajectory` is converted to a tracktable trajectory with ``is_terrestrial=True`` then the ``z`` axis values are stored as a property called ``'altitude'`` for each point.

⚠️ Only yupi :py:class:`~yupi.Trajectory` objects with two or three dimensions can be converted to tracktable trajectories due to tracktable limitations.

Converting a tracktable trajectory into a yupi :py:class:`~yupi.Trajectory`
***************************************************************************

If you have a tracktable trajectory you can always convert it to a yupi :py:class:`~yupi.Trajectory` by using:

.. code-block:: python

  yupi_track_1 = tracktable2yupi(tracktable_track_1)

