Getting Started
===============

With yupi you can handle trajectory data in many different ways. You can generate artificial trajectories from stochastic models, capture the trajectory of an object in a sequence of images or explore different visualizations and statistical properties of a trajectory (or collection of trajectories). Each of the aforementioned functionalities is contained in its own submodule as shown:

.. figure:: /images/modules.png
   :alt: Distribution in submodules
   :align: center

   *Library is arranged in submodules that operate using Trajectory objects.*


Creating Trajectory objects
---------------------------

We can see from the figure that everything is supposed to work with Trayectory objects. There are 3 different ways to construct this kind of object and next we are going to cover them all.

Converting your data into Trajectory objects
++++++++++++++++++++++++++++++++++++++++++++

Yupi stores the data using an efficient internal representation based on Numpy arrays. If you already have some trajectory data, convert it to a yupi Trajectory can be done by creating an instance of the class using the expected parameters.  For instance, if you have a bidimensional trajectory sampled at a constant time *dt*. 

.. code-block:: python

   from yupi import Trajectory

   x = [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85]
   y = [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]

   track = Trajectory(x_arr=x, y_arr=y, id="Spiral")


Generating artificial Trajectory objects
++++++++++++++++++++++++++++++++++++++++

If you want to generate Trajectory objects based on some statistical constrains, you can use a Generator:

.. code-block:: python

   from yupi import Trajectory
   Generators('Comming Soon')


Extracting Trajectory objects from videos
+++++++++++++++++++++++++++++++++++++++++

If your input data is a video like this one:

VIDEO SHOWN HERE

You can create a TrackingScenario to capture the center of the red ball using:

.. code-block:: python

   from yupi import Trajectory
   TrackingScenario('Comming Soon')



Writting and Reading Trajectory objects
---------------------------------------

Regardless the source of the Trajectory object, you can save it on disk and later load it for further processing.

Writting Trajectory objects
+++++++++++++++++++++++++++

To store your Trajectory object:

.. code-block:: python

   track.save('spiral', file_type='json')


Reading Trajectory objects
++++++++++++++++++++++++++

To load a previously written Trajectory object:

.. code-block:: python

   from yupi import Trajectory
   track2 = Trajectory.load('spiral.json')


Sample analysis of Trajectory objects
-------------------------------------

coming soon...

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   