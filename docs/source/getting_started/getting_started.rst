Getting Started
===============

With yupi you can handle trajectory data in many different ways. You can generate artificial trajectories from stochastic models, capture the trajectory of an object in a sequence of images or explore different visualizations and statistical properties of a trajectory (or collection of trajectories). Each of the aforementioned functionalities is contained in its own submodule as shown:

.. figure:: /images/modules.png
   :alt: Distribution in submodules
   :align: center

   *Library is arranged in submodules that operate using Trajectory objects.*


Creating Trajectory objects
---------------------------

We can see from the figure that everything is supposed to work with :py:class:`~yupi.Trajectory` objects. There are 3 different ways to construct this kind of object and next we are going to cover them all.

Converting your data into Trajectory objects
++++++++++++++++++++++++++++++++++++++++++++

Yupi stores the data using an efficient internal representation based on Numpy arrays. If you already have some trajectory data, convert it to a yupi :py:class:`~yupi.Trajectory` can be done by creating an instance of the class using the expected parameters.  A simple example:

.. code-block:: python

   from yupi import Trajectory

   x = [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85]
   y = [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]

   track = Trajectory(x=x, y=y, traj_id="Spiral")


Generating artificial Trajectory objects
++++++++++++++++++++++++++++++++++++++++

If you want to generate :py:class:`~yupi.Trajectory` objects based on some statistical constrains, you can use a :py:class:`~generating.Generator` to construct a list of :py:class:`~yupi.Trajectory` objects:

.. code-block:: python

   from yupi.generating import LatticeRandomWalkGenerator
      
   # set parameter values
   T = 500     # total time (number of time steps if dt==1)
   dim = 2     # dimension of the walker trajectories
   N = 3       # number of random walkers
   dt = 1      # time step

   # vector of actions the walker can take:
   # [move to the right/up, stay quiet, move to the left/down]
   actions = [1, 0, -1]

   # probability of every action to be taken
   # according to every axis
   prob = [[.5, .1, .4],  # x-axis
         [.5, 0, .5]]   # y-axis

   # get RandomWalk object and get position vectors
   rw = LatticeRandomWalkGenerator(T, dim, N, dt, actions, prob)
   tr = rw.generate()


Extracting Trajectory objects from videos
+++++++++++++++++++++++++++++++++++++++++

There are several methods to discern the position of an object through the frames. If your input is a video where the color of the object you want to track is quite different from everything else, like this one:

.. raw:: html

   <center>
   <video width="400" controls>   
      <source src="../_static/demo.mp4" type="video/mp4">
   </video>
   </center>

You can exploit this fact to capture the whole trajectory using a yupi script like:

.. code-block:: python

   from yupi.tracking import ROI, ObjectTracker, TrackingScenario
   from yupi.tracking import ColorMatching

   # Initialize main tracking objects
   algorithm = ColorMatching((180,125,35), (190,135,45))
   blue_ball = ObjectTracker('blue', algorithm, ROI((100, 100)))
   scenario = TrackingScenario([blue_ball])

   # Track the video using the preconfigured scenario
   retval, tl = scenario.track('resources/videos/demo.avi', pix_per_m=10)

The value of ``tl``, will contain a list of all the :py:class:`~yupi.Trajectory` objects the :py:class:`~tracking.trackers.TrackingScenario` tracked among all the frames of the video. In this case, the list will contain only one object describing the trajectory of the blue ball in the video.

Writting and Reading Trajectory objects
---------------------------------------

Regardless the source of the :py:class:`~yupi.Trajectory` object, you can save it on disk and later load it for further processing.

Writting Trajectory objects
+++++++++++++++++++++++++++

To store your :py:class:`~yupi.Trajectory` object, for instance the same we build at the begining, you only need to call the :py:class:`~yupi.Trajectory.save` method as in:

.. code-block:: python

   track.save('spiral', file_type='json')


Reading Trajectory objects
++++++++++++++++++++++++++

To :py:class:`~yupi.Trajectory.load` a previously written :py:class:`~yupi.Trajectory` object:

.. code-block:: python

   from yupi import Trajectory
   track2 = Trajectory.load('spiral.json')


Sample analysis of Trajectory objects
-------------------------------------

There are several tools you can use to analyze :py:class:`~yupi.Trajectory` objects. The most basic one is the plot of the trajectories in the space. If you have a list of :py:class:`~yupi.Trajectory` objects, like the ones you get from a generator, you can plot them with:


.. code-block:: python

   from yupi.analyzing import plot_trajectories
   plot_trajectories(tr)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   