Example 4
=========

Tracking an intruder while penetrating a granular 
material in a quasi 2D enviroment. Code and multimedia resources are 
available `here <https://github.com/yupidevs/yupi_examples/>`_.

The example is structured as follows:
 #. Setup dependencies
 #. Tracking tracking objects
 #. Computation of the variables
 #. Results
 #. References



1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   from yupi.tracking.trackers import ROI, ObjectTracker, TrackingScenario
   from yupi.tracking.undistorters import RemapUndistorter
   from yupi.tracking.algorithms import ColorMatching
   from yupi.analyzing.visualization import plot_trajectories
   from numpy import pi

Set up the path to multimedia resources:

.. code-block:: python

   video_path = 'resources/videos/Diaz2020.MP4'
   camera_file = 'resources/cameras/gph3+.npz'


2. Tracking tracking objects
----------------------------

First, we create an instance of an undistorter to correct the distortion 
caused by the camera lens.

.. code-block:: python

   undistorter = RemapUndistorter(camera_file)


Then, we initialize two trackers, one for each marker of the intruder:

.. code-block:: python

   algorithm1 = ColorMatching((70,40,20), (160,80,20)) # BGR
   cyan = ObjectTracker('cyan marker', algorithm1, ROI((50, 50)))

   algorithm2 = ColorMatching((30,20, 50), (95, 45,120))         
   magenta = ObjectTracker('magenta marker', algorithm2,  ROI((30, 50)))


Create a Tracking Scenario with all the trackers.

.. code-block:: python

   scenario = TrackingScenario([cyan, magenta], 
                            undistorter=undistorter)

Then, we track the video using the preconfigured scenario. We should notice 
that we will have to initialize the Region-of-Interest (ROI) of each tracker 
manually. See the API reference for different initialization methods of ROIs.

.. code-block:: python

   retval, tl = scenario.track(video_path, pix_per_m=2826, start_in_frame=200)
   plot_trajectories(tl)

.. figure:: /images/example4-1.png
   :alt: Output of example4
   :align: center

3. Computation of the variables
-------------------------------

We can improve the standard plot, by making some transformation to the tracked
trajectories. 

First, we can rotate them 90 degrees to better illustrate the effect of 
gravity:

.. code-block:: python

   tl[0].add_polar_offset(0, - pi / 2)
   tl[1].add_polar_offset(0, - pi / 2)


Next, we update the system of reference to place it in the initial position of
the center of the intruder:

.. code-block:: python

   off = tl[0].r[0]
   tl[1] -= off
   tl[0] -= off



4. Results
----------
Now, we can produce a plot quite similar to the one of the original paper [1]:

.. code-block:: python

   plot_trajectories(tl, line_style='-o', connected=True, color=['blue', 'red'])


.. figure:: /images/example4-2.png
   :alt: Output of example42
   :align: center



5. References
--------------------------

| [1] Díaz-Melián, V. L., et al. "Rolling away from the Wall into Granular Matter." Physical Review Letters 125.7 (2020): 078002.
