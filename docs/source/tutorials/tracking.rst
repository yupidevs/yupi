Getting Trajectory objects from videos
--------------------------------------

There are several methods to discern the position of an object through consecutive frames of a video.

Color Matching
==============

If the input is a video where the color of the object you want to track is quite different from everything else, like this one:

.. raw:: html

   <center>
   <video width="400" controls>
      <source src="../_static/demo.mp4" type="video/mp4">
   </video>
   </center>

You can exploit this fact to capture the whole trajectory using the :py:class:`~yupi.ColorMatching` algorithm implemented in yupi:

.. code-block:: python

   from yupi.tracking import ColorMatching
   algorithm = ColorMatching((180,125,35), (190,135,45))

Where the parameters passed to the :py:class:`~tracking.algorithms.ColorMatching` constructor are the lower and upper bounds of the color vector in the selected color space, BGR by default.

Next, we can define a Region-of-Interest (:py:class:`~tracking.trackers.ROI`), the neighborhood of pixels, around the last known position of the object, that are going to be explored in the following frame. Its size will depend on the specific video and the desired tracking object.

.. code-block:: python

   from yupi.tracking import ROI
   roi = ROI((100, 100))

Now, we can create an :py:class:`~tracking.trackers.ObjectTracker`. Its function is to apply the selected algorithm along the :py:class:`~tracking.trackers.ROI` in a frame to estimate the following position of the object.

.. code-block:: python

   from yupi.tracking import ObjectTracker
   blue_ball = ObjectTracker('blue', algorithm, roi)

Finally, the tracker is passed to the :py:class:`~tracking.trackers.TrackingScenario`, the one in charge of iterating the video and making the trackers update its value on each frame. It also allows several trackers to coexist while processing the same video.


.. code-block:: python

   from yupi.tracking import TrackingScenario
   scenario = TrackingScenario([blue_ball])

The track method of a :py:class:`~tracking.trackers.TrackingScenario` object, will produce a list of all the :py:class:`~yupi.Trajectory` objects the :py:class:`~tracking.trackers.TrackingScenario` tracked among all the frames of the video:

.. code-block:: python

   retval, tl = scenario.track('resources/videos/demo.avi', pix_per_m=10)

In this case, the list ``tl`` will contain only one object describing the trajectory of the blue ball in the video.


Other Tracking Algorithms
=========================

There are several other algorithms available in yupi (see :doc:`../api_reference/tracking/algorithms` section on the :doc:`../api_reference/api_reference`). For a quick overview, we recommend you to inspect :ref:`Example 2` that contains a detailed comparison of them while developing a specific task.


Tracking objects when the camera is moving
==========================================

Yupi is able to estimate the motion of the camera (under certain circumstances) and integrate this information while reconstructing the trajectory of the tracked object. :ref:`Example 5` shows a typical application of this feature.