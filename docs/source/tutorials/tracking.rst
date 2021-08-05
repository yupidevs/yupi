Getting Trajectory objects from videos
--------------------------------------

There are several methods to discern the position of an object through consecutive frames of a video. If the input is a video where the color of the object you want to track is quite different from everything else, like this one:

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
