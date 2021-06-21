Example 3
=========

Tracking a scaled-size rover wheel moving over sand [1]. 
The wheel is forced to move at a fixed angular velocity.
The actual linear displacement is subsequently computed
to evaluate how much does it differs from the ideal 
displacement (a straight line assuming it does not slip 
or sink). Code and multimedia resources are available 
`here <https://github.com/yupidevs/yupi_examples/>`_.

The example is structured as follows:
1. Setup dependencies
2. Tracking tracking objects
3. Computation of the variables
4. Results
5. References


1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   import cv2
   from yupi.tracking import ROI, ObjectTracker, TrackingScenario
   from yupi.tracking import ColorMatching, TemplateMatching
   from yupi.analyzing import plot_trajectories

Set up the path to multimedia resources:

.. code-block:: python

   video_path = 'resources/videos/Viera2017.mp4'
   template_path = 'resources/templates/pivot.png'


2. Tracking tracking objects
----------------------------

First, we create an empty list to add all the trackers. We are going to track
the central pivot and the green led coupled with the wheel.

.. code-block:: python

   trackers = []


We are going to use TemplateMatching algorithm to track the central pivot.
Tt is required to provide a template image containing a typical sample of the 
object being tracked. Then, it will determine the point in a frame in which 
the correlation between the template and the region of the frame is maximum.

.. code-block:: python

   template = cv2.imread(template_path)
   algorithm = TemplateMatching(template, threshold=0.5)
   trackers.append( ObjectTracker('center', algorithm, ROI((80, 80))) )


Then, we are going to track the led using ColorMatching algorithm since we can
easily identify it due to its color. 

.. code-block:: python

   algorithm = ColorMatching((80,170,90), (190,255,190))
   trackers.append( ObjectTracker('green led', algorithm, ROI((50, 50))) )


Create a Tracking Scenario with all the trackers.

.. code-block:: python

   scenario = TrackingScenario(trackers)

Then, we track the video using the preconfigured scenario. We should notice 
that we will have to initialize the Region-of-Interest (ROI) of each tracker 
manually. See the API reference for different initialization methods of ROIs.

.. code-block:: python

   retval, tl = scenario.track(video_path, pix_per_m=4441, start_in_frame=10, end_in_frame=200)



3. Computation of the variables
-------------------------------

First, we estimate the trajectory of the led referred to the center pivot

.. code-block:: python

   center, led = tl
   led_centered = led - center
   led_centered.id = 'led'

Since the led and the center of the wheel are placed at a constant distance of
0.039 m, we can estimate the trajectory of the wheel referred to the center 
pivot.

.. code-block:: python

   wheel_centered = led_centered.copy()
   wheel_centered.add_polar_offset(0.039, 0)
   wheel_centered.id = 'wheel'
   plot_trajectories([wheel_centered, led_centered])


.. figure:: /images/polar_offset.png
   :alt: Output of polar offset
   :align: center

Finally, the trajectory of the wheel referred to its initial position, can be
obtained by subtracting the initial position from the whole trajectory.


.. code-block:: python

   wheel = wheel_centered - wheel_centered.r[0]


4. Results
----------

Using the trajectory of the wheel we can plot the evolution of its linear 
displacement versus time.


.. code-block:: python

   import matplotlib.pyplot as plt
   plt.plot(wheel.t, wheel.r.norm)
   plt.xlabel('time [s]')
   plt.ylabel('linear displacement [m]')
   plt.show()

.. figure:: /images/example3.png
   :alt: Output of example 3
   :align: center

We can notice how the displacement of the wheel is not increasing constantly
despite the constant angular velocity, due to slippery in the terrain [2].

5. References
--------------------------

| [1] Viera-López, G., et al. "Note: Planetary gravities made simple: Sample test of a Mars rover wheel." Review of Scientific Instruments 88.8 (2017): 086107.
| [2] Amigó-Vega, J., et al. "Measuring the Performance of a Rover Wheel In Martian Gravity." Revista Cubana de Física 36.1 (2019): 46-50.
