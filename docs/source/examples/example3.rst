Example 3
=========

Tracking a scaled-size rover wheel moving over sand. 
The wheel is forced to move at a fixed angular velocity.
The actual linear velocity is subsequently computed
to evaluate how much does it differs from the ideal 
velocity (a straight line assuming it does not slip 
or sink). Code and multimedia resources are available 
`here <https://github.com/yupidevs/yupi_examples/>`_.

The authors of [1] studied the motion of vehicles over 
granular materials experimentally. In their work, they 
report the analysis of the trajectories performed by a 
scaled-size wheel while rolling over sand at two 
different gravitational accelerations, exploiting the 
instrument designed in [2]. This example aims to partially 
reproduce some of the results shown in the paper using 
one of the original videos provided by the authors.

In the video, one can observe a wheel which is forced to 
move over sand at a fixed angular velocity. In optimal 
rolling conditions, one can expect the wheel to move at a 
constant linear velocity. However, due to slippery and 
shrinkage, the actual linear velocity differs from the 
one expected under ideal conditions. To study the factors 
that affect the wheel motion, the first step is quantifying 
how different is the rolling process respect to the expected 
in ideal conditions.

This example addresses the problem of capturing the trajectory 
of the wheel and computing its linear velocity, and the 
efficiency of the rolling process.

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
It is required to provide a template image containing a typical sample of the 
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

Next, we track the video using the configured scenario. We should notice 
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


Now, we can compute the linear velocity in optimal conditions (omega x r)

.. code-block:: python

   v_opt = 4 * 0.07

And the  linear velocity by the results of the tracking:

.. code-block:: python

   v_meas = wheel.v.norm


4. Results
----------

The efficiency of the rolling can be computed as described in [1]:

.. code-block:: python

   eff = v_meas/v_opt

And we can see the evolution of the efficiency vs time:

.. code-block:: python

   import matplotlib.pyplot as plt
   plt.plot(wheel.t[1:], eff)
   plt.xlabel('time [s]')
   plt.ylabel('efficiency')
   plt.show()

.. figure:: /images/example3.png
   :alt: Output of example 3
   :align: center

We can notice how the linear velocity of the wheel is not constant
despite the constant angular velocity, due to slippery in the terrain [1].

5. References
--------------------------

| [1] Amigó-Vega, J., et al. "Measuring the Performance of a Rover Wheel In Martian Gravity." Revista Cubana de Física 36.1 (2019): 46-50.
| [2] Viera-López, G., et al. "Note: Planetary gravities made simple: Sample test of a Mars rover wheel." Review of Scientific Instruments 88.8 (2017): 086107.
