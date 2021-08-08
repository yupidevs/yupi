.. _Example 3:

Example 3
=========

Tracking a scaled-size rover wheel moving over sand.
The wheel is forced to move at a fixed angular velocity.
The actual linear velocity is subsequently computed
to evaluate how much does it differs from the ideal
velocity. Code and multimedia resources are available
`here <https://github.com/yupidevs/yupi_examples/>`_.

The work of [1] studied the motion
of vehicles on granular materials experimentally. In their
article, they reported the analysis of the trajectories
performed by a scaled-size wheel while rolling on sand at
two different gravitational accelerations, exploiting the
instrument designed by [2]. This example aims at partially
reproducing some of the results shown in the paper using one
of the original videos provided by the authors.

In the video, one can observe a wheel forced to move on sand
at a fixed angular velocity. In optimal rolling conditions,
one can expect the wheel to move at a constant linear velocity.
However, due to slippage and compaction-decompaction of the
granular soil, the actual linear velocity differs from the one
expected under ideal conditions. To study the factors that affect
the wheel motion, the first step is quantifying how different
the rolling process is with respect to the expected in ideal
conditions.

This example addresses the problem of capturing the trajectory
of the wheel and computing its linear velocity, and the
efficiency of the rolling process.


The example is structured as follows:
  | :ref:`Setup dependencies 3`
  | :ref:`Tracking tracking objects 3`
  | :ref:`Computation of the variables 3`
  | :ref:`Results 3`
  | :ref:`References 3`

.. note::
   You can access `the script of this example <https://github.com/yupidevs/yupi_examples/blob/master/example_003.py>`_ on the `yupi examples repository <https://github.com/yupidevs/yupi_examples>`_.

.. _Setup dependencies 3:

1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   import cv2
   from yupi.tracking import ROI, ObjectTracker, TrackingScenario
   from yupi.tracking import ColorMatching, TemplateMatching
   from yupi.graphics import plot_2D

Set up the path to multimedia resources:

.. code-block:: python

   video_path = 'resources/videos/Viera2017.mp4'
   template_path = 'resources/templates/pivot.png'


.. _Tracking tracking objects 3:

2. Tracking tracking objects
----------------------------

Similarly to the previous example, we start by storing all the
trackers in a list and pass it to the TrackingScenario. We are going
to track the central pivot using TemplateMatching algorithm and the
green LED coupled with the wheel using ColorMatching algorithm.

.. code-block:: python

   trackers = []

   template = cv2.imread(template_path)
   algorithm = TemplateMatching(template, threshold=0.5)
   trackers.append( ObjectTracker('center', algorithm, ROI((80, 80))) )

   algorithm = ColorMatching((80,170,90), (190,255,190))
   trackers.append( ObjectTracker('green led', algorithm, ROI((50, 50))) )

   scenario = TrackingScenario(trackers)

In this case we are forcing the processing to start at frame 10 and stop
in frame 200. Additionally, we are using  a scale factor of 4441
pixels per meter.

.. code-block:: python

   retval, tl = scenario.track(video_path, pix_per_m=4441, start_frame=10, end_frame=200)


.. _Computation of the variables 3:

3. Computation of the variables
-------------------------------

Next, we can estimate the trajectory of the LED referred to the center pivot:

.. code-block:: python

   center, led = tl
   led_centered = led - center
   led_centered.traj_id= 'led'

Since the led and the center of the wheel are placed at a constant distance of
0.039 m, we can estimate the trajectory of the wheel referred to the center
pivot:

.. code-block:: python

   wheel_centered = led_centered.copy()
   wheel_centered.add_polar_offset(0.039, 0)
   wheel_centered.traj_id = 'wheel'
   plot_2D([wheel_centered, led_centered])


.. figure:: /images/polar_offset.png
   :alt: Output of polar offset
   :align: center

Finally, the trajectory of the wheel referred to its initial position, can be
obtained by subtracting the initial from the final position after completing
the whole trajectory.


.. code-block:: python

   wheel = wheel_centered - wheel_centered.r[0]


Now, we can compute the linear velocity in optimal conditions (omega x r)

.. code-block:: python

   v_opt = 4 * 0.07

And compute the linear velocity using the trajectory estimated by the
tracking process:

.. code-block:: python

   v_meas = wheel.v.norm


.. _Results 3:

4. Results
----------

The efficiency of the rolling can be computed as described in [1]:

.. code-block:: python

   eff = v_meas/v_opt

The temporal evolution of the efficiency can be plotted by:

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
despite the constant angular velocity, due to slippery in the terrain.
Even when we are observing only one realization of the experiment,
and assuming the angular velocity of the wheel being perfectly constant,
we can notice the consistency of this result with the ones reported in
the original paper [1].

.. _References 3:

5. References
--------------------------

| [1] Amigó-Vega, J., et al. "Measuring the Performance of a Rover Wheel In Martian Gravity." Revista Cubana de Física 36.1 (2019): 46-50.
| [2] Viera-López, G., et al. "Note: Planetary gravities made simple: Sample test of a Mars rover wheel." Review of Scientific Instruments 88.8 (2017): 086107.
