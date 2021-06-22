Examples
========

In addition to this section, you can find a 
`dedicated GitHub repository <https://github.com/yupidevs/yupi_examples/>`_
hosting code examples and additional multimedia resources required to use them.
You may want to clone that repository first in order to easily reproduce the
results shown in this section.

Examples are designed to illustrate a complex integration of yupi tools in 
order to reproduce the results of an existing (or yet to come) scientific paper. 
Before start with the examples make sure you have already read our :doc:`Getting 
Started<../getting_started/getting_started>` section.

Description of current examples
-------------------------------

In this table you can easily find the examples that better suits you.

.. list-table::
   :header-rows: 1
   
   * -
     - Related API functions
   * - | :doc:`Example 1<example1>`
       |
       | A simulation of the statistical properties for the motion
       | of a lysozyme molecule in water. Several molecule 
       | trajectories are generated and later analyzed.
     - * Generation:
          * :py:class:`~generating.LangevinGenerator`
       * Statistics:
          * :py:func:`~statistics.estimate_velocity_samples`
          * :py:func:`~statistics.estimate_turning_angles`
          * :py:func:`~statistics.estimate_msd`
          * :py:func:`~statistics.estimate_kurtosis`
          * :py:func:`~statistics.estimate_vacf`
       * Visualization:
          * :py:func:`~visualization.plot_trajectories`
          * :py:func:`~visualization.plot_velocity_hist`
          * :py:func:`~visualization.plot_angle_distribution`
          * :py:func:`~visualization.plot_msd`
          * :py:func:`~visualization.plot_kurtosis`
          * :py:func:`~visualization.plot_vacf`
   * - | :doc:`Example 2<example2>`
       |
       | A comparison of different tracking methods over the 
       | same input video where the camera is fixed at a constant 
       | distance from the plane where an ant moves.
     - * Visualization:
          * :py:func:`~visualization.plot_trajectories`
       * Tracking:
          * :py:func:`~tracking.trackers.ROI`
          * :py:func:`~tracking.trackers.ObjectTracker`
          * :py:func:`~tracking.trackers.TrackingScenario`
          * :py:func:`~tracking.algorithms.ColorMatching`
          * :py:func:`~tracking.algorithms.FrameDifferencing`
          * :py:func:`~tracking.algorithms.BackgroundEstimator`
          * :py:func:`~tracking.algorithms.BackgroundSubtraction`
          * :py:func:`~tracking.algorithms.TemplateMatching`
          * :py:func:`~tracking.algorithms.OpticalFlow`
   * - | :doc:`Example 3<example3>`
       |
       | Tracking a scaled-size rover wheel moving over the sand. 
       | The position is subsequently compared to the ideal 
       | position assuming it does not slip or sink.
     - * Visualization:
          * :py:func:`~visualization.plot_trajectories`
       * Tracking:
          * :py:func:`~tracking.trackers.ROI`
          * :py:func:`~tracking.trackers.ObjectTracker`
          * :py:func:`~tracking.trackers.TrackingScenario`
          * :py:func:`~tracking.algorithms.ColorMatching`
          * :py:func:`~tracking.algorithms.TemplateMatching`
   * - | :doc:`Example 4<example4>`
       |
       | Tracking an intruder while penetrating a granular 
       | material in a quasi 2D enviroment.
     - * Visualization:
          * :py:func:`~visualization.plot_trajectories`
       * Tracking:
          * :py:func:`~tracking.trackers.ROI`
          * :py:func:`~tracking.trackers.ObjectTracker`
          * :py:func:`~tracking.trackers.TrackingScenario`
          * :py:func:`~tracking.algorithms.ColorMatching`
          * :py:func:`~tracking.algorithms.RemapUndistorter`


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   example1 
   example2 
   example3 
   example4 