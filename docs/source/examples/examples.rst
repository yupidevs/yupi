Examples
--------

In addition to this section, you can find a
`dedicated GitHub repository <https://github.com/yupidevs/yupi_examples/>`_
hosting code examples and additional multimedia resources required to use them.
You may want to clone that repository first in order to easily reproduce the
results shown in this section.

Examples are designed to illustrate a complex integration of
yupi tools by reproducing the results of published research
(with the exception of the first example where we have included
an original equation-based simulation). The selected articles are related
to the analysis of trajectories and its extraction from video sources.
Before start with the examples make sure you have already reviewed the tutorials.

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
          * :py:class:`~generators.LangevinGenerator`
       * Statistics:
          * :py:func:`~stats.speed_ensemble`
          * :py:func:`~stats.turning_angles_ensemble`
          * :py:func:`~stats.msd`
          * :py:func:`~stats.kurtosis`
          * :py:func:`~stats.vacf`
       * Visualization:
          * :py:func:`~graphics.plot_2D`
          * :py:func:`~graphics.plot_velocity_hist`
          * :py:func:`~graphics.plot_angle_distribution`
          * :py:func:`~graphics.plot_msd`
          * :py:func:`~graphics.plot_kurtosis`
          * :py:func:`~graphics.plot_vacf`
   * - | :doc:`Example 2<example2>`
       |
       | A comparison of different tracking methods over the
       | same input video where the camera is fixed at a constant
       | distance from the plane where an ant moves.
     - * Visualization:
          * :py:func:`~graphics.plot_2D`
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
          * :py:func:`~graphics.plot_2D`
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
          * :py:func:`~graphics.plot_2D`
       * Tracking:
          * :py:func:`~tracking.trackers.ROI`
          * :py:func:`~tracking.trackers.ObjectTracker`
          * :py:func:`~tracking.trackers.TrackingScenario`
          * :py:func:`~tracking.algorithms.ColorMatching`
          * :py:func:`~tracking.algorithms.RemapUndistorter`
   * - | :doc:`Example 5<example5>`
       |
       | Simultaneous tracking of an ant and the camera
       | capturing its movement with the reconstruction of the
       | trajectory of the ant respect its initial position.
     - * Visualization:
          * :py:func:`~graphics.plot_2D`
       * Tracking:
          * :py:func:`~tracking.trackers.ROI`
          * :py:func:`~tracking.trackers.ObjectTracker`
          * :py:func:`~tracking.trackers.CameraTracker`
          * :py:func:`~tracking.trackers.TrackingScenario`
          * :py:func:`~tracking.algorithms.ColorMatching`
          * :py:func:`~tracking.algorithms.RemapUndistorter`

.. toctree::
   :maxdepth: 2
   :caption: Advanced Resources
   :hidden:
  
   example1
   example2
   example3
   example4
   example5 