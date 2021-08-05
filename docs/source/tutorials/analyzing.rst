Sample analysis of Trajectory objects
-------------------------------------

There are several tools you can use to analyze :py:class:`~yupi.Trajectory` objects. The most basic one is the plot of the trajectories in the space. If you have a list of :py:class:`~yupi.Trajectory` objects, like the ones you get from a generator, you can plot them with:


.. code-block:: python

   from yupi.analyzing import plot_trajectories
   plot_trajectories(tr)
   