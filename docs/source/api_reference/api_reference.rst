API Reference
=============

The API of yupi is divided into seven modules:
  * :doc:`yupi`: General classes defining concepts used by all the other modules
  * :doc:`tracking/tracking`: Tools to extract trajectories from image sequences
  * :doc:`generators/generators`: Models to generate trajectories with given statistical constrains.
  * :doc:`transformations/transformations`: Tools to transform trajectories (resamplers, filters, etc.).
  * :doc:`stats/stats`: Tools to extract statistical data from a set of trajectories.
  * :doc:`graphics/graphics`: Tools for visualizing trajectories and statistical data.
    based on the color or the motion of the object being tracked.

.. automodule:: yupi


.. toctree::
   :maxdepth: 2
   :caption: Quick access:
   :hidden:

   yupi
   tracking/tracking
   generators/generators
   transformations/transformations
   stats/stats
   graphics/graphics