Welcome to yupi's documentation!
================================

.. image:: https://zenodo.org/badge/304602979.svg
   :target: https://zenodo.org/badge/latestdoi/304602979

Standing for Yet Underused Path Instruments, **yupi** is a set of tools designed for collecting, generating and processing trajectory data. With **yupi** you can handle trajectory data in many different ways. You can generate artificial trajectories from stochastic models, capture the trajectory of an object in a sequence of images, apply transformations to the trajectories such as filtering and resampling or explore different visualizations and statistical properties of a trajectory (or collection of them). Each of the aforementioned functionalities is contained in its own submodule as shown:

.. figure:: /images/modules.png
   :alt: Distribution in submodules
   :align: center

   *Library is arranged in submodules that operate using Trajectory objects.*

Main features
-------------

- **Convert raw data to trajectories** ... *different input manners*
- **I/O operations with trajectories** ... *json and csv serializers*
- **Trajectory extraction from video inputs** ... *even with moving camera*
- **Artificial trajectory generation** ... *several models implemented*
- **Trajectory basic operations** ... *rotation, shift, scaling, ...*
- **Trajectory transformations** ... *filters, resamplers, ...*
- **Statistical analysis rom trajectories ensembles** ... *turning angles histogram, velocity autocorrelation function, power spectral density, and much more ...*
- **Results visualization** ... *each statistical observable has a related plot function*
- **Spacial projection of trajectories** ... *for 2D and 3D trajectories*


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
  
   GS/Installation
   Getting Support <GS/Support>
   GS/Contributing
   GS/About
   GS/FAQ
  
.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   Converting data into Trajectory objects <tutorials/converting>
   tutorials/generating
   tutorials/tracking
   tutorials/operations
   tutorials/storing
   tutorials/analyzing
   tutorials/collecting

.. toctree::
   :maxdepth: 2
   :caption: Advanced Resources

   examples/examples
   api_reference/api_reference
   integration/integration

Indices
-------

* :ref:`genindex`
* :ref:`modindex`

