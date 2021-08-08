Welcome to yupi's documentation!
================================

Standing for Yet Underused Path Instruments, **yupi** is a set of tools designed for collecting, generating and processing trajectory data. With **yupi** you can handle trajectory data in many different ways. You can generate artificial trajectories from stochastic models, capture the trajectory of an object in a sequence of images, apply transformations to the trajectories such as filtering and resampling or explore different visualizations and statistical properties of a trajectory (or collection of them). Each of the aforementioned functionalities is contained in its own submodule as shown:

.. figure:: /images/modules.png
   :alt: Distribution in submodules
   :align: center

   *Library is arranged in submodules that operate using Trajectory objects.*

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
  
   GS/Installation
   Getting Support <GS/Support>
   GS/Contributing
   GS/About
  
.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   Converting data into Trajectory objects <tutorials/converting>
   tutorials/generating
   tutorials/tracking
   tutorials/operations
   tutorials/storing
   tutorials/analyzing

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

