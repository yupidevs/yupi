.. Yupi documentation master file, created by
   sphinx-quickstart on Sat Feb 27 14:36:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to yupi's documentation!
================================

Standing for Yet Underused Path Instruments, **yupi** is a set of tools designed for collecting, generating and processing trajectory data. With **yupi** you can handle trajectory data in many different ways. You can generate artificial trajectories from stochastic models, capture the trajectory of an object in a sequence of images or explore different visualizations and statistical properties of a trajectory (or collection of trajectories). Each of the aforementioned functionalities is contained in its own submodule as shown:

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
   integration/integration
   api_reference/api_reference

Indices
-------

* :ref:`genindex`
* :ref:`modindex`

