Analysis of Trajectory objects
------------------------------

There are several tools you can use to analyze :py:class:`~yupi.Trajectory`
objects. To illustrate the capabilities of yupi, let us consider a list of
:py:class:`~yupi.Trajectory` objects  generated with a Langevin Generator (See
dedicated tutorial of :ref:`Langevin Generator` for a more detailed explanation
of the parameters.)


.. code-block:: python

    from yupi.generators import LangevinGenerator
    lg = LangevinGenerator(500, 2, 500, 0.5, 2, 0.1, seed=0)
    trajs = lg.generate()


Two-dimensional spatial projections
===================================

The most basic analysis tool is the plot of the trajectories in the space. If
you have a list of :py:class:`~yupi.Trajectory` objects, like the ones you get
from a generator, you can  plot them with:


.. code-block:: python

    from yupi.graphics import plot_2D
    plot_2D(trajs[:10], legend=False)
  

.. figure:: /images/tutorial001.png
   :alt: Distribution in submodules
   :align: center

Notice that we limited to 10 the number of trajectories to plot for the sake of
observability, but we will be using the full list of trajectories (traj) all
over this tutorial.


Three-dimensional spatial projections
=====================================

Plotting in three dimensions can be achieved in a similar way. Let us generate
3D trajectories:

.. code-block:: python

    from yupi.generators import LangevinGenerator
    lg = LangevinGenerator(500, 3, 5, 0.5, 2, 0.1, seed=0)
    trajs3D = lg.generate()


Then, we can plot them using:

.. code-block:: python

    from yupi.graphics import plot_3D
    plot_3D(trajs3D, legend=False)


.. figure:: /images/tutorial011.png
   :alt: Distribution in submodules
   :align: center


Histogram of Velocity
=====================

The analysis of the distribution of velocities among all the samples of an
ensemble of trajectories is also possible using:

.. code-block:: python

    from yupi.stats import speed_ensemble
    from yupi.graphics import plot_velocity_hist

    v = speed_ensemble(trajs, step=1)
    plot_velocity_hist(v, bins=20)
  

.. figure:: /images/tutorial002.png
   :alt: Distribution in submodules
   :align: center


Histogram of Turning Angles
===========================

The analysis of the distribution of turning angles allows to understand how
likely is the moving object to turn to specific directions during its motion.
It can be observe with yupi by using:

.. code-block:: python

    from yupi.stats import turning_angles_ensemble
    from yupi.graphics import plot_angles_hist

    theta = turning_angles_ensemble(trajs)
    plot_angles_hist(theta, bins=30)
  

.. figure:: /images/tutorial003.png
   :alt: Distribution in submodules
   :align: center


Mean Squared Displacement
=========================

The Mean Square Displacement (MSD) is a typical indicator to classify processes
away from normal diffusion. The MSD of a normal diffusive trajectory arises as
a linear function of time. To estimate the MSD of a list of
:py:class:`~yupi.Trajectory` objects, you can use:

.. code-block:: python

    from yupi.stats import msd
    from yupi.graphics import plot_msd

    msd_mean, msd_std = msd(trajs, time_avg=True, lag=30)
    plot_msd(msd_mean, msd_std, 0.5, lag=30)
  

.. figure:: /images/tutorial004.png
   :alt: Distribution in submodules
   :align: center


Kurtosis
========

Another useful quantity is the kurtosis, a measure of the disparity of spatial
scales of a dispersal process and also an intuitive means to understand
normality. It can be estimated using:

.. code-block:: python

    from yupi.stats import kurtosis, kurtosis_reference
    from yupi.graphics import plot_kurtosis

    kurt_ref = kurtosis_reference(trajs)
    kurt_mean, _ = kurtosis(trajs, time_avg=False, lag=30)
    plot_kurtosis(kurt_mean, kurtosis_ref=kurt_ref, dt=dt)
  

.. figure:: /images/tutorial005.png
   :alt: Distribution in submodules
   :align: center


Velocity Autocorrelation Function
=================================

The Velocity Autocorrelation Function (VACF) gives valuable information about
the influence of correlations during a whole trajectory. To compute it and plot
the results, you can use:

.. code-block:: python

    from yupi.stats import vacf
    from yupi.graphics import plot_vacf

    vacf_mean, _ = vacf(trajs, time_avg=True, lag=50)
    plot_vacf(vacf_mean, dt, 50)
  

.. figure:: /images/tutorial006.png
   :alt: Distribution in submodules
   :align: center



Power Spectral Density
======================

The Power Spectral Density, or Power Spectrum, indicates the frequency content
of the trajectory. The inspection of the PSD from a collection of trajectories
enables the characterization of the motion in terms of the frequency components.

.. code-block:: python

    from yupi.stats import psd
    from yupi.graphics import plot_psd

    psd_mean, psd_std, frec = psd(trajs, lag=150)
    plot_psd(psd_mean, frec, psd_std)

.. figure:: /images/tutorial009.png
   :alt: PSD IMAGE
   :align: center
