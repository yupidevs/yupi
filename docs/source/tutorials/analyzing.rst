Analysis of Trajectory objects
------------------------------

There are several tools you can use to analyze :py:class:`~yupi.Trajectory`
objects. To illustrate the capabilities of yupi, let us consider a list of
:py:class:`~yupi.Trajectory` objects  generated with a Langevin Generator
(See tutorial https://yupi.readthedocs.io/en/latest/tutorials/generating.html#langevin-generator):


.. code-block:: python

   T = 500     # Total time (number of time steps if dt==1)
   dim = 2     # Dimension of the walker trajectories
   N = 500     # Number of random walkers
   dt = 0.5    # Time step

   tau = 2               # Relaxation time
   noise_pdf = 'normal'  # Noise probabilistic distribution function
   noise_scale = 0.1     # Scale of the noise pdf

    from yupi.generators import LangevinGenerator
    lg = LangevinGenerator(T, dim, N, dt, tau, noise_pdf, noise_scale)
    trajs = lg.generate()


Two-dimensional spacial projections
===================================

The most basic analysis tool is the plot of the trajectories in the space. If
you have a list of :py:class:`~yupi.Trajectory` objects, like the ones you get
from a generator, you can  plot them with:


.. code-block:: python

    from yupi.graphics import plot_2D
    plot_2D(trajs[:10])
  

.. figure:: /images/tutorial001.png
   :alt: Distribution in submodules
   :align: center

We limit to 10 the number of trajectories to plot for the sake of observability.


Histogram of Velocity
=====================

The analysis of the distribution of velocities among all the samples of an
ensemble of trajectories is also posible using:

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

The analysis of the distribution of turning angles alows to understand how
likely is the moving object to turn to specific directions during its motion.
It can be observec with yupi by using:

.. code-block:: python

    from yupi.stats import turning_angles_ensemble
    from yupi.graphics import plot_angles_hist

    theta = turning_angles_ensemble(trajs)
    plot_angles_hist(theta)
  

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

    msd, msd_std = msd(trajs, time_avg=True, lag=30)
    plot_msd(msd, msd_std, dt, lag=30)
  

.. figure:: /images/tutorial004.png
   :alt: Distribution in submodules
   :align: center


Kurtosis
========

Another useful quantity is the kurtosis, $\kappa$, a measure of the disparity of
spatial scales of a dispersal process and also an intuitive means to understand
normality. It can be estimated using:

.. code-block:: python

    from yupi.stats import kurtosis, kurtosis_reference
    from yupi.graphics import plot_kurtosis

    ref = yupi.stats.kurtosis_reference(trajs)
    kurtosis = yupi.stats.kurtosis(trajs, time_avg=False, lag=30)
    yupi.graphics.plot_kurtosis(kurtosis, kurtosis_ref=ref, dt=dt)
  

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

    vacf, _ = vacf(trajs, time_avg=True, lag=50)
    plot_vacf(vacf, dt, 50)
  

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

    psd_mean, psd_std, omega = psd(trajs, lag=150, omega=True)
    plot_psd(psd_mean, omega, psd_std)

.. figure:: /images/tutorial009.png
   :alt: PSD IMAGE
   :align: center