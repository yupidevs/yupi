Extracting data from an ensemble of trajectories
------------------------------------------------

Overview
========

Collecting data such as displacements, velocities, speeds is a common task when
analyzing trajectory data. Therefore, looping over an ensemble of trajectories
ends up in repeated blocks of code specially when the data is wanted to, for
instance, be analyzed at different time scales or be concatenated.

This tutorial provides an step-by-step view over the :py:func:`~stats.collect`
function. It gets a friendly way to "collect" the basic time series researchers
that process trajectories are continuously dealing with.


Let's begin
===========

Let us first create a fake ensemble of just two (for the sake of illustration)
trajectories using the :py:class:`~yupi.Trajectory` class from `yupi`.

.. code-block:: python

   import numpy as np
   from yupi import Trajectory
   from yupi.stats import collect

   # x-coordinate for the first/second trajectory
   x1 = 2 * np.arange(5)
   x2 = x1 + 1

   # y-coordinates
   y1 = x1**2
   y2 = x2**2

   # Time step and ids
   dt = .5
   id_traj1 = 'traj_01'
   id_traj2 = 'traj_02'

   # Instantiating the class
   traj1 = Trajectory(x=x1, y=y1, dt=dt, traj_id=id_traj1)
   traj2 = Trajectory(x=x2, y=y2, dt=dt, traj_id=id_traj2)

   # Gather in an ensemble
   trajs = [traj1, traj2]


At this point, it is quite easy using `yupi` to extract some time series from a 
single trajectory. For example:

- position: ``traj1.r``
- displacement: ``traj1.r.delta``
- velocity: ``traj1.v``
- speed: ``traj1.v.norm``
- distance: ``traj1.delta_r.norm``

But how to get these from an ensemble? How to extract it when time scales are
different from the time step? In what is next, these questions are answered
with some illustrative examples while a detailed explanation of every parameter
is given on the :py:func:`~stats.collect` function.

Simple collect
==============

By default, the :py:func:`~stats.collect` function takes a list trajectory an returns
an array with all the positional data of each trajectory concatenated.

.. code-block:: python

   collect(trajs)

.. code-block:: text

    array([[ 0.  0.]
           [ 2.  4.]
           [ 4. 16.]
           [ 6. 36.]
           [ 8. 64.]
           [ 1.  1.]
           [ 3.  9.]
           [ 5. 25.]
           [ 7. 49.]
           [ 9. 81.]])

The following sections will describe all the parameters available that manipulate 
the resulting data within the :py:func:`~stats.collect` function.

The ``lag_step`` and ``lag_time`` parameters
============================================

Suppose the underlying ensemble of trajectories as being realizations of a
process with different statistical properties at different time scales. For
such a case, ``lag_step`` and ``lag_time`` can be helpful if they are set
properly. If lag is an integer that account for number of samples, ``lag_step``
should be used. Instead, use ``lag_time`` if its units are those of the time
array (i.e., ``traj.t``).

If none of this parameters are given, ``lag_step=0`` will be assumed.

.. code-block:: python

   collect(trajs, lag_step=2)

.. code-block:: text

   array([[ 4., 16.],
          [ 4., 32.],
          [ 4., 48.],
          [ 4., 24.],
          [ 4., 40.],
          [ 4., 56.]])

.. code-block:: python

   collect(trajs, lag_time=2*dt)

.. code-block:: text

   array([[ 4., 16.],
          [ 4., 32.],
          [ 4., 48.],
          [ 4., 24.],
          [ 4., 40.],
          [ 4., 56.]])

The ``concat`` parameter
========================

As we show in the very first example, the code for `concat(trajs)` will return
an array with all the positional data of each trajectory concatenated.

If the data is wanted to be split by realizations, the ``concat`` parameter
should be set to ``False``.

.. code-block:: python

   collect(trajs, concat=False)

.. code-block:: text

   array([[[ 0.,  0.],
           [ 2.,  4.],
           [ 4., 16.],
           [ 6., 36.],
           [ 8., 64.]],

          [[ 1.,  1.],
           [ 3.,  9.],
           [ 5., 25.],
           [ 7., 49.],
           [ 9., 81.]]])

The ``warnings`` parameter
==========================

If the given lag is larger than one of the trajectories length, a warning
message will arise and the position of the trajectory in the ensemble and its
*id* will be shown. The :py:func:`~stats.collect` function will skip this
trajectory. To avoid warning messages set the parameter to ``False``.

.. code-block:: python

   traj1.dt = .01  # redefining dt for the first trajectory
   collect(trajs, lag_time=dt)

.. code-block:: text

   15:07:11 [WARNING] Trajectory 0 with id=traj_01 is shorten than 50 samples
   array([[ 2.,  8.],
          [ 2., 16.],
          [ 2., 24.],
          [ 2., 32.]])

.. code-block:: python

   collect(trajs, lag_time=dt, warnings=False)

.. code-block:: text

   array([[ 2.,  8.],
          [ 2., 16.],
          [ 2., 24.],
          [ 2., 32.]])

The ``velocity`` parameter
==========================

Some times it is useful to have the velocity of the trajectory. To indicate that
the velocity is wanted, the ``velocity`` parameter should be set to ``True``.

.. code-block:: python

   collect(trajs, velocity=True)

.. code-block:: text

    array([[ 4.  8.]
           [ 4. 24.]
           [ 4. 40.]
           [ 4. 56.]
           [ 4. 16.]
           [ 4. 32.]
           [ 4. 48.]
           [ 4. 64.]])

Additional if the ``lag_step`` (or ``lag_time``) is used, the velocity will be
calculated according the given lag.

.. code-block:: python

   collect(trajs, lag_step=2, velocity=True)

.. code-block:: text

    array([[ 4. 16.]
           [ 4. 32.]
           [ 4. 48.]
           [ 4. 24.]
           [ 4. 40.]
           [ 4. 56.]])

The ``func`` parameter
======================

All the examples described above only returns raw data from the trajectories. If
the data is wanted to be transformed, the ``func`` parameter should be set to
a function that will be applied to each vector (before concatenation).

This could help if we want to extract for example the delta velocity of the
trajectories.

.. code-block:: python

   collect(trajs, velocity=True, func=lambda vec: vec.delta)

.. code-block:: text

    array([[ 0. 16.]
           [ 0. 16.]
           [ 0. 16.]
           [ 0. 16.]
           [ 0. 16.]
           [ 0. 16.]])

.. code-block:: python

   collect(trajs, func=lambda vec: vec.norm)

.. code-block:: text

    array([ 4.47213595 12.16552506 20.09975124 28.0713377 8.24621125 16.1245155
           24.08318916 32.06243908])
