:py:class:`~yupi.stats.collect`: Extracting data from an ensemble of trajectories
----------------------------------------

1. Overview

Collecting data such as displacements, velocities, speeds is a common task when analyzing trajectory data. Therefore, looping over an ensemble of trajectories ends up in repeated blocks of code specially when the data is wanted to, for instance, be analyzed at different time scales or be concatenated.

This tutorial provides an step-by-step view over the ``collect`` method from `yupi`, a python library to handle trajectory data in python. It gets a friendly way to "collect" the basic time series researchers that process trajectories are continuously dealing with.


2. Let's begin

Let us first create a fake ensemble of just two (for the sake of illustration) trajectories using the :py:class:`~yupi.Trajectory` class from `yupi`.

.. code-block:: python

   import numpy as np
   from yupi import Trajectory

   # x-coordinate for the first/second trajectory
   x1 = 2 * np.arange(5)
   x2 = x1 + 1

   # y-coordinates
   y1 = x1**2
   y2 = x2**2

   # time step and ids
   dt = .5
   id_traj1 = 'traj_01'
   id_traj2 = 'traj_02'

   # instantiating the class
   traj1 = Trajectory(x=x1, y=y1, dt=dt, traj_id=id_traj1)
   traj2 = Trajectory(x=x2, y=y2, dt=dt, traj_id=id_traj2)

   # gather in an ensemble
   trajs = [traj1, traj2]


At this point, it is quite easy using `yupi` to extract some time series from a 
single trajectory. For example:

- position: ``traj1.r``
- displacement: ``traj1.r.delta``
- velocity: ``traj1.v``
- speed: ``traj1.v.norm``
- distance: ``traj1.delta_r.norm``

But how to get these from an ensemble? How to extract it when time scales are different from the time step? In what is next, these questions are answered with some illustrative examples while a detailed explanation of every method parameter is given.


2.1 The ``concat`` parameter

If we want for instance the position vectors for all the set of trajectories, one can simply do:

.. code-block:: python

   from yupi.stats import collect
   collect(trajs, 'r')

.. code-block:: python

   array([[ 0.,  0.],
          [ 2.,  4.],
          [ 4., 16.],
          [ 6., 36.],
          [ 8., 64.],
          [ 1.,  1.],
          [ 3.,  9.],
          [ 5., 25.],
          [ 7., 49.],
          [ 9., 81.]])

In this case, the column vectors are the concatenated components of position vectors. If the data is wanted to be split by realizations, the ``concat`` parameter should be set to ``False``.

.. code-block:: python

   collect(trajs, 'r', concat=False)

.. code-block:: python

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

2.2 The ``key`` parameter

The user is able to manage the "collected" data by means of the ``key`` parameter. Position and velocity are requested by ``'r'`` and ``'v'``, respectively, which will be defined as the *main keys*. If the letter *d* is set before a main key (e.g., ``'dr'``, ``'dv'``), the ``delta`` property from the :py:class:`~yupi.Trajectory` class will be invoked to compute variations.

.. code-block:: python

   collect(trajs, key='dr')

.. code-block:: python

   array([[ 2.,  4.],
          [ 2., 12.],
          [ 2., 20.],
          [ 2., 28.],
          [ 2.,  8.],
          [ 2., 16.],
          [ 2., 24.],
          [ 2., 32.]])

If the letter *n* is set at the end of a key (e.g., ``'rn'``, ``'dvn'``), the absolute value of the requested vectors are returned.

.. code-block:: python

   collect(trajs, 'drn')

.. code-block:: python

   array([ 4.47213595, 12.16552506, 20.09975124, 28.0713377 ,  8.24621125,  16.1245155 , 24.08318916, 32.06243908])

Components can also be extracted from the collected vectors. It can be done in two ways: by specifying the *x*, *y* or *z* component (e.g., ``'rx'``, ``'dvy'``), or by explicitly appending to the key the position of the column vector (e.g., ``'r0'``, ``'dv1'``). The former is specially useful when dealing with multidimensional trajectories.

.. code-block:: python

   collect(trajs, 'dvy') == collect(trajs, 'dv1')

.. code-block:: python

   array([ True,  True,  True,  True,  True,  True])


2.3 The ``lag_step`` and ``lag_time`` parameters

Suppose the underlying ensemble of trajectories as being realizations of a process with different statistical properties at different time scales. For such a case, ``lag_step`` and ``lag_time`` can be helpful if they are set properly. If lag is an integer that account for number of samples, ``lag_step`` should be used. Instead, use ``lag_time`` if its units are those of the time array (i.e., ``traj.t``).

If none of this parameters are given, ``lag_step=1`` will be assumed.

.. code-block:: python

   collect(trajs, 'dr', lag_step=2)

.. code-block:: python

   array([[ 4., 16.],
          [ 4., 32.],
          [ 4., 48.],
          [ 4., 24.],
          [ 4., 40.],
          [ 4., 56.]])

.. code-block:: python

   collect(trajs, 'dr', lag_time=2*dt)

.. code-block:: python

   array([[ 4., 16.],
          [ 4., 32.],
          [ 4., 48.],
          [ 4., 24.],
          [ 4., 40.],
          [ 4., 56.]])

When ``key='r'`` and lag is not ``None``, position vectors will be sampled with a sample frequency given by the inverse of ``lag_step`` or ``lag_time``.

.. code-block:: python

   collect(trajs, 'r', lag_step=2)

.. code-block:: python

   array([[ 0.,  0.],
          [ 4., 16.],
          [ 8., 64.],
          [ 1.,  1.],
          [ 5., 25.],
          [ 9., 81.]])


2.4 The `warnings` parameter

If the given lag is larger than one of the trajectories length, a warning message will arise and the position of the trajectory in the ensemble and its *id* will be shown. The ``collect`` method will skip this trajectory. To avoid warning messages set the parameter to ``False``.

.. code-block:: python

   traj1.dt = .01  # redefining id for the first trajectory
   collect(trajs, 'dr', lag_time=dt)

.. code-block:: python

   15:07:11 [WARNING] Trajectory 0 with id=traj_01 is shorten than 50 samples
   array([[ 2.,  8.],
          [ 2., 16.],
          [ 2., 24.],
          [ 2., 32.]])

.. code-block:: python

   collect(trajs, 'dr', lag_time=dt, warnings=False)

.. code-block:: python

   array([[ 2.,  8.],
          [ 2., 16.],
          [ 2., 24.],
          [ 2., 32.]])