Operations on Trajectory objects
--------------------------------

Let us consider a simple 2D trajectory:

.. code-block:: python

    traj = Trajectory(points=[[1,2], [3,3], [4,2]])


There are several ways a trajectory can be modified in yupi. 

Shifting
========

If the trajectory needs to be shifted it can be done by
adding (or substracting) a tuple (or any array-like structure) with the same dimensions of the
trajectory points:

.. code-block:: python

    transf_traj = traj + (1, 4)   # transf_traj points: [[2,6], [4,7], [5,6]]
    centered = traj - traj.r[0]   # centered points: [[0,0], [2,1], [3,0]]

Both operations can be made in-place when using the operators: **+=** or
**-=**. 

Scaling
=======

Spacial scaling of a trajectory can be also achieved by multiplying it by a constant:

.. code-block:: python

    scaled_traj = traj * 3   # scaled_traj points: [[3,6], [9,9], [12,6]]


This operation can be made in-place when using the operator *=.


Rotation
========

Rotation can be made using the **rotate** method:

.. code-block:: python

    traj_2 = Trajectory(points=[[0,0], [1,0]])
    traj_2.rotate(-np.pi / 2)   # traj_2 points: [[0,0], [0,1]]


Adding and subtracting
======================

If two trajectories have the same length and dimensions they can be added or
subtracted by:


.. code-block:: python

    traj_a = Trajectory(points=[[1,2], [3,3], [4,2]])
    traj_b = Trajectory(points=[[0,0], [1,4], [2,3]])
    traj_c = traj_a + traj_b   # traj_c points: [[1,2], [4,7], [6,5]]

Indexing and slicing
====================

Trajectories can also be indexing and obtein the i-th :ref:`~yupi.TrajectoryPoint`:

.. code-block:: python

    traj = Trajectory(points=[[1,2], [3,3], [4,2]])
    p2 = traj[2]   # p2.r = [4,2]

Slicing is possible too and it is used to obtain a subtrajectory. All variance of sciling in python are possible:

.. code-block:: python

    traj = Trajectory(points=[[1,2], [3,3], [4,2], [4,1], [2,7]])
    sub_traj_1 = traj[2:]    # sub_traj_1.r = [[4,2], [4,1], [2,7]]
    sub_traj_2 = traj[:-1]   # sub_traj_2.r = [[1,2], [3,3], [4,2], [4,1]]
    sub_traj_3 = traj[2:4]   # sub_traj_3.r = [[4,2], [4,1]]
