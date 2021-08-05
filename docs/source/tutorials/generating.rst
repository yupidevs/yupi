


Generating artificial Trajectory objects
----------------------------------------

If you want to generate :py:class:`~yupi.Trajectory` objects based on some statistical constrains, you can use a :py:class:`~generating.Generator` to construct a list of :py:class:`~yupi.Trajectory` objects:

.. code-block:: python

   from yupi.generating import RandomWalkGenerator

   # Set parameter values
   T = 500     # Total time (number of time steps if dt==1)
   dim = 2     # Dimension of the walker trajectories
   N = 3       # Number of random walkers
   dt = 1      # Time step

   # Probability of every action to be taken
   # according to every axis (the actions are [-1, 0, 1])
   prob = [[.5, .1, .4],   # x-axis
           [.5,  0, .5]]   # y-axis

   # Get RandomWalk object and get position vectors
   rw = RandomWalkGenerator(T, dim, N, dt, prob)
   tr = rw.generate()

