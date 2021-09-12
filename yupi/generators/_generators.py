import abc
from typing import Callable, Tuple
import numpy as np
from yupi import Trajectory


class Generator(metaclass=abc.ABCMeta):
    """
    Abstract class to model a Trajectory Generator. Classes inheriting
    from this class should implement ``generate`` method.

    Parameters
    ----------
    T : float
        Total duration of each Trajectory.
    dim : int, optional
        Dimension of each Trajectory, by default 1.
    N : int, optional
        Number of trajectories, by default 1.
    dt : float, optional
        Time step of the Trajectory, by default 1.0.

    Attributes
    ----------
    T : float
        Total duration of each Trajectory.
    dim : int, optional
        Dimension of each Trajectory, by default 1.
    N : int, optional
        Number of trajectories, by default 1.
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    n  : int
        Number of samples on each Trajectory.
    """

    def __init__(self, T: float, dim: int = 1, N: int = 1, dt: float = 1.0):
        # Siulation parameters
        self.T = T            # Total time
        self.dim = dim        # Trajectory dimension
        self.N = N            # Number of trajectories
        self.dt = dt          # Time step of the simulation
        self.n = int(T / dt)  # Number of time steps

    @abc.abstractmethod
    def generate(self):
        """
        Abstract method that is implemented on inheriting classes.
        It should compute a list of ``N`` Trajectory objects with the
        given parameters using a method specific to the inheriting class.
        """


class RandomWalkGenerator(Generator):
    """
    Multidimensional Random Walk Generator.

    Parameters
    ----------
    T : float
        Total duration of each Trajectory.
    dim : int, optional
        Dimension of each Trajectory, by default 1.
    N : int, optional
        Number of trajectories, by default 1.
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    actions_prob : np.ndarray, optional
        Probability of each action (i.e., decrease, stead or increase)
        to be taken, according to every axis. If this parameter is not
        passed the walker will assume uniform probability for each
        action, by default None.
    step_length_func : Callable[[Tuple], np.ndarray], optional
        Function that returns the distribution of step lengths that
        will be taken by the walker on each timestep, dimension and
        instance of a trajectory. Expected shape of the return value is
        (int(T/dt)-1, dim, N), by default np.ones.
    step_length_kwargs : dict, optional
        Key-word arguments of the ``step_length_func``, by default
        ``{}``.
    """

    def __init__(self, T: float, dim: int = 1, N: int = 1, dt: float = 1,
                 actions_prob: np.ndarray = None,
                 step_length_func: Callable[[Tuple], np.ndarray] = np.ones,
                 **step_length_kwargs):

        super().__init__(T, dim, N, dt)

        # Dynamic variables
        self.t = np.arange(self.n) * dt      # Time array
        self.r = np.zeros((self.n, dim, N))  # Position array

        # Model parameters
        actions = np.array([-1,0,1])

        if actions_prob is None:
            actions_prob = np.tile([1/3, 1/3, 1/3], (dim, 1))

        actions_prob = np.asarray(actions_prob, dtype=np.float32)

        if actions_prob.shape[0] != dim:
            raise ValueError('actions_prob must have shape like (dims, 3)')
        if actions_prob.shape[1] != actions.shape[0]:
            raise ValueError('actions_prob must have shape like (dims, 3)')

        shape_tuple = (self.n - 1, dim, N)
        step_length = step_length_func(shape_tuple, **step_length_kwargs)

        self.actions = actions
        self.actions_prob = actions_prob
        self.step_length = step_length

    # Compute vector position as a function of time for
    # All the walkers of the ensemble
    def _get_r(self):
        # Get movements for every space coordinates according
        # To the sample space of probabilities in self.actions_prob
        dr = [np.random.choice(self.actions, p=p, size=(self.n - 1, self.N))
              for p in self.actions_prob]

        # Set time/coordinates as the first/second axis
        dr = np.swapaxes(dr, 0, 1)

        # Scale displacements according to the jump length statistics
        dr = dr * self.step_length

        # Integrate displacements to get position vectors
        self.r[1:] = np.cumsum(dr, axis=0)
        return self.r

    # Get position vectors and generate RandomWalk object
    def generate(self):
        # Get position vectors
        r = self._get_r()

        # Generate RandomWalk object
        trajs = []
        for i in range(self.N):
            points = r[:, :, i]
            trajs.append(Trajectory(points=points, dt=self.dt, t=self.t,
                                    traj_id=f"Random Walker {i + 1}"))
        return trajs


class LangevinGenerator(Generator):
    """
    Random Walk class from a multidimensional Langevin Equation.

    Parameters
    ----------
    T : float
        Total duration of each Trajectory.
    dim : int, optional
        Dimension of each Trajectory, by default 1.
    N : int, optional
        Number of trajectories, by default 1.
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    tau : float, optional
        Relaxation characteristic time, by default 1.
    noise_scale : float, optional
        Scale parameter of the noise, by default 1.
    v0 : np.ndarray, optional
        Initial velocities, by default None.
    r0 : np.ndarray, optional
        Initial positions, by default None.
    """

    def __init__(self, 
                 T: float, 
                 dim: int = 1, 
                 N: int = 1, 
                 dt: float = 1., 
                 tau: float = 1., 
                 noise_scale: float = 1., 
                 v0: np.ndarray = None, 
                 r0: np.ndarray = None):

        super().__init__(T, dim, N, dt)

        # Main id of generated trajectories
        self.traj_id = 'Langevin'

        # Model parameters
        self.tau = tau                  # Relaxation time
        self.noise_scale = noise_scale  # Noise scale parameter

        # Intrinsic reference parameters
        self.t_scale = tau                                  # Time scale
        self.v_scale = noise_scale * np.sqrt(self.t_scale)  # Speed scale
        self.r_scale = self.v_scale * self.t_scale          # Length scale

        # Simulation parameters
        self.dt = dt / self.t_scale    # Dimensionless time step
        self.shape = (self.n, dim, N)  # Shape of dynamic variables

        # Dynamic variables
        self.t = np.arange(self.n) * self.dt  # Time array
        self.r = np.empty(self.shape)         # Position array
        self.v = np.empty(self.shape)         # Velocity array
        self.noise = None                     # Noise array (filled in _get_noise method)

        # Initial conditions
        self.r0 = r0           # Initial position
        self.v0 = v0           # Initial velocity
        self._set_init_cond()  # Check and set initial conditions


    # Set initial conditions
    def _set_init_cond(self):
        if self.r0 is None:
            self.r[0] = np.zeros((self.dim, self.N))  # Default initial positions
        elif np.shape(self.r0) == (self.dim, self.N) or np.shape(self.r0) == ():
            self.r[0] = self.r0                       # User initial positions
        else:
            raise ValueError('r0 is expected to be a float or an '
                            f'array of shape {(self.dim, self.N)}.')
        
        if self.v0 is None:
            self.v[0] = np.random.normal(size=(self.dim, self.N))  # Default initial velocities
        elif np.shape(self.v0) == (self.dim, self.N) or np.shape(self.v0) == ():
            self.v[0] = self.v0                                    # User initial velocities
        else:
            raise ValueError('v0 is expected to be a float or an '
                            f'array of shape {(self.dim, self.N)}.')


    # Fill noise array with custom noise properties
    def _get_noise(self):
        self.noise = np.random.normal(size=self.shape)


    # Solve dimensionless Langevin Equation using 
    # the numerical method of Euler-Maruyama
    def _solve(self):
        for i in range(self.n - 1):
            # Solving for position
            self.r[i + 1] = self.r[i] + \
                            self.v[i] * self.dt

            # Solving for velocity
            self.v[i + 1] = self.v[i] + \
                            -self.v[i] * self.dt + \
                            np.sqrt(self.dt) * self.noise[i]


    # Scale by intrinsic reference quantities
    def _set_scale(self):
        self.r *= self.r_scale
        self.v *= self.v_scale
        self.t *= self.t_scale
        self.dt *= self.t_scale


    # Simulate the process
    def _simulate(self):
        self._get_noise()  # Set the attribute self.noise
        self._solve()      # Solve the Langevin equation
        self._set_scale()  # Scaling


    # Generate yupi Trajectory objects
    def generate(self):
        self._simulate()

        trajs = []
        for i in range(self.N):
            points = self.r[:, :, i]
            trajs.append(Trajectory(points=points, dt=self.dt,
                                    traj_id=f"{self.traj_id} {i + 1}"))
        return trajs


class DiffDiffGenerator(Generator):
    """
        Random Walk class for the Diffusing Diffusivity model.

    Parameters
    ----------
    T : float
        Total duration of each Trajectory.
    dim : int, optional
        Dimension of each Trajectory, by default 1.
    N : int, optional
        Number of trajectories, by default 1.
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    tau : float, optional
        Relaxation characteristic time of the auxiliar variable, by default 1.
    noise_scale : float, optional
        Scale parameter of the auxiliar variable noise, by default 1.
    dim_aux: int, optional
        Dimension of the auxiliar process, which is the square of the diffusivity, by default 1.
    r0 : np.ndarray, optional
        Initial positions, by default None.
    """

    def __init__(self, 
                 T: float, 
                 dim: int = 1, 
                 N: int = 1, 
                 dt: float = 1., 
                 tau: float = 1., 
                 noise_scale: float = 1., 
                 dim_aux: int = 1, 
                 r0: np.ndarray = None):

        super().__init__(T, dim, N, dt)

        # Model parameters
        self.tau = tau                  # Relaxation time
        self.noise_scale = noise_scale  # Noise scale parameter of auxiliar variable

        # Intrinsic reference parameters
        self.t_scale = tau                         # Time scale
        self.r_scale = noise_scale * self.t_scale  # Length scale

        # Simulation parameters
        self.dt = dt / self.t_scale    # Dimensionless time step
        self.shape = (self.n, dim, N)  # Shape of dynamic variables
        self.dim_aux = dim_aux         # Dimension of the aux variable

        # Dynamic variables
        self.t = np.arange(self.n, dtype=np.float32)  # Time array
        self.r = np.empty(self.shape)                 # Position array
        self.Y = np.empty((dim_aux, N))               # Aux variable: square of diffusivity
        self.noise_r = None                           # Noise array for position (filled in _get_noise method)
        self.noise_Y = None                           # Noise array for aux variable (filled in _get_noise method)

        # Initial conditions
        self.Y = np.random.normal(size=(dim_aux, N))  # Initial aux variable configuration
        self.D = np.sum(self.Y**2, axis=0)            # Initial diffusivity configuration

        if r0 is None:
            self.r[0] = np.zeros((dim, N))                    # Default intial positions
        elif np.shape(r0) == (dim, N) or np.shape(r0) == ():
            self.r[0] = r0                                    # User intial positions
        else:
            raise ValueError(f'r0 is expected to be a float or an array of shape {(self.dim, self.N)}.')


    # Fill noise arrays
    def _get_noise(self):
        dist = np.random.normal
        self.noise_r = dist(size=self.shape)
        self.noise_Y = dist(size=(self.n, self.dim_aux, self.N))
    

    # Solve coupled Langevin equations
    def _solve(self):
        for i in range(self.n - 1):
            # Solving for position
            self.r[i + 1] = self.r[i] + \
                            np.sqrt(2 * self.D * self.dt) * self.noise_r[i]

            # Solving for auxiliar variable
            self.Y = self.Y + \
                     -self.Y * self.dt + \
                     np.sqrt(self.dt) * self.noise_Y[i]
            
            # Updating the diffusivities
            self.D = np.sum(self.Y**2, axis=0)


    # Scale by intrinsic reference quantities
    def _set_scale(self):
        self.r *= self.r_scale
        self.t *= self.t_scale
        self.dt *= self.t_scale


    # Simulate the process
    def _simulate(self):
        self._get_noise()  # Set the attribute self.noise
        self._solve()      # Solve the Langevin equation
        self._set_scale()  # Scaling


    # Generate yupi Trajectory objects
    def generate(self):
        self._simulate()

        trajs = []
        for i in range(self.N):
            points = self.r[:, :, i]
            trajs.append(Trajectory(points=points, dt=self.dt,
                                    traj_id=f"DiffDiff {i + 1}"))
        return trajs