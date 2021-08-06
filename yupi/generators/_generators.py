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
    noise_pdf : str, optional
        Statistical model for the noise. ``noise_pdf`` should be a
        distribution from ``np.random``. By default 'normal'.
    noise_scale : float, optional
        Scale parameter of the noise, by default 1.
    v0 : np.ndarray, optional
        Initial velocities, by default None.
    r0 : np.ndarray, optional
        Initial positions, by default None.
    """

    def __init__(self, T: float, dim: int = 1, N: int = 1, dt: float = 1,
                 tau: float = 1.,
                 noise_pdf: str = 'normal',
                 noise_scale: float = 1,
                 v0: np.ndarray = None, r0: np.ndarray = None):

        super().__init__(T, dim, N, dt)

        # Model parameters

        # Relaxation time
        self.tau = tau
        # Noise PDF
        self.noise_pdf = noise_pdf
        # Scale parameter
        self.noise_scale = noise_scale
        # Noise array that will be filled in get_noise method
        self.noise = None

        # Dynamic variables

        # Shape of the dynamic variables
        self.shape = (self.n, dim, N)
        # Time array
        self.t = np.linspace(0, T, num=self.n, endpoint=False)
        # Position array
        self.r = np.empty(self.shape)
        # Velocity array
        self.v = np.empty(self.shape)

        # Initial conditions
        # TODO: Check that r0 have the rigth shape
        self.r[0] = np.zeros((dim, N)) if r0 is None else r0
        # TODO: Check that v0 have the rigth shape
        self.v[0] = np.zeros((dim, N)) if v0 is None else v0

        self.v_scale = 1
        self.r_scale = 1
        self.t_scale = 1

    # Set intrinsic reference parameters
    # TODO: Check if scales are compatibles
    def set_scale(self, v_scale=None, r_scale=None, t_scale=None):
        if v_scale:
            self.v_scale = v_scale
        if r_scale:
            self.r_scale = r_scale
        if t_scale:
            self.t_scale = t_scale

    # Fill noise array with custom noise properties
    def _get_noise(self):
        dist = getattr(np.random, self.noise_pdf)
        self.noise = dist(scale=self.noise_scale, size=self.shape)

    # Solve Langevin Equation using the numerical method of Euler-Maruyama
    def _solve_rv(self):
        for i in range(self.n - 1):
            # Solving for position
            self.r[i + 1] = self.r[i] + \
                            self.v[i] * self.dt

            # Solving for velocity
            self.v[i + 1] = self.v[i] + \
                            -np.dot(1 / self.tau, self.v[i]) * self.dt + \
                            self.noise[i] * np.sqrt(self.dt)

    # Simulate the process
    def _simulate(self):
        self._get_noise()  # Set the attribute self.noise
        self._solve_rv()   # Solve the Langevin equation

    # Generate yupi Trajectory objects
    def generate(self):
        self._simulate()

        self.r *= self.r_scale
        self.v *= self.v_scale
        self.t *= self.t_scale
        self.dt *= self.t_scale

        trajs = []
        for i in range(self.N):
            points = self.r[:, :, i]
            trajs.append(Trajectory(points=points, dt=self.dt, t=self.t,
                                    traj_id=f"LangevinSolution {i + 1}"))
        return trajs
