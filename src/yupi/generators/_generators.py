import abc
from typing import Any, Callable, Collection

import numpy as np

from yupi._differentiation import DiffMethod, WindowType
from yupi.trajectory import Trajectory
from yupi.units import Units

InitialPosition = Collection[Collection[float]] | Collection[float] | float
InitialVelocity = Collection[Collection[float]] | Collection[float] | float


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
    units: Units | None
        Units of the Trajectory, by default None. If None, the units
        are set to the default ones (i.e., m/s).
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    seed : int, optional
        Seed for the random number generator. If None, no seed is set.
        By default None.

    Attributes
    ----------
    T : float
        Total duration of each Trajectory.
    dim : int, optional
        Dimension of each Trajectory, by default 1.
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    n  : int
        Number of samples on each Trajectory.
    rng : np.random.Generator
        Random number generator.
    """

    def __init__(
        self,
        T: float,
        units: Units | None = None,
        dim: int = 1,
        dt: float = 1.0,
        seed: int | None = None,
    ):
        self.T = T
        self.units = units
        self.dim = dim
        self.dt = dt
        self.n = int(T / dt)
        self.rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

    @abc.abstractmethod
    def generate(self, N: int) -> list[Trajectory]:
        """
        Abstract method that is implemented on inheriting classes.
        It should compute a list of ``N`` Trajectory objects with the
        given parameters using a method specific to the inheriting class.

        Parameters
        ----------
        N : int
            Number of Trajectory objects to be generated.
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
    units: Units | None
        Units of the Trajectory, by default None. If None, the units
        are set to the default ones (i.e., m/s).
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    actions_prob : np.ndarray, optional
        Probability of each action (i.e., decrease, stead or increase)
        to be taken, according to every axis. If this parameter is not
        passed the walker will assume uniform probability for each
        action, by default None.
    step_length_func : Callable[[tuple], np.ndarray], optional
        Function that returns the distribution of step lengths that
        will be taken by the walker on each time step, dimension and
        instance of a trajectory. Expected shape of the return value is
        (int(T/dt)-1, dim, N), by default np.ones.
    step_length_kwargs : dict, optional
        Key-word arguments of the ``step_length_func``, by default
        ``{}``.
    """

    def __init__(
        self,
        T: float,
        units: Units | None = None,
        dim: int = 1,
        dt: float = 1,
        actions_prob: np.ndarray | list[list[float]] | None = None,
        step_length_func: Callable[[tuple], np.ndarray] = np.ones,
        seed: int | None = None,
        **step_length_kwargs: Any,
    ) -> None:
        super().__init__(T=T, units=units, dim=dim, dt=dt, seed=seed)

        # Main id of generated trajectories
        self.traj_id = "RandomWalk"
        self.t: np.ndarray
        self.r: np.ndarray
        self.step_length: np.ndarray

        self.step_length_func = step_length_func
        self.step_length_kwargs = step_length_kwargs

        # Model parameters
        actions = np.array([-1, 0, 1])

        if actions_prob is None:
            actions_prob = np.tile([1 / 3, 1 / 3, 1 / 3], (dim, 1))

        actions_prob = np.asarray(actions_prob, dtype=np.float32)

        if actions_prob.shape[0] != dim or actions_prob.shape[1] != actions.shape[0]:
            raise ValueError("actions_prob must have shape like (dims, 3)")

        self.actions = actions
        self.actions_prob = actions_prob

    def _set_init_cond(self, N: int) -> None:
        self.t = np.arange(self.n) * self.dt  # Time array
        self.r = np.zeros((self.n, self.dim, N))  # Position array

        shape_tuple = (self.n - 1, self.dim, N)
        step_length = self.step_length_func(shape_tuple, **self.step_length_kwargs)
        self.step_length = step_length

    # Compute vector position as a function of time for
    # All the walkers of the ensemble
    def _get_r(self, N: int) -> np.ndarray:
        # Get displacement for every coordinates according
        # to the probabilities in self.actions_prob
        delta_r = np.array(
            [
                self.rng.choice(self.actions, p=p, size=(self.n - 1, N))
                for p in self.actions_prob
            ]
        )

        # Set time/coordinates as the first/second axis
        delta_r = np.swapaxes(delta_r, 0, 1)

        # Scale displacements according to the jump length statistics
        delta_r = delta_r * self.step_length

        # Integrate displacements to get position vectors
        self.r[1:] = np.cumsum(delta_r, axis=0)
        return self.r

    # Get position vectors and generate RandomWalk object
    def generate(self, N: int) -> list[Trajectory]:
        # Set parameters
        self._set_init_cond(N)

        # Get position vectors
        r = self._get_r(N)

        # Generate RandomWalk object
        trajs = []
        for i in range(N):
            points = r[:, :, i]
            trajs.append(
                Trajectory(
                    points=points,
                    dt=self.dt,
                    t=self.t,
                    units=self.units,
                    traj_id=f"{self.traj_id} {i + 1}",
                    diff_est={
                        "method": DiffMethod.LINEAR_DIFF,
                        "window_type": WindowType.FORWARD,
                    },
                )
            )
        return trajs


class _LangevinGenerator(Generator):
    def __init__(
        self,
        T: float,
        units: Units | None = None,
        dim: int = 1,
        dt: float = 1.0,
        gamma: float = 1.0,
        sigma: float = 1.0,
        v0: InitialPosition | None = None,
        r0: InitialVelocity | None = None,
        seed: int | None = None,
    ):
        super().__init__(T=T, units=units, dim=dim, dt=dt, seed=seed)

        # Main id of generated trajectories
        self.traj_id = "Langevin"

        # Model parameters
        self.gamma = gamma  # Relaxation time
        self.sigma = sigma  # Noise scale parameter

        # Initial conditions
        self.r0 = np.array(0 if r0 is None else r0)
        self.v0 = np.array(0 if v0 is None else v0)

    # Intrinsic reference parameters
    def _set_scaling_params(self) -> None:
        self.t_scale = self.gamma**-1  # Time scale
        self.v_scale = self.sigma * np.sqrt(self.t_scale)  # Speed scale
        self.r_scale = self.v_scale * self.t_scale  # Length scale

    # Set initial conditions
    def _set_init_cond(self, N: int) -> None:
        # Simulation parameters
        self.dt = self.dt / self.t_scale  # Dimensionless time step
        self.shape = (self.n, self.dim, N)  # Shape of dynamic variables

        # Dynamic variables
        self.t = np.arange(self.n) * self.dt  # Time array
        self.r = np.empty(self.shape)  # Position array
        self.v = np.empty(self.shape)  # Velocity array

        # Initial positions
        if np.shape(self.r0) == (self.dim,):
            self.r[0][:][:] = self.r0[:, None]
        elif np.shape(self.r0) == (self.dim, N) or np.ndim(self.r0) == 0:
            self.r[0] = self.r0  # User initial positions
        else:
            raise ValueError(
                "r0 is expected to be a float, or an "
                f"array of shape {(self.dim)}, or an "
                f"array of shape {(self.dim, N)}."
            )
        self.r[0] /= self.r_scale

        # Initial velocities
        if np.shape(self.v0) == (self.dim,):
            self.v[0][:][:] = self.v0[:, None]
        elif np.shape(self.v0) == (self.dim, N) or np.ndim(self.v0) == 0:
            self.v[0] = self.v0  # User input
        else:
            raise ValueError(
                "v0 is expected to be a float, or an "
                f"array of shape {(self.dim)}, or an "
                f"array of shape {(self.dim, N)}."
            )
        self.v[0] /= self.v_scale

    # Fill noise array with custom noise properties
    def _set_noise(self) -> None:
        self.noise = self.rng.normal(size=self.shape)

    # Solve dimensionless Langevin Equation using
    # the numerical method of Euler-Maruyama
    @abc.abstractmethod
    def _solve(self) -> None: ...

    # Scale by intrinsic reference quantities
    def _set_scale(self) -> None:
        self.r *= self.r_scale
        self.v *= self.v_scale
        self.t *= self.t_scale
        self.dt *= self.t_scale

    # Simulate the process
    def _simulate(self, N: int) -> None:
        # Init variables before simulate and validate initial conditions
        self._set_scaling_params()  # Set intrinsic reference parameters
        self._set_init_cond(N)  # Set initial conditions
        self._set_noise()  # Set the attribute self.noise

        self._solve()  # Solve the Langevin equation
        self._set_scale()  # Recovering dimensions

    # Generate yupi Trajectory objects
    def generate(self, N: int) -> list[Trajectory]:
        self._simulate(N)

        trajs = []
        for i in range(N):
            points = self.r[:, :, i]
            trajs.append(
                Trajectory(
                    points=points,
                    units=self.units,
                    dt=self.dt,
                    traj_id=f"{self.traj_id} {i + 1}",
                )
            )
        return trajs


class LangevinGenerator(_LangevinGenerator):
    """
    Random Walk class from a multidimensional Langevin Equation.
    Boundary conditions to model confined or semi-infinite processes
    are supported.

    Parameters
    ----------
    T : float
        Total duration of trajectories.
    dim : int, optional
        Trajectories dimension, by default 1.
    units: Units | None
        Units of the Trajectory, by default None. If None, the units
        are set to the default ones (i.e., m/s).
    dt : float, optional
        Time step, by default 1.0.
    gamma : float, optional
        Drag parameter or inverse of the persistence time, by default 1.
    sigma : float, optional
        Noise intensity (i.e., scale parameter of noise pdf), by default 1.
    bounds: np.ndarray | None
        Lower and upper reflecting boundaries that confine the trajectories.
        Must have shape (2, dim). The first row contains the lower
        boundaries and the second row contains the upper boundaries.
        If None is passed, trajectories are simulated in a free space.
        By default None.
    bounds_extent: np.ndarray | None
        Decay length of boundary forces, by default None.
    bounds_strength: np.ndarray | None
        Boundaries strength, by default None.
    v0 : Collection[float] | float | None
        Initial velocities, by default None.
        Could be a float, an array of shape (dim,) or an array of shape
        (dim, N).
    r0 : Collection[float] | float | None
        Initial positions, by default None.
        Could be a float, an array of shape (dim,) or an array of shape
        (dim, N).
    """

    def __init__(
        self,
        T: float,
        dim: int = 1,
        units: Units | None = None,
        dt: float = 1.0,
        gamma: float = 1.0,
        sigma: float = 1.0,
        bounds: np.ndarray | None = None,
        bounds_extent: np.ndarray | None = None,
        bounds_strength: np.ndarray | None = None,
        v0: InitialVelocity | None = None,
        r0: InitialPosition | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            T=T,
            dim=dim,
            units=units,
            dt=dt,
            gamma=gamma,
            sigma=sigma,
            v0=v0,
            r0=r0,
            seed=seed,
        )

        # Verify if there is any boundary
        self.bounds = bounds
        self.bounds_ext = bounds_extent
        self.bounds_stg = bounds_strength

    def _set_init_cond(self, N: int) -> None:
        super()._set_init_cond(N)
        # Set bounds and check initial positions
        self._set_bounds()

    # Broadcast and convert None into np.nan
    def _broadcast_bounds(self) -> None:
        ones = np.ones((2, self.dim))
        self.bounds = np.float32(self.bounds) * ones
        self.bounds_ext = np.float32(self.bounds_ext) * ones
        self.bounds_stg = np.float32(self.bounds_stg) * ones

    # Set dimensionless bounds properties
    def _dimless_bounds(self) -> None:
        self.bounds = self.bounds / self.r_scale
        self.bounds_ext = self.bounds_ext / self.r_scale
        self.bounds_stg = self.bounds_stg / (self.r_scale / self.t_scale**2)

    # Check if all initial positions are whithin boundaries
    def _check_r0(self) -> None:
        # Unpack lower and upper bounds
        assert self.bounds is not None
        lower_bound, upper_bound = self.bounds

        # Find axes without boundaries
        idx_lb = np.where(np.isnan(lower_bound))
        idx_ub = np.where(np.isnan(upper_bound))

        # Ignore position components when no boundaries are specified
        r_lb = np.delete(self.r[0], idx_lb, axis=0)
        r_ub = np.delete(self.r[0], idx_ub, axis=0)

        # Same for bounds
        lower_bound = np.delete(lower_bound, idx_lb)
        upper_bound = np.delete(upper_bound, idx_ub)

        # Check if all positions are within both type of boundaries
        is_above_lb = np.all(lower_bound[:, None] <= r_lb)
        is_bellow_ub = np.all(upper_bound[:, None] >= r_ub)

        if not is_above_lb:
            raise ValueError("Initial positions must be above lower bounds.")

        if not is_bellow_ub:
            raise ValueError("Initial positions must be bellow upper bounds.")

    # Set bounds and check initial positions
    # TODO: check that `bounds` are compatibles with `dim`
    def _set_bounds(self) -> None:
        # Broadcast and convert None into np.nan
        self._broadcast_bounds()
        assert self.bounds is not None

        # Check if there is at least one bound
        self.has_bounds = not np.all(np.isnan(self.bounds))

        if self.has_bounds:
            self._dimless_bounds()
            self._check_r0()

    # Get net force from the boundaries
    def _bound_force(self, r: np.ndarray, tolerance: float = 10) -> np.ndarray | float:
        # Return zero force if there is no bounds
        if not self.has_bounds:
            return 0.0

        # Set r to have shape = (N, dim)
        r = r.T

        assert self.bounds is not None
        assert self.bounds_ext is not None
        assert self.bounds_stg is not None

        # Lower and upper bound limits, extents and strengths
        lower_bound, upper_bound = self.bounds
        ext_lb, ext_ub = self.bounds_ext
        stg_lb, stg_ub = self.bounds_stg

        # Get distance from the bounds and scale
        # by the bound extent parameter
        dr_lb = (r - lower_bound) / ext_lb
        dr_ub = (r - upper_bound) / ext_ub

        # An exponential models the force from the wall.
        # Get zero force if there is no bound or the particle
        # is far enough.
        force_lb = np.where(
            np.isnan(lower_bound) | (dr_lb > tolerance), 0.0, stg_lb * np.exp(-dr_lb)
        )

        force_ub = np.where(
            np.isnan(upper_bound) | (-dr_ub > tolerance), 0.0, -stg_ub * np.exp(dr_ub)
        )

        # Adding boundary effects and transpose to recover
        # shape as (dim, N)
        bound_force = (force_lb + force_ub).T
        return bound_force

    # Solve dimensionless Langevin Equation using
    # the numerical method of Euler-Maruyama
    def _solve(self) -> None:
        sqrt_dt = np.sqrt(self.dt)
        for i in range(self.n - 1):
            # Solving for position
            self.r[i + 1] = self.r[i] + self.v[i] * self.dt

            # Solving for velocity
            self.v[i + 1] = (
                self.v[i]
                - self.v[i] * self.dt
                + self.noise[i] * sqrt_dt
                + self._bound_force(self.r[i]) * self.dt
            )


class _DiffDiffGenerator(Generator):
    def __init__(
        self,
        T: float,
        dim: int = 1,
        units: Units | None = None,
        dt: float = 1.0,
        tau: float = 1.0,
        sigma: float = 1.0,
        dim_aux: int = 1,
        r0: InitialPosition | None = None,
        seed: int | None = None,
    ):
        super().__init__(T=T, dim=dim, units=units, dt=dt, seed=seed)

        # Main id of generated trajectories
        self.traj_id = "DiffDiff"

        # Model parameters
        self.tau = tau  # Relaxation time
        self.sigma = sigma  # Noise scale parameter of auxiliary variable

        # Intrinsic reference parameters
        self.t_scale = tau  # Time scale
        self.r_scale = sigma * self.t_scale  # Length scale

        # Simulation parameters
        self.dt = dt / self.t_scale  # Dimensionless time step
        self.shape: tuple  # Shape of dynamic variables
        self.dim_aux = dim_aux  # Dimension of the aux variable

        # Dynamic variables
        self.t = np.arange(self.n, dtype=np.float32)  # Time array
        self.r: np.ndarray  # Position array
        self.aux_var: np.ndarray  # Square of diffusivity
        self.noise_r: np.ndarray  # Noise for position (filled in _set_noise method)
        self.noise_Y: np.ndarray  # Aux variable (filled in _set_noise method)

        # Initial conditions
        self.r0 = np.array(0 if r0 is None else r0)

    # Set initial conditions
    def _set_init_cond(self, N: int) -> None:
        self.shape = (self.n, self.dim, N)  # Shape of dynamic variables

        self.r = np.empty(self.shape)  # Position array

        self.aux_var = self.rng.normal(
            size=(self.dim_aux, N)
        )  # Initial aux variable configuration
        self.D = np.sum(self.aux_var**2, axis=0)  # Initial diffusivity configuration

        if np.shape(self.r0) == (self.dim,):
            self.r[0][:][:] = self.r0[:, None]
        elif np.shape(self.r0) == (self.dim, N) or np.ndim(self.r0) == 0:
            self.r[0] = self.r0  # User initial positions
        else:
            raise ValueError(
                "r0 is expected to be a float, an "
                f"array of shape {(self.dim)}, or an "
                f"array of shape {(self.dim, N)}."
            )

    # Fill noise arrays
    def _set_noise(self, N: int) -> None:
        dist = self.rng.normal
        self.noise_r = dist(size=self.shape)
        self.noise_Y = dist(size=(self.n, self.dim_aux, N))

    @abc.abstractmethod
    def _solve(self) -> None: ...

    # Scale by intrinsic reference quantities
    def _set_scale(self) -> None:
        self.r *= self.r_scale
        self.t *= self.t_scale
        self.dt *= self.t_scale

    # Simulate the process
    def _simulate(self, N: int) -> None:
        self._set_noise(N)  # Set the attribute self.noise
        self._solve()  # Solve the Langevin equation
        self._set_scale()  # Scaling

    # Generate yupi Trajectory objects
    def generate(self, N: int) -> list[Trajectory]:
        self._set_init_cond(N)  # Check and set initial conditions
        self._simulate(N)

        trajs = []
        for i in range(N):
            points = self.r[:, :, i]
            trajs.append(
                Trajectory(
                    points=points,
                    units=self.units,
                    dt=self.dt,
                    traj_id=f"{self.traj_id} {i + 1}",
                )
            )
        return trajs


class DiffDiffGenerator(_DiffDiffGenerator):
    """
        Random Walk class for the Diffusing Diffusivity model. Boundary
        conditions to model confined or semi-infinite processes are supported.

    Parameters
    ----------
    T : float
        Total duration of each Trajectory.
    dim : int, optional
        Dimension of each Trajectory, by default 1.
    units: Units | None
        Units of the Trajectory, by default None. If None, the units
        are set to the default ones (i.e., m/s).
    dt : float, optional
        Time step of the Trajectory, by default 1.0.
    tau : float, optional
        Relaxation characteristic time of the auxiliary variable, by default 1.
    sigma : float, optional
        Scale parameter of the auxiliary variable noise, by default 1.
    dim_aux: int, optional
        Dimension of the auxiliary process, which is the square of
        the diffusivity, by default 1.
    bounds: np.ndarray | None
        Lower and upper reflecting boundaries that confine the trajectories.
        Must have shape (2, dim). The first row contains the lower
        boundaries and the second row contains the upper boundaries.
        If None is passed, trajectories are simulated in a free space.
        By default None.
    bounds_extent: np.ndarray | None
        Decay length of boundary forces, by default None.
    bounds_strength: np.ndarray | None
        Boundaries strength, by default None.
    r0 :
        Initial positions, by default None.
    """

    def __init__(
        self,
        T: float,
        dim: int = 1,
        units: Units | None = None,
        dt: float = 1.0,
        tau: float = 1.0,
        sigma: float = 1.0,
        dim_aux: int = 1,
        bounds: np.ndarray | None = None,
        bounds_extent: np.ndarray | None = None,
        bounds_strength: np.ndarray | None = None,
        r0: InitialPosition | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            T=T,
            dim=dim,
            units=units,
            dt=dt,
            tau=tau,
            sigma=sigma,
            dim_aux=dim_aux,
            r0=r0,
            seed=seed,
        )

        # Verify if there is any boundary
        self.bounds = (
            np.full((2, dim), np.nan, dtype=np.float32) if bounds is None else bounds
        )
        self.has_bounds = not np.all(np.isnan(self.bounds))  # Check for all bounds
        self.bounds_extent = bounds_extent
        self.bounds_strength = bounds_strength

    def _set_init_cond(self, N: int) -> None:
        super()._set_init_cond(N)
        if self.has_bounds:
            # Broadcast and scale bounds properties
            ones = np.ones((2, self.dim))
            self.bounds = self.bounds * ones / self.r_scale
            self.bounds_ext = np.float32(self.bounds_extent) * ones / self.r_scale
            self.bounds_stg = (
                np.float32(self.bounds_strength)
                * ones
                * (self.t_scale**2 / self.r_scale)
            )

            # Check is initial positions are within bounds
            self._check_r0()

    # Check if all initial positions are inside boundaries
    def _check_r0(self) -> None:
        # Unpack lower and upper bounds
        assert self.bounds is not None
        lower_bound, upper_bound = self.bounds

        # Find axes without boundaries
        idx_lb = np.where(np.isnan(lower_bound))
        idx_ub = np.where(np.isnan(upper_bound))

        # Ignore position components when no boundaries are specified
        r_lb = np.delete(self.r[0], idx_lb, axis=0)
        r_ub = np.delete(self.r[0], idx_ub, axis=0)

        # Same for bounds
        lower_bound = np.delete(lower_bound, idx_lb)
        upper_bound = np.delete(upper_bound, idx_ub)

        # Check if all positions are within both type of boundaries
        is_above_lb = np.all(lower_bound[:, None] <= r_lb)
        is_bellow_ub = np.all(upper_bound[:, None] >= r_ub)

        if not is_above_lb:
            raise ValueError("Initial positions must be above lower bounds.")

        if not is_bellow_ub:
            raise ValueError("Initial positions must be bellow upper bounds.")

    # Get net force from the boundaries
    def _bound_force(self, r: np.ndarray, tolerance: float = 10) -> np.ndarray | float:
        # Return zero force if there is no bounds
        if not self.has_bounds:
            return 0.0

        # Set r to have shape = (N, dim)
        r = r.T

        # Lower and upper bound limits, extents and strengths
        assert self.bounds is not None
        lower_bound, upper_bound = self.bounds
        ext_lb, ext_ub = self.bounds_ext
        stg_lb, stg_ub = self.bounds_stg

        # Get distance from the bounds and scale
        # by the bound extent parameter
        dr_lb = (r - lower_bound) / ext_lb
        dr_ub = (r - upper_bound) / ext_ub

        # An exponential models the force from the wall.
        # Get zero force if there is no bound or the particle
        # is far enough.
        force_lb = np.where(
            np.isnan(lower_bound) | (dr_lb > tolerance), 0.0, stg_lb * np.exp(-dr_lb)
        )

        force_ub = np.where(
            np.isnan(upper_bound) | (-dr_ub > tolerance), 0.0, -stg_ub * np.exp(dr_ub)
        )

        # Adding boundary effects and transpose to recover
        # shape as (dim, N)
        bound_force = (force_lb + force_ub).T
        return bound_force

    # Solve dimensionless coupled Langevin equations
    def _solve(self) -> None:
        sqrt_dt = np.sqrt(self.dt)
        for i in range(self.n - 1):
            # Solving for position
            self.r[i + 1] = (
                self.r[i]
                + np.sqrt(2 * self.D * self.dt) * self.noise_r[i]
                + self._bound_force(self.r[i]) * self.dt
            )

            # Solving for auxiliary variable
            self.aux_var += -self.aux_var * self.dt + self.noise_Y[i] * sqrt_dt

            # Updating the diffusivities
            self.D = np.sum(self.aux_var**2, axis=0)
