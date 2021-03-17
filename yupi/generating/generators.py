import numpy as np
import abc
from yupi import Trajectory

class Generator():

    """docstring for Generator"""

    def __init__(self, T:float, dim:int=1, N:int=1, dt:int=1):
        # siulation parameters
        self.T = T            # total time
        self.dim = dim        # trajectory dimensions
        self.N = N            # number of trajectories
        self.dt = dt          # time step of the simulation
        self.n = int(T / dt)  # number of time steps

    @abc.abstractmethod
    def generate(self):
        pass



class RandomWalkGenerator(Generator):
    """
    Multidimensional Random Walk class.
    """
    
    def __init__(self, T:float, dim:int=1, N:int=1, dt:int=1,
            actions:np.ndarray=None, 
            actions_prob:np.ndarray=None, 
            jump_len:np.ndarray=None):

        super().__init__(T, dim, N, dt) 

        # dynamic variables
        self.t = np.arange(self.n) * dt      # time array
        self.r = np.zeros((self.n, dim, N))  # position array

        # model parameters
        # only right/left jumps, uniform probabilities for it
        # and equal length for all jumps are set as default
        # TODO: Check that the model parameters received have the expected shape
        self.actions = np.array([1, -1]) if actions is None else actions
        self.actions_prob = np.tile([.5, .5], (dim, 1)) if actions_prob is None else actions_prob
        self.jump_len = np.ones((dim, N)) if jump_len is None else jump_len



    # compute vector position as a function of time for
    # all the walkers of the ensemble
    def get_r(self):
        # get movements for every space coordinates according 
        # to the sample space of probabilities in self.actions_prob
        dr = [np.random.choice(self.actions, p=p, size=(self.n - 1, self.N)) for p in self.actions_prob]
        
        # set time/coordinates as the first/second axis
        dr = np.swapaxes(dr, 0, 1)
        
        # scale displacements according to the jump length statistics
        dr = dr * self.jump_len

        # integrate displacements to get position vectors
        self.r[1:] = np.cumsum(dr, axis=0)
        return self.r


    def generate(self):
        # get RandomWalk object and get position vectors
        r = self.get_r()

        trajs  = []
        for i in range(self.N):
            x = r[:,0,i]
            y = r[:,1,i] if self.dim > 1 else None
            z = r[:,2,i] if self.dim > 2 else None
            trajs.append(Trajectory(x=x, y=y, z=z, dt=self.dt,
                                  id="Random Walker {}".format(i+1)))
        return trajs

class LangevinGenerator(Generator):
    """
    Random Walk class from a multidimensional Langevin Equation.
    """

    def __init__(self, T:float, dim:int=1, N:int=1, dt:int=1,
        tau:float=1.,
        noise_pdf:str='normal',
        noise_scale:float=1,
        v0:np.ndarray=None, r0:np.ndarray=None):
        
        super().__init__(T, dim, N, dt) 

        # model parameters
        self.tau = tau                  # relaxation characteristic time
        self.noise_pdf = noise_pdf      # noise PDF
        self.noise_scale = noise_scale  # scale parameter (not stan. dev.)
        self.noise = np.ndarray         # noise array that will be fill in get_noise method

        # dynamic variables
        self.shape = (self.n, dim, N)           # shape of the dynamic variables
        self.t = np.linspace(0, T, num=self.n)  # time array
        self.r = np.empty(self.shape)           # position array
        self.v = np.empty(self.shape)           # velocity array

        # initial conditions
        # TODO: Check that r0 have the rigth shape
        self.r[0] = np.zeros((dim, N)) if r0 is None else r0
        # TODO: Check that v0 have the rigth shape
        self.v[0] = np.zeros((dim, N)) if v0 is None else v0

        self.v_scale = 1
        self.r_scale = 1
        self.t_scale = 1

    def set_scale(self, v_scale=None, r_scale=None, t_scale=None):
        if v_scale:
            self.v_scale = v_scale
        if r_scale:
            self.r_scale = r_scale
        if t_scale:
            self.t_scale = t_scale

    # fill noise array with custom noise properties
    def get_noise(self):
        dist = getattr(np.random, self.noise_pdf)
        self.noise = dist(scale=self.noise_scale, size=self.shape)


    # solve Langevin Equation using the numerical method of Euler-Maruyama
    def solve_rv(self):
        for i in range(self.n - 1):
            # solving for position
            self.r[i + 1] = self.r[i] + \
                            self.v[i] * self.dt

            # solving for velocity
            self.v[i + 1] = self.v[i] + \
                            -np.dot(1 / self.tau, self.v[i]) * self.dt + \
                            self.noise[i] * np.sqrt(self.dt)

    # simulate the process
    def simulate(self):
        self.get_noise()  # create the attribute self.noise
        self.solve_rv()   # solve the Langevin equation

    # Generate yupi Trajectory objects
    def generate(self):
        self.simulate()

        self.r *= self.r_scale 
        self.v *= self.v_scale 
        self.t *= self.t_scale 
        self.dt *= self.t_scale 

        trajs  = []
        for i in range(self.N):
            x = self.r[:,0,i]
            y = self.r[:,1,i] if self.dim > 1 else None
            z = self.r[:,2,i] if self.dim > 2 else None
            trajs.append(Trajectory(x=x, y=y, z=z, dt=self.dt, t=self.t
                                  id="LangevinSolution {}".format(i+1)))
        return trajs



# testing
if __name__ == '__main__':
    from yupi.analyzing.visualization import plot_trajectories
    
    np.random.seed(0)

    print("Testing RandomWalker")
    
    # set parameter values
    T = 500
    dim = 2
    N = 5
    dt = 1
    actions = [1, 0, -1]
    prob = [[.5, .1, .4],
            [.5, 0, .5]]

    # get RandomWalk object and get position vectors
    rw = RandomWalkGenerator(T, dim, N, dt, actions, prob)
    tr = rw.generate()
    plot_trajectories(tr)