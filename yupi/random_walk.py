import numpy as np



class RandomWalk:
    """
    Multidimensional Random Walk class.
    """
    
    def __init__(self, n:int, dim:int=1, N:int=1, dt:float=1, 
            actions:np.ndarray=None, actions_prob:np.ndarray=None, 
            jump_len:np.ndarray=None):

        # simulation parameters
        self.n = n      # total number of steps
        self.dim = dim  # random walk dimension
        self.N = N      # number of walkers
        self.dt = dt    # time step
        
        # model parameters
        self.actions = actions            # available RW movements
        self.actions_prob = actions_prob  # probability for every action
        self.jump_len = jump_len          # statistic for jumps length

        # dynamic variables
        self.t = np.arange(n) * dt      # time array
        self.r = np.zeros((n, dim, N))  # position array

        # only right/left jumps, uniform probabilities for it
        # and equal length for all jumps are set as default
        if actions is None:
            self.actions = np.array([1, -1])
        if actions_prob is None:
            self.actions_prob = np.tile([.5, .5], (dim, 1))
        if jump_len is None:
            self.jump_len = np.ones((dim, N))


    # compute vector position as a function of time for
    # all the walkers of the ensemble
    def get_r(self):
        # get movements for every space coordinates according 
        # to the sample space of probabilities in self.actions_prob
        dr = [np.random.choice(actions, p=p, size=(self.n - 1, self.N)) for p in self.actions_prob]
        
        # set time/coordinates as the first/second axis
        dr = np.swapaxes(dr, 0, 1)
        
        # scale displacements according to the jump length statistics
        dr = dr * self.jump_len

        # integrate displacements to get position vectors
        self.r[1:] = np.cumsum(dr, axis=0)
        return self.r



# testing
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    np.random.seed(0)

    # set parameter values
    n = 500
    dim = 2
    N = 5
    dt = 1
    actions = [1, 0, -1]
    prob = [[.5, .1, .4],
            [.5, 0, .5]]

    # get RandomWalk object and get position vectors
    rw = RandomWalk(n, dim, N, dt, actions, prob)
    r = rw.get_r()
    x, y = r[:,0,:], r[:,1,:]

    # plotting
    plt.plot(x, y)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()