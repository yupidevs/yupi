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


