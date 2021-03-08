import numpy as np
from yupi import Trajectory


class LE:
	"""
	Random Walk class from a multidimensional Langevin Equation.
	"""

	def __init__(self, T:float, tau:float, noise_params:dict, 
				dim:int=1, N:int=1, dt:int=1):

		# siulation parameters
		self.T = T                    # total time
		self.dt = dt                  # time step of the simulation
		self.N = N                    # number of trials
		self.n = int(T / dt)          # number of time steps
		self.size = (self.n, dim, N)  # shape of the dynamic variables

		# model parameters
		self.tau = tau                    # relaxation characteristic time
		self.noise_params = noise_params  # noise properties (keys: 'pdf_name', 'scale')
		self.noise = np.ndarray           # noise array that will be fill in get_noise method

		# dynamic variables
		self.t = np.linspace(0, T, num=self.n)  # time array
		self.r = np.empty(self.size)            # position array
		self.v = np.empty(self.size)            # velocity array

		# initial conditions
		self.r0 = np.zeros((dim, self.N))
		self.v0 = np.zeros((dim, self.N))
		self._r0_ = False                  # True if initial positions are set by the user
		self._v0_ = False                  # True if initial velocities are set by the user


	# set initial condition for position vectors
	def set_r_init_cond(self, r0=None):
		if r0 is None:
			self.r[0] = self.r0  # set as default
		else:
			self.r[0] = r0       # set by the user

		self._r0_ = True


	# set initial condition for velocity vectors
	def set_v_init_cond(self, v0=None):
		self.v[0] = self.v0 if v0 is None else v0 # set as default
		self._v0_ = True


	# fill noise array with custom noise properties
	def get_noise(self):
		pdf_name = self.noise_params['pdf_name']  # noise PDF
		scale = self.noise_params['scale']        # scale parameter (not stan. dev.)

		dist = getattr(np.random, pdf_name)
		noise = dist(scale=scale, size=self.size)

		self.noise = noise


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
		# set initial conditions
		if not self._r0_:
			self.set_r_init_cond()
		if not self._v0_:
			self.set_v_init_cond()

		self.get_noise()  # create the attribute self.noise
		self.solve_rv()   # solve the Langevin equation

		x, y = self.r[:,0,:], self.r[:,1,:]
		# return self.r, self.v
		return Trajectory(x_arr=x, y_arr=y, dt=self.dt,
                                  id="LangevinSolution")



# testing
if __name__ == '__main__':
	from yupi.analyzing.visualization import plot_trajectories
	
	np.random.seed(0)

	# set parameter values
	dim = 2
	T = 5 * 60
	dt = .05
	N = 5
	tau = 1.	

	kTm = 10
	scale = np.sqrt(2 * kTm / tau)
	noise_params = {'pdf_name': 'normal', 'scale': scale}

	# get LE object and get position vectors
	le = LE(T, tau, noise_params, dim, N, dt)
	le.set_v_init_cond(scale * np.random.randn(le.N))

	tr = le.simulate()
	plot_trajectories([tr])
