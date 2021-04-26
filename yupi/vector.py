import numpy as np
from numpy.linalg.linalg import norm as nrm

class Vector(np.ndarray):

    @property
    def norm(self):
        return Vector.create([nrm(p) for p in self])

    @property
    def delta(self):
        if len(self.shape) > 1:
            new_vec = []
            for i in range(self.shape[1]):
                new_vec.append(np.ediff1d(self[:,i]))
            return Vector.create(new_vec).T
        else:
            return np.ediff1d(self)

    @property
    def x(self):
        return self.component(0)

    @property
    def y(self):
        return self.component(1)

    @property
    def z(self):
        return self.component(2)

    def component(self, dim):
        if len(self.shape) < 2:
            raise TypeError('Operation not supperted on simple vectors')
        if not isinstance(dim, int):
            raise TypeError("Parameter 'dim' must be an integer")
        if self.shape[1] < dim + 1:
            raise ValueError(f'Vector has not component {dim}')
        return self[:,dim]

    @staticmethod
    def create(*args, **kwargs):
        arr = np.array(*args, **kwargs)
        return arr.view(Vector)
