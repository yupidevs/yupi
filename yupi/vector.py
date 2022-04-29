from __future__ import annotations

import numpy as np
from typing import Union
from numpy.linalg.linalg import norm as nrm


class Vector(np.ndarray):
    """Represents a vector"""

    @property
    def norm(self) -> Union[Vector, float]:
        """Vector : Calculates the norm of the vector. If the vector
        is alist of vectors then the norm of each item is calculated"""
        if len(self.shape) < 2:
            return float(nrm(self))
        return Vector.create([nrm(p) for p in self])

    @property
    def delta(self) -> Vector:
        """Vector : Calculates the differnece between each item"""
        if len(self.shape) > 1:
            new_vec = []
            for i in range(self.shape[1]):
                new_vec.append(np.ediff1d(self[:, i]))
            return Vector.create(new_vec).T
        return Vector.create(np.ediff1d(self))

    @property
    def x(self) -> Vector:
        """Vector : X component of all vector items"""
        return self.component(0)

    @property
    def y(self) -> Vector:
        """Vector : Y component of all vector items"""
        return self.component(1)

    @property
    def z(self) -> Vector:
        """Vector : Z component of all vector items"""
        return self.component(2)

    def component(self, dim) -> Vector:
        """
        Extract a given component from all vector items.

        Parameters
        ----------
        dim : int
            Component index.

        Returns
        -------
        Vector
            Component extracted.

        Raises
        ------
        TypeError
            If the vector has no axis 1.
        TypeError
            If `dim` is not an integer.
        ValueError
            If the shape of axis 1 of the vector is lower than dim.

        Examples
        --------
        >>> v = Vector.create([[1,2],[0,2],[3,0]])
        >>> v.component(0)
        Vector([1, 0, 3])
        >>> v.component(1)
        Vector([2, 2, 0])
        """

        if len(self.shape) < 2:
            raise TypeError("Operation not supperted on simple vectors")
        if not isinstance(dim, int):
            raise TypeError("Parameter 'dim' must be an integer")
        if self.shape[1] < dim + 1:
            raise ValueError(f"Vector has not component {dim}")
        return self[:, dim].view(Vector)

    @staticmethod
    def create(*args, **kwargs) -> Vector:
        """
        Creates a new vector.

        Returns
        -------
        Vector
            Vector created
        """

        arr = np.array(*args, **kwargs)
        return arr.view(Vector)
