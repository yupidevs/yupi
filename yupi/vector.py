"""
This contains the Vector structure used across the library to store data.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from numpy.linalg.linalg import norm as nrm


class Vector(np.ndarray):
    """Represents a vector"""

    def __new__(cls, arr, dtype=None, copy=False):
        vec = np.asarray(arr, dtype=dtype)
        if copy:
            vec = vec.copy()
        return vec.view(cls)

    @property
    def norm(self) -> Union[Vector, float]:
        """Vector : Calculates the norm of the vector. If the vector
        is alist of vectors then the norm of each item is calculated"""
        if len(self.shape) < 2:
            return float(nrm(self))
        return Vector(nrm(self, axis=1))

    @property
    def delta(self) -> Vector:
        """Vector : Calculates the differnece between each item"""
        return Vector(np.diff(self, axis=0))

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
        >>> v = Vector([[1,2],[0,2],[3,0]])
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
        .. deprecated:: 0.10.0
            :func:`Vector.create` will be removed in a future version, use
            :class:`Vector` constructor instead.

        Creates a new vector.

        Returns
        -------
        Vector
            Vector created
        """

        warnings.warn(
            "Vector.create is deprecated and it will be removed in a future version, "
            "use Vector constructor instead.",
            DeprecationWarning,
        )
        return np.array(*args, **kwargs).view(Vector)
