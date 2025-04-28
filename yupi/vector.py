"""
This contains the Vector structure used across the library to store data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import norm as nrm


class Vector(np.ndarray):
    """Represents a vector"""

    __array_priority__ = 100

    def __new__(
        cls: type[Vector],
        arr: Any,
        dtype: Any = None,
        copy: bool = False,
    ) -> Vector:
        try:
            vec = np.asarray(arr, dtype=dtype)
        except Exception as e:
            raise TypeError(
                f"Input 'arr' is not convertible to a NumPy array: {e}"
            ) from e
        if copy:
            vec = vec.copy()
        return vec.view(cls)

    def __add__(self, other: Any) -> Vector:
        return super().__add__(other).view(Vector)

    def __iadd__(self, other: Any) -> Vector:
        return super().__iadd__(other).view(Vector)

    def __sub__(self, other: Any) -> Vector:  # type: ignore[override]
        return super().__sub__(other).view(Vector)

    def __isub__(self, other: Any) -> Vector:  # type: ignore[override]
        return super().__isub__(other).view(Vector)

    def __mul__(self, other: Any) -> Vector:
        return super().__mul__(other).view(Vector)

    def __imul__(self, other: Any) -> Vector:
        return super().__imul__(other).view(Vector)

    @property
    def norm(self) -> Vector | float:
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

    def component(self, dim: int) -> Vector:
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
