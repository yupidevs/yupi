from typing import List
import numpy as np


class Features():
    """
    Extracts useful information from a trajectory.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory
    """

    def __init__(self, traj):
        self.traj = traj

    @property
    def mean_vel(self) -> float:
        """float : Calculates de mean velocity"""
        return float(np.average(self.traj.v))

    @property
    def displacement(self) -> float:
        """float : Calculates de displacement"""
        traj = self.traj
        return (traj.r[-1] - traj.r[0]).norm

    @property
    def length(self) -> float:
        """float : Calculates de length"""
        traj = self.traj
        return sum(traj.delta_r.norm)

    def as_dict(self, only: List[str] = None,
                remove: List[str] = None) -> dict:
        """
        Get all the features as dictionary.

        Parameters
        ----------
        only : List[str], optional
            Filters the features by giving the ones named in the list,
            by default None.
        remove : List[str], optional
            Filters the features by removing the ones named in the list,
            by default None.

        Returns
        -------
        dict
            Features dictionary.
        """

        _dict = dict(self.__class__.__dict__)
        _dict = {k:v for k,v in _dict.items() if isinstance(v, property)}
        _keys = _dict.keys()
        if remove is not None:
            _keys = [k for k in _keys if k not in remove]
        if only is not None:
            _keys = [k for k in _keys if k in only]
        _dict = {k:v.__get__(self) for k,v in _dict.items() if k in _keys}
        return _dict
