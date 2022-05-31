"""
This contains the base class for all serializers.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import yupi


class Serializer(abc.ABC):
    """
    Abstract class for trajectory files.
    """

    @staticmethod
    @abc.abstractmethod
    def save(
        traj: yupi.Trajectory, file_name: str, overwrite: bool = False, **kwargs
    ) -> None:
        """
        Saves a trajectory to a file.

        Parameters
        ----------
        traj : Trajectory
            The trajectory to be saved.
        file_name : str
            The name of the file to save.
        overwrite : bool
            If True, overwrites the file if it already exists.
        kwargs
            Additional keyword arguments.
        """

    @staticmethod
    @abc.abstractmethod
    def load(file_name: str, **kwargs) -> yupi.Trajectory:
        """
        Loads a trajectory from a file.

        Parameters
        ----------
        file_name : str
            The name of the file to loaded.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Trajectory
            The trajectory loaded from the file.
        """

    @staticmethod
    def check_save_path(
        file_name: str, overwrite: bool, extension: Optional[str]
    ) -> None:
        """
        Checks if the file can be saved.

        Parameters
        ----------
        file_name : str
            The name of the file to save.
        overwrite : bool
            If True, overwrites the file if it already exists.
        extension : Optional[str]
            If given, it checks that the file has the given extension.
        """
        _path = Path(file_name)
        if extension is not None and _path.suffix != extension:
            raise ValueError(
                f"File extension must be '{extension}', not {_path.suffix}"
            )

        if _path.exists() and not overwrite:
            raise FileExistsError(f"File '{file_name}' already exists.")

        _path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_load_path(file_name: str, extension: Optional[str]) -> None:
        """
        Checks if the file can be loaded.

        Parameters
        ----------
        file_name : str
            The name of the file to loaded.
        extension : Optional[str]
            If given, it checks that the file has the given extension.
        """
        _path = Path(file_name)
        if extension is not None and _path.suffix != extension:
            raise ValueError(
                f"File extension must be '{extension}', not {_path.suffix}"
            )

        if not _path.exists():
            raise FileNotFoundError(f"File '{file_name}' not found.")
