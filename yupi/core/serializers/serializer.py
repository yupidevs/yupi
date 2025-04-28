"""
This contains the base class for all serializers.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import yupi
from yupi.exceptions import YupiExceptionError


class InvalidTrajectoryFileExtensionError(YupiExceptionError):
    """Raised when the trajectory file extension is invalid."""

    def __init__(
        self, file_path: str | Path, expected_extension: str | None = None
    ) -> None:
        message = f"Invalid trajectory file extension for '{file_path}'"
        if expected_extension is not None:
            message += f". Expected '{expected_extension}'."
        super().__init__(
            message,
        )
        self.file_path = file_path
        self.expected_extension = expected_extension


class Serializer(abc.ABC):
    """
    Abstract class for trajectory files.
    """

    @staticmethod
    @abc.abstractmethod
    def save(
        traj: yupi.Trajectory,
        file_path: str | Path,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Saves a trajectory to a file.

        Parameters
        ----------
        traj : Trajectory
            The trajectory to be saved.
        file_path : str | Path
            The path of the file to save.
        overwrite : bool
            If True, overwrites the file if it already exists.
        kwargs
            Additional keyword arguments.
        """

    @staticmethod
    @abc.abstractmethod
    def load(file_path: str | Path, **kwargs: Any) -> yupi.Trajectory:
        """
        Loads a trajectory from a file.

        Parameters
        ----------
        file_path : str | Path
            The path of the file to loaded.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Trajectory
            The trajectory loaded from the file.
        """

    @staticmethod
    def check_save_path(
        file_path: str | Path, overwrite: bool, extension: str | None
    ) -> None:
        """
        Checks if the file can be saved.

        Parameters
        ----------
        file_path : str | Path
            The path of the file to save.
        overwrite : bool
            If True, overwrites the file if it already exists.
        extension : str | None
            If given, it checks that the file has the given extension.
        """
        _path = Path(file_path) if isinstance(file_path, str) else file_path
        if extension is not None and _path.suffix != extension:
            raise InvalidTrajectoryFileExtensionError(
                file_path, expected_extension=extension
            )

        if _path.exists() and not overwrite:
            raise FileExistsError(f"File '{file_path}' already exists.")

        _path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_load_path(file_path: str | Path, extension: str | None) -> None:
        """
        Checks if the file can be loaded.

        Parameters
        ----------
        file_path : str | Path
            The path of the file to loaded.
        extension : str | None
            If given, it checks that the file has the given extension.
        """
        _path = Path(file_path) if isinstance(file_path, str) else file_path
        if extension is not None and _path.suffix != extension:
            raise InvalidTrajectoryFileExtensionError(
                file_path, expected_extension=extension
            )

        if not _path.exists():
            raise FileNotFoundError(f"File '{file_path}' not found.")
