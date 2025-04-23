from pathlib import Path

import pytest

from yupi.core.serializers import Serializer
from yupi.core.serializers.serializer import InvalidTrajectoryFileExtensionError


def test_check_save_path():
    invalid_ext_path = Path("traj.mp4")

    # Test with invalid extension
    with pytest.raises(InvalidTrajectoryFileExtensionError) as excinfo:
        Serializer.check_save_path(invalid_ext_path, overwrite=False, extension=".json")
        assert excinfo.value.file_path == invalid_ext_path
        assert excinfo.value.expected_extension == ".json"


def test_check_load_path():
    invalid_ext_path = Path("traj.mp4")

    # Test with invalid extension
    with pytest.raises(InvalidTrajectoryFileExtensionError) as excinfo:
        Serializer.check_load_path(invalid_ext_path, extension=".json")
        assert excinfo.value.file_path == invalid_ext_path
        assert excinfo.value.expected_extension == ".json"

    valid_non_existing_path = Path("traj.json")

    # Test with valid non-existing path
    with pytest.raises(FileNotFoundError):
        Serializer.check_load_path(valid_non_existing_path, extension=".json")
