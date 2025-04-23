"""
This module contains the structures for serializing and deserializing
trajectories.
"""

from yupi.core.serializers.csv_serializer import CSVSerializer
from yupi.core.serializers.json_serializer import JSONSerializer
from yupi.core.serializers.serializer import (
    InvalidTrajectoryFileExtensionError,
    Serializer,
)

__all__ = [
    "CSVSerializer",
    "InvalidTrajectoryFileExtensionError",
    "JSONSerializer",
    "Serializer",
]
