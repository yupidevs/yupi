from __future__ import annotations

import enum
from typing import Any, Sequence


class UnitEnum(str, enum.Enum):
    """
    Base class for unit enums.

    Parameters
    ----------
    value : str
        The string value of the enum.
    """

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, value: Any) -> bool:
        if isinstance(value, str):
            return str(self) == value
        raise TypeError(
            f"Cannot compare {type(self)} with {type(value)}. "
            f"Expected str or {self.__class__.__name__}."
        )

    def __hash__(self) -> int:
        return self.value.__hash__()


class TimeU(UnitEnum):
    """
    Enum for the time unit.

    Parameters
    ----------
    Ts : str
        Teraseconds (10^12 seconds)
    Gs : str
        Gigaseconds (10^9 seconds)
    Ts : str
        Teraseconds (10^12 seconds)
    year : str
        Year (365 days)
    Ms : str
        Megaseconds (10^6 seconds)
    week : str
        Week (7 days)
    day : str
        Day (24 hours)
    h : str
        Hour (60 minutes)
    ks : str
        Kiloseconds (10^3 seconds)
    hs : str
        Hectoseconds (10^2 seconds)
    min : str
        Minute (60 seconds)
    das : str
        Decaseconds (10^1 seconds)
    s : str
        Seconds (10^0 seconds)
    ds : str
        Deciseconds (10^-1 seconds)
    ms : str
        Milliseconds (10^-3 seconds)
    us : str
        Microseconds (10^-6 seconds)
    ns : str
        Nanoseconds (10^-9 seconds)
    ps : str
        Picoseconds (10^-12 seconds)
    """

    Ts = "Ts"
    Gs = "Gs"
    year = "year"
    Ms = "Ms"
    week = "week"
    day = "day"
    h = "h"
    ks = "ks"
    hs = "hs"
    min = "min"
    das = "das"
    s = "s"
    ds = "ds"
    ms = "ms"
    us = "us"
    ns = "ns"
    ps = "ps"


class DistU(UnitEnum):
    """
    Enum for the distance unit.

    Parameters
    ----------
    Tm : str
        Terameter (10^12 m)
    Gm : str
        Gigameter (10^9 m)
    Mm : str
        Megameter (10^6 m)
    km : str
        Kilometer (10^3 m)
    hm : str
        Hectometer (10^2 m)
    dam : str
        Decameter (10^1 m)
    m : str
        Meter (10^0 m)
    dm : str
        Decimeter (10^-1 m)
    cm : str
        Centimeter (10^-2 m)
    mm : str
        Millimeter (10^-3 m)
    um : str
        Micrometer (10^-6 m)
    nm : str
        Nanometer (10^-9 m)
    pm : str
        Picometer (10^-12 m)
    geo : str
        Geospatial unit (latitude/longitude degree)

        Trajectories with geospatial coordinates are required to
        be 2-dimensional.
    """

    Tm = "Tm"
    Gm = "Gm"
    Mm = "Mm"
    km = "km"
    hm = "hm"
    dam = "dam"
    m = "m"
    dm = "dm"
    cm = "cm"
    mm = "mm"
    um = "um"
    nm = "nm"
    pm = "pm"
    geo = "geo"

    def __truediv__(self, time_u: TimeU | str) -> Units:
        return Units(self, time_u)


UnitsParsable = str | Sequence[str | DistU | TimeU]


class Units:
    """
    Units of a Trajectory (distance and time)

    Parameters
    ----------
    distance : DistU
        Distance unit.
    time : TimeU
        Time unit.
    """

    _dist_factors: dict[str, float] = {
        str(DistU.Tm): 1e12,
        str(DistU.Gm): 1e9,
        str(DistU.Mm): 1e6,
        str(DistU.km): 1e3,
        str(DistU.hm): 1e2,
        str(DistU.dam): 1e1,
        str(DistU.m): 1.0,
        str(DistU.dm): 1e-1,
        str(DistU.cm): 1e-2,
        str(DistU.mm): 1e-3,
        str(DistU.um): 1e-6,
        str(DistU.nm): 1e-9,
        str(DistU.pm): 1e-12,
    }

    _time_factors: dict[str, float] = {
        str(TimeU.Ts): 1e12,
        str(TimeU.Gs): 1e9,
        str(TimeU.year): 365 * 24 * 60 * 60,
        str(TimeU.Ms): 1e6,
        str(TimeU.week): 7 * 24 * 60 * 60,
        str(TimeU.day): 24 * 60 * 60,
        str(TimeU.h): 60 * 60,
        str(TimeU.ks): 1e3,
        str(TimeU.hs): 1e2,
        str(TimeU.min): 60,
        str(TimeU.das): 10,
        str(TimeU.s): 1.0,
        str(TimeU.ds): 1e-1,
        str(TimeU.ms): 1e-3,
        str(TimeU.us): 1e-6,
        str(TimeU.ns): 1e-9,
        str(TimeU.ps): 1e-12,
    }

    def __init__(self, distance: DistU | str, time: TimeU | str) -> None:
        if not isinstance(distance, (DistU, str)):
            raise TypeError(f"Invalid distance unit: {distance}")
        if not isinstance(time, (TimeU, str)):
            raise TypeError(f"Invalid time unit: {time}")

        self.dist = str(distance)
        self.time = str(time)

        if self.dist not in Units._dist_factors:
            raise ValueError(
                f"Unknown distance unit {self.dist}. "
                "You may need to register it first using Units.register_dist_unit."
            )

        if self.time not in Units._time_factors:
            raise ValueError(
                f"Unknown time unit {self.time}. "
                "You may need to register it first using Units.register_time_unit."
            )

    @staticmethod
    def parse(unit: UnitsParsable | Units) -> Units:
        """
        Create a Units object from a string.

        Parameters
        ----------
        unit : str
            String representation of the unit.

        Returns
        -------
        Units
            Units object.
        """
        if isinstance(unit, Units):
            return unit
        if isinstance(unit, str):
            try:
                dist, time = unit.split("/")
                return Units(dist.strip(), time.strip())
            except ValueError:
                raise ValueError(f"Invalid unit string: {unit}") from None

        if len(unit) != 2:
            raise ValueError(f"Invalid unit tuple: {unit}")
        return Units(unit[0], unit[1])

    def __repr__(self) -> str:
        return f"Units(distance={self.dist}, time={self.time})"

    def __str__(self) -> str:
        return f"{self.dist}/{self.time}"

    @staticmethod
    def register_dist_unit(unit: str, factor: float) -> None:
        """
        Register a new distance unit.

        Parameters
        ----------
        unit : str
            Name of the unit.
        factor : float
            Conversion factor to meters.
        """
        if unit in Units._dist_factors:
            raise ValueError(f"Unit {unit} already registered.")
        Units._dist_factors[unit] = factor

    @staticmethod
    def register_time_unit(unit: str, factor: float) -> None:
        """
        Register a new time unit.

        Parameters
        ----------
        unit : str
            Name of the unit.
        factor : float
            Conversion factor to seconds.
        """
        if unit in Units._time_factors:
            raise ValueError(f"Unit {unit} already registered.")
        Units._time_factors[unit] = factor

    def dist_to(self, to_u: DistU | str) -> float:
        """
        Convert distance to the specified unit.

        Parameters
        ----------
        to_u : DistU
            Unit to convert to.

        Returns
        -------
        float
            Conversion factor. Multiply this by the value to convert.
        """
        if to_u == self.dist:
            return 1.0

        from_u: str = str(self.dist)
        to_u = str(to_u)
        if to_u not in Units._dist_factors:
            raise ValueError(f"Uknown or invalid conversion unit {to_u}")

        return Units._dist_factors[from_u] / Units._dist_factors[to_u]

    def time_to(self, to_u: TimeU | str) -> float:
        """
        Convert time to the specified unit.

        Parameters
        ----------
        to_u : TimeU
            Unit to convert to.

        Returns
        -------
        float
            Conversion factor. Multiply this by the value to convert.
        """
        if to_u == self.time:
            return 1.0

        from_u: str = str(self.time)
        to_u = str(to_u)
        if to_u not in Units._time_factors:
            raise ValueError(f"Uknown or invalid conversion unit {to_u}")

        return Units._time_factors[from_u] / Units._time_factors[to_u]

    def to(self, to_u: Units) -> float:
        """
        Convert distance and time to the specified unit.

        Parameters
        ----------
        to_u : Units
            Unit to convert to.

        Returns
        -------
        float
            Conversion factor. Multiply this by the value to convert.
        """
        return self.dist_to(to_u.dist) / self.time_to(to_u.time)

    def __eq__(self, value: Any) -> bool:
        if isinstance(value, Units):
            return str(self) == str(value)
        try:
            val_units = Units.parse(value)
            return str(self) == str(val_units)
        except ValueError:
            raise TypeError(
                f"Cannot compare {type(self)} with {type(value)}. "
            ) from None

    def __hash__(self) -> int:
        return str(self).__hash__()
