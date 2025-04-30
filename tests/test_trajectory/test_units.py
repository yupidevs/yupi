import pytest

from yupi import DistU, TimeU, Units


def test_unit_creation() -> None:
    unit = Units("m", "s")

    assert unit.dist == "m"
    assert unit.time == "s"

    unit = Units(DistU.km, TimeU.h)

    assert unit.dist == "km"
    assert unit.time == "h"

    unit = Units("cm", "min")

    assert unit.dist == "cm"
    assert unit.time == "min"

    unit = DistU.nm / TimeU.s

    assert unit.dist == "nm"
    assert unit.time == "s"

    unit = Units.parse("cm/min")

    assert unit.dist == "cm"
    assert unit.time == "min"

    unit = Units.parse(("cm", "min"))

    assert unit.dist == "cm"
    assert unit.time == "min"

    with pytest.raises(TypeError):
        Units("m", 2)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        Units(2, "s")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        Units.parse("cms")

    with pytest.raises(ValueError):
        Units.parse(["cm", "min", "km"])


def test_equality() -> None:
    assert DistU.km == "km"
    assert DistU.km == DistU.km
    assert DistU.km != DistU.m

    assert TimeU.day == "day"
    assert TimeU.day == TimeU.day
    assert TimeU.day != TimeU.h

    assert DistU.km / TimeU.h == "km/h"
    assert Units("m", "s") == "m/s"
    assert Units("m", "s") == Units.parse("m/s")

    with pytest.raises(TypeError):
        _ = Units("m", "s") == 2


def test_conversion() -> None:
    unit = DistU.km / TimeU.h

    assert unit.dist_to(DistU.km) == pytest.approx(1)
    assert unit.dist_to(DistU.m) == pytest.approx(1000)

    assert unit.time_to(TimeU.h) == pytest.approx(1)
    assert unit.time_to(TimeU.s) == pytest.approx(3600)

    assert unit.to(Units.parse("m/s")) == pytest.approx(1000 / 3600)


def test_register_units() -> None:
    Units.register_dist_unit("foo", 42)
    Units.register_time_unit("bar", 73)

    unit = Units("foo", "bar")

    assert unit.dist == "foo"
    assert unit.time == "bar"

    assert unit.dist_to("foo") == pytest.approx(1)
    assert unit.dist_to(DistU.m) == pytest.approx(42)
    assert unit.dist_to("km") == pytest.approx(42 / 1000)

    assert unit.time_to("bar") == pytest.approx(1)
    assert unit.time_to("min") == pytest.approx(73 / 60)

    assert unit.to(Units("foo", "bar")) == pytest.approx(1)
    assert unit.to(Units("foo", TimeU.s)) == pytest.approx(1 / 73)
    assert unit.to(Units(DistU.m, TimeU.s)) == pytest.approx(42 / 73)

    with pytest.raises(ValueError):
        Units.register_dist_unit("foo", 42)

    with pytest.raises(ValueError):
        Units.register_time_unit("bar", 73)

    with pytest.raises(ValueError):
        unit.dist_to("foo2")

    with pytest.raises(ValueError):
        unit.time_to("bar2")

    with pytest.raises(ValueError):
        _ = Units("foo2", "s")

    with pytest.raises(ValueError):
        _ = Units("m", "bar2")
