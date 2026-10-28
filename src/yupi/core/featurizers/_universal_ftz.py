from yupi.core.featurizers._kinetic_ftz import KineticFeaturizer
from yupi.core.featurizers._time_ftz import TimeFeaturizer
from yupi.core.featurizers.featurizer import DEFAULT_ZERO_THRESHOLD, CompoundFeaturizer


class UniversalFeaturizer(CompoundFeaturizer):
    """A featurizer that combines all the featurizers in yupi."""

    def __init__(
        self,
        vel_stop_rate_threshold: float = 1,
        vel_change_rate_threshold: float = 1,
        time_from_zero: bool = True,
        zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
    ):
        super().__init__(
            TimeFeaturizer(from_zero=time_from_zero, zero_threshold=zero_threshold),
            KineticFeaturizer(
                vel_stop_rate_threshold=vel_stop_rate_threshold,
                vel_change_rate_threshold=vel_change_rate_threshold,
                zero_threshold=zero_threshold,
            ),
        )
