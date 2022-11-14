from yupi.core.featurizers._acceleration_ftz import AccelerationFeaturizer
from yupi.core.featurizers._angle_ftz import AngleFeaturizer
from yupi.core.featurizers._spatial_ftz import SpatialFeaturizer
from yupi.core.featurizers._velocity_ftz import VelocityFeaturizer
from yupi.core.featurizers.featurizer import DEFAULT_ZERO_THRESHOLD, CompoundFeaturizer


class KineticFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that combines the spatial, velocity,
    acceleration, and angle featurizers.
    """

    def __init__(
        self,
        vel_stop_rate_threshold: float = 1,
        vel_change_rate_threshold: float = 1,
        zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
    ):
        super().__init__(
            SpatialFeaturizer(zero_threshold=zero_threshold),
            VelocityFeaturizer(
                stop_rate_threshold=vel_stop_rate_threshold,
                change_rate_threshold=vel_change_rate_threshold,
                zero_threshold=zero_threshold,
            ),
            AccelerationFeaturizer(zero_threshold=zero_threshold),
            AngleFeaturizer(zero_threshold=zero_threshold),
        )
