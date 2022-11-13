from yupi.core.featurizers._acceleration_ftz import AccelerationFeaturizer
from yupi.core.featurizers._angle_ftz import AngleFeaturizer
from yupi.core.featurizers._spatial_ftz import SpatialFeaturizer
from yupi.core.featurizers._velocity_ftz import VelocityFeaturizer
from yupi.core.featurizers.featurizer import CompoundFeaturizer


class KineticFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that combines the spatial, velocity,
    acceleration, and angle featurizers.
    """

    def __init__(
        self, vel_stop_rate_threshold: float = 1, vel_change_rate_threshold: float = 1
    ):
        super().__init__(
            SpatialFeaturizer(),
            VelocityFeaturizer(vel_stop_rate_threshold, vel_change_rate_threshold),
            AccelerationFeaturizer(),
            AngleFeaturizer(),
        )
