from dataclasses import dataclass

from rt.rel2tab.featurizers import (
    EntityFeaturizerConfig,
    GlobalFeaturizerConfig,
    PrecomputedFeaturizerConfig,
    RDBLearnFeaturizerConfig,
    RTFeaturizerConfig,
)
from rt.rel2tab.predictors import (
    IdentityPredictorConfig,
    LGBMPredictorConfig,
    LinearPredictorConfig,
    MeanPredictorConfig,
    RidgePredictorConfig,
    TabICLBatchedPredictorConfig,
    TabPredictorConfig,
    XGBoostPredictorConfig,
)

FeaturizerConfig = (
    GlobalFeaturizerConfig
    | EntityFeaturizerConfig
    | RTFeaturizerConfig
    | RDBLearnFeaturizerConfig
    | PrecomputedFeaturizerConfig
)
PredictorConfig = (
    MeanPredictorConfig
    | LinearPredictorConfig
    | TabPredictorConfig
    | TabICLBatchedPredictorConfig
    | IdentityPredictorConfig
    | RidgePredictorConfig
    | LGBMPredictorConfig
    | XGBoostPredictorConfig
)


@dataclass
class Rel2TabModelConfig:
    """Config for Rel2TabModel.

    Fully independent of the RT model config.  Use ``build(device)`` to
    construct a ready-to-use Rel2TabModel.
    """

    featurizer: FeaturizerConfig
    predictor: PredictorConfig
    featurize_batch_size: int
    embedder: str
    d_text: int

    def build(self, device):
        from rt.rel2tab.model import Rel2TabModel

        featurizer = self.featurizer.build(device)
        predictor = self.predictor.build()

        return Rel2TabModel(
            featurizer=featurizer,
            predictor=predictor,
            featurize_batch_size=self.featurize_batch_size,
        )
