from dataclasses import dataclass

from rt.rel2tab.featurizer import Featurizer


@dataclass
class GlobalFeaturizerConfig:
    """Config for GlobalFeaturizer (no fields needed)."""

    def build(self, device):
        return GlobalFeaturizer()


class GlobalFeaturizer(Featurizer):
    """Pass all train rows through with no features (for global mean baseline)."""

    def compute_features(self, task, node_idxs, device, batch_size):
        return None

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        return None, train_labels, None
