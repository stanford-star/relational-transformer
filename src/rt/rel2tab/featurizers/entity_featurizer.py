from dataclasses import dataclass

from rt.rel2tab.featurizer import Featurizer


@dataclass
class EntityFeaturizerConfig:
    """Config for EntityFeaturizer (no fields needed)."""

    def build(self, device):
        return EntityFeaturizer()


class EntityFeaturizer(Featurizer):
    """Pass only train rows sharing the target's foreign key entity."""

    def compute_features(self, task, node_idxs, device, batch_size):
        return None

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        match = (train_f2ps == target_f2p).all(dim=-1)
        matched_labels = train_labels[match]
        if len(matched_labels) == 0:
            return None, train_labels, None
        return None, matched_labels, None
