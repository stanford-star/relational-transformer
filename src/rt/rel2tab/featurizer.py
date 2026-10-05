import json
from abc import ABC, abstractmethod
from pathlib import Path


def load_table_info(db: str, pre_dir: str) -> dict:
    """Load table metadata from ``pre_dir/db/table_info.json``."""
    path = Path(pre_dir).expanduser() / db / "table_info.json"
    with open(path) as f:
        return json.load(f)


def get_table_splits(table_info: dict, table_name: str) -> dict[str, dict]:
    """Return ``{split: {node_idx_offset, num_nodes}}`` for available splits."""
    splits = {}
    for split in ["train", "val", "test", "db"]:
        key = f"{table_name}:{split.capitalize()}"
        if key in table_info:
            splits[split] = table_info[key]
    return splits


def validate_contiguous(splits_info: dict[str, dict], db: str, table_name: str):
    """Raise if node indices are not contiguous across splits."""
    sorted_offsets = sorted(
        (info["node_idx_offset"], info["num_nodes"]) for info in splits_info.values()
    )
    for i in range(len(sorted_offsets) - 1):
        end = sorted_offsets[i][0] + sorted_offsets[i][1]
        nxt = sorted_offsets[i + 1][0]
        if end != nxt:
            raise ValueError(
                f"Non-contiguous node_idxs across splits for {db}/{table_name}. "
                f"Offsets: {sorted_offsets}"
            )


class Featurizer(ABC):
    """Controls how task-node rows are transformed into features for the predictor.

    Lifecycle within ``Rel2TabModel.predict``:
      1. ``compute_features`` is called once per batch with all N task-node
         indices.  Use this for expensive bulk work (e.g. building local
         contexts and running a neural encoder).
      2. ``featurize`` is called once per (batch-item, context-size) with the
         visible train rows for a single target.  Use this for row selection
         or transformation (e.g. filtering to same-entity rows).

    """

    @abstractmethod
    def compute_features(self, task, node_idxs, device, batch_size):
        """Compute per-node features in bulk.

        Args:
            task: Eval Task namedtuple (has .db_name, .table_name, .split,
                .task_type, etc.).
            node_idxs: 1-D LongTensor of length N (node indices in the graph).
            device: torch device for computation.
            batch_size: Suggested micro-batch size for chunked inference.

        Returns:
            (N, d_feat) Tensor of per-node features, or None if no features
            are produced.
        """

    @abstractmethod
    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        """Select/transform visible train rows for one target before prediction.

        Args:
            train_labels: 1-D float Tensor of visible train labels.
            train_f2ps: (num_train, F) LongTensor of f2p_nbr_idxs per train row.
            target_f2p: (F,) LongTensor, f2p_nbr_idxs of the target row.
            train_feats: (num_train, d_feat) Tensor or None (from compute_features).
            test_feat: (d_feat,) Tensor or None (from compute_features).

        Returns:
            3-tuple (train_feats, train_labels, test_feat) to pass to the
            predictor.  Any element may be None.
        """
