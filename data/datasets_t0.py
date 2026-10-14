"""Frame-subset variant of AngioFlowWallDataset (default: t=0 only, for baselines)."""
from __future__ import annotations

import os

from angioflow.data.dataset_tags import normalize_tag
from angioflow.data.datasets import AngioFlowWallDataset


class AngioFlowWallDatasetT0(AngioFlowWallDataset):
    """One sample per (case, frame) for a chosen frame subset. Same on-disk layout
    as the parent dataset, so a `save_dir` populated by the full dataset works directly.

    Args:
        versions: which datasets to include, e.g. ``("aneug_flow",)`` or ``("aneug_flow", "angioflow")``.
            Legacy "v1"/"v2" (e.g. from an old checkpoint's saved args) are mapped to
            these. Defaults to all available datasets.
        frames: which frame indices to sample, e.g. ``[0]`` or ``range(0, 80, 8)``.
            Defaults to ``[0]`` (t=0 only).
    """

    def __init__(self, *args, versions=None, frames=None, **kwargs):
        self._versions = {normalize_tag(v) for v in versions} if versions is not None else None
        self._frames = list(frames) if frames is not None else [0]
        super().__init__(*args, **kwargs)

    def _case_complete(self, version, cid):
        return os.path.exists(self._static_path(version, cid)) and \
               all(os.path.exists(self._wss_path(version, cid, t)) for t in self._frames)

    def _rebuild_successful_index(self):
        case_keys, cases, index = [], [], []
        for (version, cid) in self.case_keys:
            if self._versions is not None and version not in self._versions:
                continue
            if not self._case_complete(version, cid):
                continue
            case_keys.append((version, cid))
            cases.append(self.case_dir_by_key[(version, cid)])
            index.extend((version, cid, t) for t in self._frames)
        self.case_keys = case_keys
        self.cases = cases
        self._index = index

    @property
    def processed_file_names(self):
        def _rel(version, name):
            return f"{version}/{name}" if self.per_version_dirs else name
        statics = [_rel(v, f"static_{v}_{cid}.pt") for (v, cid) in self.case_keys]
        wss = [_rel(v, f"wss_{v}_{cid}_{t}.pt") for (v, cid) in self.case_keys for t in self._frames]
        return statics + wss
