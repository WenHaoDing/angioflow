"""Dataset tags for processed CFD graphs. Dependency-free on purpose, so light
scripts (pollution scans, deploy checks, HF Jobs) can import it without pulling in
pytorch3d/pyvista via angioflow.data.datasets."""

# Dataset tags -- baked into processed filenames (static_<tag>_<cid>.pt,
# wss_<tag>_<cid>_<t>.pt) and carried as Data.version.
#   aneug_flow: whding123/AneuG-Flow, synthetic geometries (formerly "v1")
#   angioflow:  whding123/AngioFlow, real patient geometries (formerly "v2")
DATASET_TAGS = ("aneug_flow", "angioflow")
# Pre-rename tags still found in older processed roots, anomaly JSONs and saved
# checkpoint args. Map them on read only; nothing writes them anymore.
LEGACY_TAGS = {"v1": "aneug_flow", "v2": "angioflow"}


def normalize_tag(tag):
    return LEGACY_TAGS.get(tag, tag)


def split_tag_cid(stem):
    """"aneug_flow_stable_0" -> ("aneug_flow", "stable_0"); (None, None) if no known
    tag. Tags themselves contain "_", so match by prefix, never split("_", 1)."""
    for tag in DATASET_TAGS:
        if stem.startswith(tag + "_"):
            return tag, stem[len(tag) + 1:]
    return None, None
