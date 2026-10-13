"""Type-aware GHD reconstruction: picks the bifurcated vs sidewall canonical
template per case based on its aneurysm_type.

This used to be a trimmed local fork of the class. AneuG is an independent
package, so we now import its MultiCanonicalGHDReconstruct instead of carrying a
copy that can drift from it.

The one wrinkle is the import path. AneuG's module does `from models.ghd_reconstruct
import ...` at top level, and angioflow has its OWN top-level `models` package that
shadows it on sys.path, so AneuG's root has to go to the FRONT of sys.path before
that import. We import it under its bare `models.` name (rather than through the
`AneuG.` namespace package) so only one copy of the module is ever loaded.
"""

import sys
import types
from pathlib import Path

ANEUG_ROOT = Path(__file__).resolve().parents[2] / "AneuG"

# `models` is a regular package on both sides, so putting AneuG first is enough.
if str(ANEUG_ROOT) not in sys.path:
    sys.path.insert(0, str(ANEUG_ROOT))

# `utils` is not: AneuG's has no __init__.py, and a REGULAR package of the same
# name (angioflow/utils) wins over a namespace one wherever it sits on sys.path.
# So bind AneuG's directory to the bare name ourselves, or AneuG's internal
# `from utils import utils_registration` picks up angioflow's package and fails.
# Nothing in angioflow imports the bare name, only `angioflow.utils`.
if not hasattr(sys.modules.get("utils"), "__path__") or \
        str(ANEUG_ROOT / "utils") not in getattr(sys.modules["utils"], "__path__", []):
    _utils = types.ModuleType("utils")
    _utils.__path__ = [str(ANEUG_ROOT / "utils")]
    sys.modules["utils"] = _utils

from models.multi_canonical_ghd_reconstruct import (  # noqa: E402
    CanonicalSpec, MultiCanonicalGHDReconstruct,
)

__all__ = ["CanonicalSpec", "MultiCanonicalGHDReconstruct", "ANEUG_ROOT"]
