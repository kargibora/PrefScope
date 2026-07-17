"""Export a PrefScope lens + results into a compact JSON bundle for the web viewer.

The browser app can't read .npy/.parquet, so this flattens everything the
`viewer-web` app needs into small JSON files under its public/data/ dir:

    meta.json        headline numbers (EV, #verified, LOO-R^2, counts)
    features.json    per-feature concept + fidelity + win_assoc
    validation.json  per-model predicted vs actual win rate
    diagnosis.json   per-model x per-feature delta_vs_pool / net_direction (from the bank)
    examples/<fid>.json  (optional, --corpus) top battles per NAMED feature, sharded
                         so the viewer lazy-loads only the feature it's showing

Usage:
    python scripts/export_viewer_data.py --lens-dir lens_arena8b_m32_k4 \
        --corpus corpora/arena_merged.parquet --out viewer-web/public/data

See docs/reference/viewer-bundle.md for the full bundle contract.
"""
from __future__ import annotations

from .cli import main
from .diagnosis import export_diagnosis, export_head_to_head
from .examples import export_examples, export_examples_by_model, export_report_battles
from .features import (export_features, export_meta, feature_fire_rate,
                       feature_prompt_types)
from .maps import (_battle_ids_of, _clip_text, _concept_map, _project2d,
                   export_map, export_prompt_map, export_response_map)
from .sanitize import _concept_or_none, _dumps, _read_csv, _round, _sanitize
from .tables import (export_bias_screen, export_conditional, export_delta,
                     export_elicitation, export_prompt_features)

# NB: the underscore-prefixed helpers are deliberately in __all__ — the historical
# scripts/export_viewer_data.py module exposed them at module level (tests import
# e.g. `_dumps`), and the thin shim re-exports this package via `import *`.
__all__ = [
    # sanitize
    "_concept_or_none", "_dumps", "_read_csv", "_round", "_sanitize",
    # features
    "export_features", "export_meta", "feature_fire_rate", "feature_prompt_types",
    # diagnosis
    "export_diagnosis", "export_head_to_head",
    # examples
    "export_examples", "export_examples_by_model", "export_report_battles",
    # tables
    "export_bias_screen", "export_conditional", "export_delta",
    "export_elicitation", "export_prompt_features",
    # maps
    "_battle_ids_of", "_clip_text", "_concept_map", "_project2d",
    "export_map", "export_prompt_map", "export_response_map",
    # cli
    "main",
]
