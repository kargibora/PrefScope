"""Export a PrefScope lens + results into a compact static JSON bundle.

Browsers cannot read .npy/.parquet directly, so this flattens the supported
artifacts into small JSON files under a standalone output directory:

    meta.json        headline numbers (EV, #verified, LOO-R^2, counts)
    features.json    per-feature concept + fidelity + win_assoc
    validation.json  per-model predicted vs actual win rate
    diagnosis.json   per-model x per-feature delta_vs_pool / net_direction (from the bank)
    examples/<fid>.json  (optional, --corpus) top battles per NAMED feature, sharded
                         so the viewer lazy-loads only the feature it's showing

Usage:
    prefscope-export-viewer --lens-dir lens_arena8b_m32_k4 \
        --corpus corpora/arena_merged.parquet --out viewer-data

See docs/reference/viewer-bundle.md for the full bundle contract.
"""
from __future__ import annotations

from .cli import BUNDLE_SCHEMA_VERSION, main
from .clusters import export_feature_clusters
from .diagnosis import export_diagnosis, export_head_to_head
from .comparison import export_paired_comparison
from .examples import (export_examples, export_examples_by_model,
                       export_joint_examples, export_prompt_examples,
                       export_report_battles)
from .features import (export_features, export_meta, feature_fire_rate,
                       feature_prompt_types)
from .maps import (_battle_ids_of, _clip_text, _concept_map, _project2d,
                   export_feature_map, export_map, export_prompt_map,
                   export_response_map)
from .overview import (export_coactivation, export_concept_distribution,
                       export_prompt_coactivation,
                       export_prompt_concept_distribution)
from .sanitize import _concept_or_none, _dumps, _read_csv, _round, _sanitize
from .tables import (export_bias_screen, export_conditional, export_delta,
                     export_elicitation, export_prompt_features)

# The underscore-prefixed helpers remain importable for compatibility with early
# viewer integrations. New consumers should prefer the documented export functions.
__all__ = [
    # sanitize
    "_concept_or_none", "_dumps", "_read_csv", "_round", "_sanitize",
    # features
    "export_features", "export_meta", "feature_fire_rate", "feature_prompt_types",
    # diagnosis
    "export_diagnosis", "export_head_to_head",
    "export_paired_comparison",
    # examples
    "export_examples", "export_examples_by_model", "export_joint_examples",
    "export_prompt_examples",
    "export_report_battles",
    # overview
    "export_coactivation", "export_concept_distribution", "export_prompt_coactivation",
    "export_prompt_concept_distribution",
    "export_feature_clusters",
    # tables
    "export_bias_screen", "export_conditional", "export_delta",
    "export_elicitation", "export_prompt_features",
    # maps
    "_battle_ids_of", "_clip_text", "_concept_map", "_project2d",
    "export_feature_map", "export_map", "export_prompt_map", "export_response_map",
    # cli
    "BUNDLE_SCHEMA_VERSION", "main",
]
