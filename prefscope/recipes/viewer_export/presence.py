"""Backward-compatible imports for the reusable semantic-presence API.

New analysis code should import from :mod:`prefscope.recipes.analysis.presence`; viewer exports
retain this module so third-party imports do not break.
"""
from prefscope.recipes.analysis.presence import feature_thresholds, semantic_presence

__all__ = ["feature_thresholds", "semantic_presence"]
