"""Backward-compatible imports for the reusable semantic-presence API.

New analysis code should import from :mod:`prefscope.analysis.presence`; viewer exports
retain this module so third-party imports do not break.
"""
from prefscope.analysis.presence import feature_thresholds, semantic_presence

__all__ = ["feature_thresholds", "semantic_presence"]
