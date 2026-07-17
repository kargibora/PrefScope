from prefscope.analysis.dataset import (
    auto_undesirable, dataset_reward, diagnose_dataset, feature_confound_correlation,
    label_inconsistency, region_behavior_contrast, spurious_share, split_half_stable,
    symmetric_activity,
)
from prefscope.analysis.preference import evaluate_preference
from prefscope.analysis.run import diagnose, feature_preference_relevance
from prefscope.analysis.stats import inside_outside_contrast

__all__ = [
    "diagnose", "feature_preference_relevance", "evaluate_preference",
    "inside_outside_contrast", "dataset_reward", "split_half_stable",
    "spurious_share", "label_inconsistency", "diagnose_dataset",
    "symmetric_activity", "region_behavior_contrast",
    "feature_confound_correlation", "auto_undesirable",
]
