from prefscope.analysis.dataset import (
    auto_undesirable, dataset_reward, diagnose_dataset, feature_confound_correlation,
    label_inconsistency, region_behavior_contrast, region_membership_contrast,
    spurious_share, split_half_stable, symmetric_activity,
)
from prefscope.analysis.prompt_regions import (
    prompt_region_membership, regions_from_feature_presence,
)
from prefscope.analysis.preference import evaluate_preference
from prefscope.analysis.outcomes import (
    OUTCOME_KINDS, OUTCOME_NORMALIZATIONS, Normalization, NormalizedOutcomes,
    OutcomeAssociationResult, OutcomeKind, associate_outcomes,
    associate_outcomes_by_group, normalize_outcomes,
)
from prefscope.analysis.context import (
    PROMPT_SCOPES, classify_feature, profile_feature_context,
    profile_prompt_linkage,
)
from prefscope.analysis.run import diagnose, feature_preference_relevance
from prefscope.analysis.stats import inside_outside_contrast
from prefscope.analysis.paired_outcomes import (
    paired_outcome_shift,
    paired_outcome_shift_by_concept,
)
from prefscope.analysis.presence import (
    PRESENCE_POLICIES, PresenceMatrix, annotation_flag, concept_presence,
    feature_thresholds, semantic_presence,
)
from prefscope.analysis.paired import (
    RESPONSE_SCOPES, bh_adjust, paired_concept_shift,
    paired_concept_shift_by_region, summarize_response_scope,
)

__all__ = [
    "diagnose", "feature_preference_relevance", "evaluate_preference",
    "inside_outside_contrast", "dataset_reward", "split_half_stable",
    "spurious_share", "label_inconsistency", "diagnose_dataset",
    "symmetric_activity", "region_behavior_contrast", "region_membership_contrast",
    "prompt_region_membership",
    "regions_from_feature_presence",
    "feature_confound_correlation", "auto_undesirable",
    "PROMPT_SCOPES", "classify_feature", "profile_feature_context",
    "profile_prompt_linkage",
    "PRESENCE_POLICIES", "PresenceMatrix", "annotation_flag", "concept_presence",
    "feature_thresholds", "semantic_presence", "RESPONSE_SCOPES", "bh_adjust",
    "paired_concept_shift", "paired_concept_shift_by_region",
    "paired_outcome_shift", "paired_outcome_shift_by_concept",
    "summarize_response_scope", "OUTCOME_KINDS", "OUTCOME_NORMALIZATIONS",
    "OutcomeKind", "Normalization", "NormalizedOutcomes",
    "OutcomeAssociationResult", "normalize_outcomes",
    "associate_outcomes", "associate_outcomes_by_group",
]
