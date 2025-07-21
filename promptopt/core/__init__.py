"""Core components of the prompt optimization framework."""

from .base import (
    Example,
    Dataset,
    Constraint,
    ConstraintValidationResult,
    TaskSpec,
    OptimizedPrompt,
    Metrics,
    BaseOptimizer
)

from .interfaces import (
    ComparisonResult,
    JudgmentResult,
    TournamentResult,
    LLMProvider,
    EvaluationMetric,
    Judge,
    DataGenerator,
    OptimizationConfig,
    BusinessContext
)

from .metrics import (
    AccuracyMetric,
    F1Metric,
    BLEUMetric,
    ConstraintAdherenceMetric,
    CostEfficiencyMetric,
    ConsistencyMetric
)

from .constraints import (
    create_conciseness_constraint,
    create_format_constraint,
    create_tone_constraint,
    create_no_pii_constraint,
    create_json_format_constraint,
    create_bullet_points_constraint,
    create_language_constraint,
    create_business_constraint
)

__all__ = [
    # Base classes
    "Example",
    "Dataset",
    "Constraint",
    "ConstraintValidationResult",
    "TaskSpec",
    "OptimizedPrompt",
    "Metrics",
    "BaseOptimizer",
    
    # Interfaces
    "ComparisonResult",
    "JudgmentResult",
    "TournamentResult",
    "LLMProvider",
    "EvaluationMetric",
    "Judge",
    "DataGenerator",
    "OptimizationConfig",
    "BusinessContext",
    
    # Metrics
    "AccuracyMetric",
    "F1Metric",
    "BLEUMetric",
    "ConstraintAdherenceMetric",
    "CostEfficiencyMetric",
    "ConsistencyMetric",
    
    # Constraint creators
    "create_conciseness_constraint",
    "create_format_constraint",
    "create_tone_constraint",
    "create_no_pii_constraint",
    "create_json_format_constraint",
    "create_bullet_points_constraint",
    "create_language_constraint",
    "create_business_constraint",
]