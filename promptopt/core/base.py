"""Base classes for the prompt optimization framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import uuid


@dataclass
class Example:
    """Represents an input-output example for prompt optimization."""
    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """Collection of examples for training/evaluation."""
    examples: List[Example]
    name: str = "unnamed_dataset"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]
    
    def split(self, train_ratio: float = 0.8) -> tuple['Dataset', 'Dataset']:
        """Split dataset into train and validation sets."""
        split_idx = int(len(self.examples) * train_ratio)
        train_examples = self.examples[:split_idx]
        val_examples = self.examples[split_idx:]
        
        train_dataset = Dataset(
            examples=train_examples,
            name=f"{self.name}_train",
            metadata={**self.metadata, "split": "train"}
        )
        val_dataset = Dataset(
            examples=val_examples,
            name=f"{self.name}_val",
            metadata={**self.metadata, "split": "val"}
        )
        
        return train_dataset, val_dataset


@dataclass
class Constraint:
    """Programmatic constraint for validating responses."""
    name: str
    description: str
    validator: Callable[[str], bool]
    weight: float = 1.0


@dataclass
class ConstraintValidationResult:
    """Results from validating constraints."""
    overall_score: float
    constraint_results: Dict[str, Dict[str, Any]]
    passed: bool = field(init=False)
    
    def __post_init__(self):
        self.passed = self.overall_score >= 0.5


@dataclass
class TaskSpec:
    """Specification for an optimization task."""
    name: str
    description: str
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    constraints: List[Constraint] = field(default_factory=list)
    examples: Optional[List[Example]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate_response(self, response: str) -> ConstraintValidationResult:
        """Programmatically validate response against constraints."""
        if not self.constraints:
            return ConstraintValidationResult(
                overall_score=1.0,
                constraint_results={}
            )
        
        results = {}
        total_score = 0.0
        total_weight = sum(c.weight for c in self.constraints)
        
        for constraint in self.constraints:
            try:
                passed = constraint.validator(response)
            except Exception as e:
                passed = False
                results[constraint.name] = {
                    'passed': passed,
                    'description': constraint.description,
                    'weight': constraint.weight,
                    'error': str(e)
                }
            else:
                results[constraint.name] = {
                    'passed': passed,
                    'description': constraint.description,
                    'weight': constraint.weight
                }
            
            if passed:
                total_score += constraint.weight
        
        return ConstraintValidationResult(
            overall_score=total_score / total_weight if total_weight > 0 else 0,
            constraint_results=results
        )


@dataclass
class OptimizedPrompt:
    """Represents an optimized prompt."""
    text: str
    examples: List[Example] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'examples': [{'input': e.input, 'output': e.output} for e in self.examples],
            'metadata': self.metadata,
            'optimization_history': self.optimization_history,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class Metrics:
    """Container for evaluation metrics."""
    scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def primary_score(self) -> float:
        """Get the primary metric score."""
        if 'primary_metric' in self.metadata:
            return self.scores.get(self.metadata['primary_metric'], 0.0)
        return list(self.scores.values())[0] if self.scores else 0.0


class BaseOptimizer(ABC):
    """Abstract base class for all prompt optimizers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimization_history = []
    
    @abstractmethod
    def optimize(self, task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt:
        """Optimize a prompt for the given task and dataset."""
        pass
    
    @abstractmethod
    def evaluate(self, prompt: OptimizedPrompt, test_set: Dataset) -> Metrics:
        """Evaluate an optimized prompt on a test set."""
        pass
    
    def with_config(self, config: Dict[str, Any]) -> 'BaseOptimizer':
        """Create a new optimizer instance with updated configuration."""
        new_config = {**self.config, **config}
        return self.__class__(config=new_config)