"""Common interfaces for the prompt optimization framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class ComparisonResult(Enum):
    """Result of comparing two responses."""
    A_BETTER = "a_better"
    B_BETTER = "b_better"
    TIE = "tie"


@dataclass
class JudgmentResult:
    """Result from an LLM judge comparison."""
    winner: ComparisonResult
    confidence: float  # 0.0 to 1.0
    reasoning: str
    metadata: Dict[str, Any] = None


@dataclass
class TournamentResult:
    """Result from a tournament evaluation."""
    rankings: List[Tuple[str, float]]  # (prompt_id, win_rate)
    matchup_results: Dict[str, Dict[str, ComparisonResult]]
    total_matches: int
    metadata: Dict[str, Any] = None


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        pass
    
    @abstractmethod
    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the cost for a generation."""
        pass


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Compute the metric score."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the metric."""
        pass
    
    @property
    def higher_is_better(self) -> bool:
        """Whether higher scores are better for this metric."""
        return True


class Judge(ABC):
    """Abstract base class for response judges."""
    
    @abstractmethod
    def compare(self, response_a: str, response_b: str, 
                criteria: str, context: Optional[str] = None) -> JudgmentResult:
        """Compare two responses and determine which is better."""
        pass


class DataGenerator(ABC):
    """Abstract base class for synthetic data generators."""
    
    @abstractmethod
    def generate(self, count: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate synthetic data examples."""
        pass


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    max_iterations: int = 10
    batch_size: int = 8
    early_stopping: bool = True
    early_stopping_patience: int = 3
    budget_limit: Optional[float] = None
    timeout_seconds: Optional[int] = None
    seed: Optional[int] = None
    additional_params: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_iterations': self.max_iterations,
            'batch_size': self.batch_size,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'budget_limit': self.budget_limit,
            'timeout_seconds': self.timeout_seconds,
            'seed': self.seed,
            'additional_params': self.additional_params or {}
        }


@dataclass
class BusinessContext:
    """Context information for business-focused optimization."""
    industry: str
    company_size: str  # "startup", "smb", "enterprise"
    use_case: str
    compliance_requirements: List[str] = None
    brand_voice: Optional[str] = None
    target_audience: Optional[str] = None
    additional_context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'industry': self.industry,
            'company_size': self.company_size,
            'use_case': self.use_case,
            'compliance_requirements': self.compliance_requirements or [],
            'brand_voice': self.brand_voice,
            'target_audience': self.target_audience,
            'additional_context': self.additional_context or {}
        }