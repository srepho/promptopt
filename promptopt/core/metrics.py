"""Evaluation metrics for prompt optimization."""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import re

from .interfaces import EvaluationMetric


class AccuracyMetric(EvaluationMetric):
    """Simple accuracy metric."""
    
    def compute(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Compute exact match accuracy."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = sum(1 for pred, truth in zip(predictions, ground_truth) 
                     if pred.strip() == truth.strip())
        return correct / len(predictions) if predictions else 0.0
    
    @property
    def name(self) -> str:
        return "accuracy"


class F1Metric(EvaluationMetric):
    """Token-level F1 score metric."""
    
    def compute(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Compute average F1 score across examples."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        f1_scores = []
        for pred, truth in zip(predictions, ground_truth):
            f1 = self._compute_f1(pred, truth)
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def _compute_f1(self, pred: str, truth: str) -> float:
        """Compute F1 score for a single example."""
        pred_tokens = set(pred.lower().split())
        truth_tokens = set(truth.lower().split())
        
        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        intersection = pred_tokens & truth_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @property
    def name(self) -> str:
        return "f1"


class BLEUMetric(EvaluationMetric):
    """BLEU score metric for text generation."""
    
    def __init__(self, n_gram: int = 4):
        self.n_gram = n_gram
    
    def compute(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Compute average BLEU score."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        bleu_scores = []
        for pred, truth in zip(predictions, ground_truth):
            score = self._compute_bleu(pred, truth)
            bleu_scores.append(score)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def _compute_bleu(self, pred: str, truth: str) -> float:
        """Compute BLEU score for a single example."""
        pred_tokens = pred.lower().split()
        truth_tokens = truth.lower().split()
        
        if not pred_tokens:
            return 0.0
        
        scores = []
        for n in range(1, min(self.n_gram + 1, len(pred_tokens) + 1)):
            score = self._compute_ngram_precision(pred_tokens, truth_tokens, n)
            scores.append(score)
        
        if not scores or all(s == 0 for s in scores):
            return 0.0
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(truth_tokens) / len(pred_tokens)))
        
        # Geometric mean of n-gram precisions
        bleu = bp * np.exp(np.mean([np.log(s) if s > 0 else -np.inf for s in scores]))
        
        return bleu if not np.isnan(bleu) and not np.isinf(bleu) else 0.0
    
    def _compute_ngram_precision(self, pred_tokens: List[str], 
                                truth_tokens: List[str], n: int) -> float:
        """Compute n-gram precision."""
        pred_ngrams = Counter(self._get_ngrams(pred_tokens, n))
        truth_ngrams = Counter(self._get_ngrams(truth_tokens, n))
        
        overlap = sum((pred_ngrams & truth_ngrams).values())
        total = sum(pred_ngrams.values())
        
        return overlap / total if total > 0 else 0.0
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Extract n-grams from token list."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    @property
    def name(self) -> str:
        return f"bleu_{self.n_gram}"


class ConstraintAdherenceMetric(EvaluationMetric):
    """Metric for measuring constraint adherence."""
    
    def __init__(self, constraints: List[Any]):
        self.constraints = constraints
    
    def compute(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Compute average constraint adherence score."""
        scores = []
        for pred in predictions:
            score = self._compute_constraint_score(pred)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_constraint_score(self, text: str) -> float:
        """Compute constraint score for a single text."""
        if not self.constraints:
            return 1.0
        
        total_weight = sum(c.weight for c in self.constraints)
        score = 0.0
        
        for constraint in self.constraints:
            try:
                if constraint.validator(text):
                    score += constraint.weight
            except:
                pass
        
        return score / total_weight if total_weight > 0 else 0.0
    
    @property
    def name(self) -> str:
        return "constraint_adherence"


class CostEfficiencyMetric(EvaluationMetric):
    """Metric for measuring cost efficiency of prompts."""
    
    def __init__(self, quality_metric: EvaluationMetric, cost_weight: float = 0.5):
        self.quality_metric = quality_metric
        self.cost_weight = cost_weight
    
    def compute(self, predictions: List[str], ground_truth: List[str], 
                costs: Optional[List[float]] = None) -> float:
        """Compute cost-adjusted quality score."""
        quality_score = self.quality_metric.compute(predictions, ground_truth)
        
        if not costs:
            return quality_score
        
        avg_cost = np.mean(costs)
        # Normalize cost to 0-1 range (assuming max reasonable cost of $1 per example)
        normalized_cost = min(avg_cost, 1.0)
        
        # Combine quality and cost (higher quality, lower cost is better)
        efficiency_score = (1 - self.cost_weight) * quality_score + \
                          self.cost_weight * (1 - normalized_cost)
        
        return efficiency_score
    
    @property
    def name(self) -> str:
        return f"cost_efficiency_{self.quality_metric.name}"


class ConsistencyMetric(EvaluationMetric):
    """Metric for measuring response consistency across similar inputs."""
    
    def compute(self, predictions: List[str], ground_truth: List[str] = None) -> float:
        """Compute consistency score based on response variance."""
        if len(predictions) < 2:
            return 1.0
        
        # Compute pairwise similarity
        similarities = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                sim = self._compute_similarity(predictions[i], predictions[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    @property
    def name(self) -> str:
        return "consistency"