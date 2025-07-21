"""Hybrid optimization strategies combining DSPy and GRPO approaches."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

from ..core.base import BaseOptimizer, Dataset, TaskSpec, OptimizedPrompt, Metrics, Example
from ..core.interfaces import OptimizationConfig, BusinessContext
from ..utils.llm_clients import BaseLLMClient
from .dspy_adapter import DSPyAdapter, DSPyConfig
from .grpo_adapter import GRPOAdapter, GRPOConfig, PromptCandidate


@dataclass
class HybridConfig:
    """Configuration for hybrid optimization."""
    strategy: str = "sequential"  # "sequential", "ensemble", "feedback", "cost_aware"
    dspy_config: Optional[DSPyConfig] = None
    grpo_config: Optional[GRPOConfig] = None
    weight_dspy: float = 0.5
    weight_grpo: float = 0.5
    max_iterations: int = 3
    budget_limit: Optional[float] = None


class SequentialHybrid(BaseOptimizer):
    """Apply DSPy optimization first, then GRPO refinement."""
    
    def __init__(self, llm_client: BaseLLMClient, 
                 judge_client: Optional[BaseLLMClient] = None,
                 config: Optional[HybridConfig] = None):
        super().__init__()
        self.llm_client = llm_client
        self.judge_client = judge_client or llm_client
        self.config = config or HybridConfig(strategy="sequential")
        
        # Initialize component optimizers
        self.dspy_optimizer = DSPyAdapter(
            llm_client=llm_client,
            config=self.config.dspy_config or DSPyConfig()
        )
        self.grpo_optimizer = GRPOAdapter(
            llm_client=llm_client,
            judge_client=self.judge_client,
            config=self.config.grpo_config or GRPOConfig()
        )
    
    def optimize(self, task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt:
        """Optimize using sequential hybrid approach."""
        logging.info("Starting Sequential Hybrid optimization")
        
        # Phase 1: DSPy optimization
        logging.info("Phase 1: DSPy optimization")
        dspy_result = self.dspy_optimizer.optimize(task_spec, dataset)
        
        # Phase 2: Create variations based on DSPy result
        initial_candidates = self._create_variations_from_dspy(dspy_result, task_spec, dataset)
        
        # Phase 3: GRPO refinement
        logging.info("Phase 2: GRPO refinement")
        # Temporarily modify GRPO to use our candidates
        original_init = self.grpo_optimizer._initialize_population
        self.grpo_optimizer._initialize_population = lambda ts, ds: initial_candidates
        
        try:
            final_result = self.grpo_optimizer.optimize(task_spec, dataset)
        finally:
            # Restore original method
            self.grpo_optimizer._initialize_population = original_init
        
        # Add hybrid metadata
        final_result.metadata.update({
            "optimizer": "sequential_hybrid",
            "dspy_phase_cost": dspy_result.metadata.get("optimization_cost", 0),
            "total_cost": self.llm_client.get_cost_summary()["total_cost"],
            "phases": ["dspy", "grpo"]
        })
        
        return final_result
    
    def _create_variations_from_dspy(self, dspy_result: OptimizedPrompt, 
                                   task_spec: TaskSpec, dataset: Dataset) -> List[PromptCandidate]:
        """Create GRPO candidates based on DSPy result."""
        candidates = []
        
        # Base candidate from DSPy
        candidates.append(PromptCandidate(
            id="dspy_base",
            text=dspy_result.text,
            examples=dspy_result.examples,
            metadata={"source": "dspy_original"}
        ))
        
        # Variations with different example counts
        for num_examples in [2, 4, 6]:
            examples = dspy_result.examples[:num_examples] if len(dspy_result.examples) >= num_examples else dspy_result.examples
            candidates.append(PromptCandidate(
                id=f"dspy_ex{num_examples}",
                text=dspy_result.text,
                examples=examples,
                metadata={"source": "dspy_variation", "num_examples": num_examples}
            ))
        
        # Rephrased versions
        rephrased_text = self._rephrase_prompt(dspy_result.text)
        candidates.append(PromptCandidate(
            id="dspy_rephrased",
            text=rephrased_text,
            examples=dspy_result.examples[:3],
            metadata={"source": "dspy_rephrased"}
        ))
        
        # Add some fresh candidates for diversity
        for i in range(3):
            candidates.append(PromptCandidate(
                id=f"fresh_{i}",
                text=self._create_fresh_prompt(task_spec),
                examples=self._select_diverse_examples(dataset, 4),
                metadata={"source": "fresh"}
            ))
        
        return candidates
    
    def _rephrase_prompt(self, original: str) -> str:
        """Rephrase prompt using LLM."""
        prompt = f"Rephrase this instruction to be clearer and more concise:\n\n{original}\n\nRephrased version:"
        response = self.llm_client.generate(prompt, temperature=0.7, max_tokens=200)
        return response.text.strip()
    
    def _create_fresh_prompt(self, task_spec: TaskSpec) -> str:
        """Create a fresh prompt variation."""
        templates = [
            f"Complete the following task: {task_spec.description}\n\nBe precise and follow the format requirements.",
            f"{task_spec.description}\n\nEnsure your response adheres to all specified constraints.",
            f"Task: {task_spec.description}\n\nProvide a well-formatted response that meets all requirements."
        ]
        import random
        return random.choice(templates)
    
    def _select_diverse_examples(self, dataset: Dataset, count: int) -> List[Example]:
        """Select diverse examples from dataset."""
        if len(dataset) <= count:
            return dataset.examples
        
        # Simple diversity selection based on length
        examples = sorted(dataset.examples, key=lambda x: len(x.input))
        indices = np.linspace(0, len(examples) - 1, count, dtype=int)
        return [examples[i] for i in indices]
    
    def evaluate(self, prompt: OptimizedPrompt, test_set: Dataset) -> Metrics:
        """Evaluate using GRPO evaluator."""
        return self.grpo_optimizer.evaluate(prompt, test_set)


class EnsembleHybrid(BaseOptimizer):
    """Run multiple optimizers and select best performer."""
    
    def __init__(self, llm_client: BaseLLMClient,
                 judge_client: Optional[BaseLLMClient] = None,
                 config: Optional[HybridConfig] = None):
        super().__init__()
        self.llm_client = llm_client
        self.judge_client = judge_client or llm_client
        self.config = config or HybridConfig(strategy="ensemble")
        
        # Initialize all optimizers
        self.optimizers = {
            "dspy": DSPyAdapter(llm_client, self.config.dspy_config),
            "grpo": GRPOAdapter(llm_client, judge_client, self.config.grpo_config),
            "sequential": SequentialHybrid(llm_client, judge_client, HybridConfig(strategy="sequential"))
        }
    
    def optimize(self, task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt:
        """Run all optimizers and select best."""
        logging.info("Starting Ensemble Hybrid optimization")
        
        # Split dataset for fair evaluation
        train_set, val_set = dataset.split(train_ratio=0.8)
        
        results = {}
        for name, optimizer in self.optimizers.items():
            logging.info(f"Running {name} optimizer")
            try:
                # Optimize on training set
                optimized = optimizer.optimize(task_spec, train_set)
                
                # Evaluate on validation set
                metrics = optimizer.evaluate(optimized, val_set)
                
                results[name] = {
                    "prompt": optimized,
                    "metrics": metrics,
                    "score": metrics.primary_score
                }
            except Exception as e:
                logging.error(f"Error in {name} optimizer: {e}")
                continue
        
        # Select best performer
        if not results:
            raise RuntimeError("All optimizers failed")
        
        best_name = max(results.keys(), key=lambda k: results[k]["score"])
        best_result = results[best_name]["prompt"]
        
        # Add ensemble metadata
        best_result.metadata.update({
            "optimizer": "ensemble_hybrid",
            "selected_optimizer": best_name,
            "all_scores": {k: v["score"] for k, v in results.items()},
            "total_cost": self.llm_client.get_cost_summary()["total_cost"]
        })
        
        return best_result
    
    def evaluate(self, prompt: OptimizedPrompt, test_set: Dataset) -> Metrics:
        """Evaluate prompt on test set."""
        # Use the optimizer that created this prompt
        optimizer_name = prompt.metadata.get("selected_optimizer", "grpo")
        return self.optimizers[optimizer_name].evaluate(prompt, test_set)


class FeedbackHybrid(BaseOptimizer):
    """Use GRPO tournament results to inform DSPy optimization iteratively."""
    
    def __init__(self, llm_client: BaseLLMClient,
                 judge_client: Optional[BaseLLMClient] = None,
                 config: Optional[HybridConfig] = None):
        super().__init__()
        self.llm_client = llm_client
        self.judge_client = judge_client or llm_client
        self.config = config or HybridConfig(strategy="feedback", max_iterations=3)
        
        self.dspy_optimizer = DSPyAdapter(llm_client, self.config.dspy_config)
        self.grpo_optimizer = GRPOAdapter(llm_client, judge_client, self.config.grpo_config)
    
    def optimize(self, task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt:
        """Iterative optimization with feedback loop."""
        logging.info("Starting Feedback Hybrid optimization")
        
        best_prompt = None
        best_score = 0.0
        winning_examples = []
        
        for iteration in range(self.config.max_iterations):
            logging.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Create enhanced dataset with winning examples
            if winning_examples:
                enhanced_dataset = self._create_enhanced_dataset(dataset, winning_examples)
            else:
                enhanced_dataset = dataset
            
            # DSPy optimization with current best examples
            dspy_result = self.dspy_optimizer.optimize(task_spec, enhanced_dataset)
            
            # Generate variations for tournament
            candidates = self._generate_candidates_from_prompt(dspy_result, task_spec, dataset)
            
            # Run tournament to evaluate candidates
            tournament_results = self._run_mini_tournament(candidates, dataset, task_spec)
            
            # Extract winning patterns
            new_winning_examples = self._extract_winning_patterns(
                tournament_results, candidates, dataset
            )
            winning_examples.extend(new_winning_examples)
            
            # Update best prompt
            if tournament_results["best_score"] > best_score:
                best_score = tournament_results["best_score"]
                best_prompt = tournament_results["best_prompt"]
            
            # Early stopping if converged
            if iteration > 0 and tournament_results["best_score"] - best_score < 0.01:
                logging.info("Converged early")
                break
        
        # Final optimization with all winning examples
        if winning_examples:
            final_dataset = self._create_enhanced_dataset(dataset, winning_examples)
            final_result = self.dspy_optimizer.optimize(task_spec, final_dataset)
        else:
            final_result = best_prompt or dspy_result
        
        final_result.metadata.update({
            "optimizer": "feedback_hybrid",
            "iterations": iteration + 1,
            "winning_examples_count": len(winning_examples),
            "total_cost": self.llm_client.get_cost_summary()["total_cost"]
        })
        
        return final_result
    
    def _create_enhanced_dataset(self, original: Dataset, 
                               winning_examples: List[Example]) -> Dataset:
        """Create dataset enhanced with winning examples."""
        # Combine original and winning examples, prioritizing winners
        all_examples = winning_examples + original.examples
        
        # Remove duplicates while preserving order
        seen = set()
        unique_examples = []
        for ex in all_examples:
            key = (ex.input, ex.output)
            if key not in seen:
                seen.add(key)
                unique_examples.append(ex)
        
        return Dataset(
            examples=unique_examples[:len(original.examples)],
            name=f"{original.name}_enhanced",
            metadata={**original.metadata, "enhanced": True}
        )
    
    def _generate_candidates_from_prompt(self, prompt: OptimizedPrompt,
                                       task_spec: TaskSpec, 
                                       dataset: Dataset) -> List[PromptCandidate]:
        """Generate candidate variations from a prompt."""
        candidates = []
        
        # Base candidate
        candidates.append(PromptCandidate(
            id="base",
            text=prompt.text,
            examples=prompt.examples,
            metadata={"variation": "original"}
        ))
        
        # Variations with different examples
        for i in range(4):
            examples = self._select_diverse_examples(dataset, 3 + i)
            candidates.append(PromptCandidate(
                id=f"var_ex_{i}",
                text=prompt.text,
                examples=examples,
                metadata={"variation": f"examples_{3+i}"}
            ))
        
        # Text variations
        for i in range(3):
            varied_text = self._create_text_variation(prompt.text, i)
            candidates.append(PromptCandidate(
                id=f"var_text_{i}",
                text=varied_text,
                examples=prompt.examples[:4],
                metadata={"variation": f"text_{i}"}
            ))
        
        return candidates
    
    def _create_text_variation(self, text: str, variation_num: int) -> str:
        """Create text variation based on strategy."""
        if variation_num == 0:
            # Add emphasis
            return text + "\n\nPay special attention to format requirements and constraints."
        elif variation_num == 1:
            # Simplify
            lines = text.split('\n')
            return '\n'.join(lines[:2]) if len(lines) > 2 else text
        else:
            # Rephrase
            return self._rephrase_prompt(text)
    
    def _run_mini_tournament(self, candidates: List[PromptCandidate],
                           dataset: Dataset, task_spec: TaskSpec) -> Dict[str, Any]:
        """Run a mini tournament on candidates."""
        # Use a subset of data for efficiency
        test_samples = dataset.examples[:min(5, len(dataset.examples))]
        
        # Simple scoring based on constraint adherence
        scores = {}
        for candidate in candidates:
            score = 0.0
            for test in test_samples:
                # Generate response
                response = self._generate_with_candidate(candidate, test)
                
                # Validate constraints
                if task_spec.constraints:
                    validation = task_spec.validate_response(response)
                    score += validation.overall_score
                else:
                    score += 1.0
            
            scores[candidate.id] = score / len(test_samples)
        
        # Find best
        best_id = max(scores.keys(), key=lambda k: scores[k])
        best_candidate = next(c for c in candidates if c.id == best_id)
        
        return {
            "best_prompt": self._candidate_to_prompt(best_candidate),
            "best_score": scores[best_id],
            "all_scores": scores,
            "best_candidate": best_candidate
        }
    
    def _generate_with_candidate(self, candidate: PromptCandidate, 
                               test_case: Example) -> str:
        """Generate response using candidate."""
        prompt = candidate.text + "\n\nExamples:\n"
        for ex in candidate.examples[:3]:
            prompt += f"\nInput: {ex.input}\nOutput: {ex.output}\n"
        prompt += f"\nInput: {test_case.input}\nOutput:"
        
        response = self.llm_client.generate(prompt, temperature=0.3, max_tokens=200)
        return response.text
    
    def _extract_winning_patterns(self, tournament_results: Dict[str, Any],
                                candidates: List[PromptCandidate],
                                dataset: Dataset) -> List[Example]:
        """Extract examples that led to winning performance."""
        best_candidate = tournament_results["best_candidate"]
        
        # Return the examples from the winning candidate
        return best_candidate.examples[:2]  # Take top 2 examples
    
    def _candidate_to_prompt(self, candidate: PromptCandidate) -> OptimizedPrompt:
        """Convert candidate to OptimizedPrompt."""
        return OptimizedPrompt(
            text=candidate.text,
            examples=candidate.examples,
            metadata=candidate.metadata
        )
    
    def _select_diverse_examples(self, dataset: Dataset, count: int) -> List[Example]:
        """Select diverse examples."""
        if len(dataset) <= count:
            return dataset.examples
        
        import random
        return random.sample(dataset.examples, count)
    
    def _rephrase_prompt(self, text: str) -> str:
        """Rephrase using LLM."""
        prompt = f"Rephrase more concisely:\n{text}\n\nConcise version:"
        response = self.llm_client.generate(prompt, temperature=0.7, max_tokens=150)
        return response.text.strip()
    
    def evaluate(self, prompt: OptimizedPrompt, test_set: Dataset) -> Metrics:
        """Evaluate prompt."""
        return self.grpo_optimizer.evaluate(prompt, test_set)


class CostAwareHybrid(BaseOptimizer):
    """Optimize for cost-effectiveness in business settings."""
    
    def __init__(self, llm_client: BaseLLMClient,
                 budget_limit: float = 10.0,
                 config: Optional[HybridConfig] = None):
        super().__init__()
        self.llm_client = llm_client
        self.budget_limit = budget_limit
        self.config = config or HybridConfig(strategy="cost_aware", budget_limit=budget_limit)
        
        # Use lightweight configs for cost efficiency
        dspy_config = DSPyConfig(
            max_bootstrapped_demos=2,
            max_labeled_demos=8,
            max_rounds=5
        )
        grpo_config = GRPOConfig(
            num_candidates=4,
            tournament_rounds=2
        )
        
        self.dspy_optimizer = DSPyAdapter(llm_client, dspy_config)
        self.grpo_optimizer = GRPOAdapter(llm_client, config=grpo_config)
    
    def optimize(self, task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt:
        """Optimize within budget constraints."""
        logging.info(f"Starting Cost-Aware Hybrid optimization with budget ${self.budget_limit}")
        
        initial_cost = self.llm_client.get_cost_summary()["total_cost"]
        remaining_budget = self.budget_limit
        
        # Phase 1: Quick DSPy optimization (typically cheaper)
        logging.info("Phase 1: Cost-efficient DSPy optimization")
        dspy_result = self.dspy_optimizer.optimize(task_spec, dataset)
        
        current_cost = self.llm_client.get_cost_summary()["total_cost"] - initial_cost
        remaining_budget -= current_cost
        logging.info(f"DSPy phase cost: ${current_cost:.2f}, remaining: ${remaining_budget:.2f}")
        
        # Phase 2: GRPO only if budget allows
        if remaining_budget > 2.0:  # Need at least $2 for meaningful GRPO
            logging.info("Phase 2: Budget-limited GRPO refinement")
            
            # Adjust GRPO config based on remaining budget
            adjusted_config = GRPOConfig(
                num_candidates=min(4, int(remaining_budget / 0.5)),
                tournament_rounds=1,
                mutation_rate=0.2
            )
            
            self.grpo_optimizer.config = adjusted_config
            
            # Create limited candidate set from DSPy result
            candidates = [
                PromptCandidate(
                    id="dspy_result",
                    text=dspy_result.text,
                    examples=dspy_result.examples
                )
            ]
            
            # Add a few variations within budget
            for i in range(min(2, int(remaining_budget / 0.3))):
                candidates.append(PromptCandidate(
                    id=f"budget_var_{i}",
                    text=self._create_budget_variation(dspy_result.text, i),
                    examples=dspy_result.examples[:3]
                ))
            
            original_init = self.grpo_optimizer._initialize_population
            self.grpo_optimizer._initialize_population = lambda ts, ds: candidates
            
            try:
                final_result = self.grpo_optimizer.optimize(task_spec, dataset)
            finally:
                self.grpo_optimizer._initialize_population = original_init
        else:
            logging.info("Insufficient budget for GRPO phase, using DSPy result")
            final_result = dspy_result
        
        total_cost = self.llm_client.get_cost_summary()["total_cost"] - initial_cost
        
        final_result.metadata.update({
            "optimizer": "cost_aware_hybrid",
            "total_cost": total_cost,
            "budget_limit": self.budget_limit,
            "budget_used_percentage": (total_cost / self.budget_limit) * 100
        })
        
        return final_result
    
    def _create_budget_variation(self, text: str, variation_num: int) -> str:
        """Create simple text variations without LLM calls."""
        if variation_num == 0:
            # Add clarity
            return text + "\n\nBe clear and precise in your response."
        else:
            # Simplify
            lines = text.split('\n')
            return '\n'.join([line for line in lines if line.strip()])
    
    def evaluate(self, prompt: OptimizedPrompt, test_set: Dataset) -> Metrics:
        """Cost-aware evaluation."""
        # Use smaller test sample to reduce cost
        small_test = Dataset(
            examples=test_set.examples[:min(5, len(test_set))],
            name=f"{test_set.name}_small"
        )
        return self.dspy_optimizer.evaluate(prompt, small_test)


class EnterpriseSequentialHybrid(SequentialHybrid):
    """Enterprise-focused sequential hybrid with business constraints."""
    
    def __init__(self, llm_client: BaseLLMClient,
                 business_context: BusinessContext,
                 judge_client: Optional[BaseLLMClient] = None,
                 config: Optional[HybridConfig] = None):
        super().__init__(llm_client, judge_client, config)
        self.business_context = business_context
    
    def optimize_for_enterprise(self, task_spec: TaskSpec,
                              synthetic_dataset: Dataset) -> OptimizedPrompt:
        """Optimize with business constraints."""
        # Enhance task spec with business constraints
        enhanced_spec = self._add_business_constraints(task_spec)
        
        # Run optimization
        result = self.optimize(enhanced_spec, synthetic_dataset)
        
        # Add business metadata
        result.metadata.update({
            "business_context": self.business_context.to_dict(),
            "enterprise_optimized": True
        })
        
        return result
    
    def _add_business_constraints(self, task_spec: TaskSpec) -> TaskSpec:
        """Add business-specific constraints."""
        from ..core.constraints import create_tone_constraint, create_business_constraint
        
        enhanced = TaskSpec(
            name=task_spec.name,
            description=task_spec.description,
            input_format=task_spec.input_format,
            output_format=task_spec.output_format,
            constraints=list(task_spec.constraints),
            examples=task_spec.examples
        )
        
        # Add brand voice
        if self.business_context.brand_voice:
            enhanced.constraints.append(
                create_tone_constraint(self.business_context.brand_voice, weight=0.4)
            )
        
        # Add compliance constraints
        for compliance in self.business_context.compliance_requirements or []:
            enhanced.constraints.append(
                create_business_constraint("compliance", weight=0.8)
            )
        
        return enhanced
    
    def create_deployment_package(self, optimized_prompt: OptimizedPrompt) -> Dict[str, Any]:
        """Create comprehensive deployment package."""
        return {
            "prompt_template": self._format_for_production(optimized_prompt),
            "integration_guides": {
                "chatgpt": self._create_chatgpt_guide(optimized_prompt),
                "api": self._create_api_guide(optimized_prompt),
                "slack": self._create_slack_guide(optimized_prompt)
            },
            "training_materials": self._create_training_materials(optimized_prompt),
            "monitoring_setup": self._create_monitoring_config(optimized_prompt),
            "rollout_plan": self._create_rollout_plan()
        }
    
    def _format_for_production(self, prompt: OptimizedPrompt) -> str:
        """Format for production use."""
        return f"""# {self.business_context.use_case} Prompt Template

## Instructions
{prompt.text}

## Brand Voice
{self.business_context.brand_voice}

## Compliance
{', '.join(self.business_context.compliance_requirements or [])}

## Examples
{self._format_examples(prompt.examples[:3])}
"""
    
    def _format_examples(self, examples: List[Example]) -> str:
        """Format examples nicely."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"{i}. Input: {ex.input}\n   Output: {ex.output}")
        return '\n'.join(formatted)
    
    def _create_chatgpt_guide(self, prompt: OptimizedPrompt) -> str:
        """Create ChatGPT integration guide."""
        return f"""ChatGPT Integration Guide

1. Copy this system prompt:
{prompt.text}

2. Set up Custom Instructions in ChatGPT settings
3. Test with these examples:
{self._format_examples(prompt.examples[:2])}
"""
    
    def _create_api_guide(self, prompt: OptimizedPrompt) -> str:
        """Create API integration guide."""
        return f"""API Integration Guide

```python
import openai

system_prompt = '''{prompt.text}'''

def generate_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": user_input}}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content
```
"""
    
    def _create_slack_guide(self, prompt: OptimizedPrompt) -> str:
        """Create Slack bot guide."""
        return "Slack Bot Integration:\n1. Set up webhook\n2. Configure bot with prompt\n3. Test in sandbox channel"
    
    def _create_training_materials(self, prompt: OptimizedPrompt) -> str:
        """Create training materials."""
        return f"""Team Training Guide

Objective: Standardize {self.business_context.use_case} responses

Key Points:
- Always maintain {self.business_context.brand_voice} tone
- Follow the provided examples
- Ensure {', '.join(self.business_context.compliance_requirements or [])} compliance

Practice Scenarios:
{self._format_examples(prompt.examples[:5])}
"""
    
    def _create_monitoring_config(self, prompt: OptimizedPrompt) -> Dict[str, Any]:
        """Create monitoring configuration."""
        return {
            "metrics_to_track": [
                "response_quality_score",
                "constraint_adherence_rate", 
                "average_response_time",
                "cost_per_response"
            ],
            "alert_thresholds": {
                "quality_score": 0.8,
                "constraint_adherence": 0.9,
                "cost_per_response": 0.05
            },
            "reporting_frequency": "weekly"
        }
    
    def _create_rollout_plan(self) -> Dict[str, Any]:
        """Create phased rollout plan."""
        return {
            "phases": [
                {
                    "phase": 1,
                    "description": "Pilot with small team",
                    "duration": "1 week",
                    "success_criteria": "90% satisfaction"
                },
                {
                    "phase": 2,
                    "description": "Expand to department",
                    "duration": "2 weeks",
                    "success_criteria": "85% adoption rate"
                },
                {
                    "phase": 3,
                    "description": "Full rollout",
                    "duration": "ongoing",
                    "success_criteria": "ROI > 200%"
                }
            ]
        }


def create_hybrid_optimizer(strategy: str, llm_client: BaseLLMClient,
                          judge_client: Optional[BaseLLMClient] = None,
                          config: Optional[HybridConfig] = None) -> BaseOptimizer:
    """Factory function to create hybrid optimizers."""
    strategies = {
        "sequential": SequentialHybrid,
        "ensemble": EnsembleHybrid,
        "feedback": FeedbackHybrid,
        "cost_aware": CostAwareHybrid
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(strategies.keys())}")
    
    optimizer_class = strategies[strategy]
    
    if strategy == "cost_aware":
        return optimizer_class(llm_client, budget_limit=config.budget_limit if config else 10.0, config=config)
    else:
        return optimizer_class(llm_client, judge_client, config)