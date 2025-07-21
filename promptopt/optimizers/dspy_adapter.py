"""DSPy adapter for prompt optimization."""

from typing import Dict, Any, List, Optional, Callable
import logging
from dataclasses import dataclass
import json

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available. Install with: pip install dspy-ai")

from ..core.base import BaseOptimizer, Dataset, TaskSpec, OptimizedPrompt, Metrics, Example
from ..core.metrics import AccuracyMetric, ConstraintAdherenceMetric
from ..core.interfaces import BusinessContext, OptimizationConfig
from ..utils.llm_clients import BaseLLMClient, create_llm_client


@dataclass
class DSPyConfig:
    """Configuration for DSPy optimization."""
    optimizer_type: str = "BootstrapFewShot"  # or "MIPROv2", "BootstrapFewShotWithRandomSearch"
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    max_rounds: int = 10
    num_candidates: int = 10
    temperature: float = 0.7
    metric_threshold: float = 0.8


if DSPY_AVAILABLE:
    class DSPySignature(dspy.Signature):
        """Dynamic signature for DSPy based on TaskSpec."""
        def __init__(self, task_spec: TaskSpec):
            self.task_spec = task_spec
            # Dynamically create input and output fields
            self.__doc__ = task_spec.description


    class DSPyModule(dspy.Module):
        """DSPy module wrapper for prompt optimization."""
        
        def __init__(self, task_spec: TaskSpec):
            super().__init__()
            self.task_spec = task_spec
            self.predictor = dspy.ChainOfThought(DSPySignature(task_spec))
        
        def forward(self, **kwargs):
            """Forward pass through the module."""
            return self.predictor(**kwargs)
else:
    # Placeholder classes when DSPy is not available
    class DSPySignature:
        pass
    
    class DSPyModule:
        pass


class DSPyAdapter(BaseOptimizer):
    """Adapter for DSPy optimization framework."""
    
    def __init__(self, llm_client: BaseLLMClient, config: Optional[DSPyConfig] = None):
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not installed. Install with: pip install dspy-ai")
        
        super().__init__()
        self.llm_client = llm_client
        self.config = config or DSPyConfig()
        self._setup_dspy_lm()
    
    def _setup_dspy_lm(self):
        """Set up DSPy language model."""
        # Configure DSPy to use our LLM client
        if isinstance(self.llm_client, type(self.llm_client).__bases__[0]):  # Check if it's OpenAI
            dspy.settings.configure(
                lm=dspy.OpenAI(
                    model=self.llm_client.model,
                    api_key=self.llm_client.api_key,
                    temperature=self.config.temperature
                )
            )
    
    def optimize(self, task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt:
        """Optimize prompt using DSPy."""
        # Convert dataset to DSPy format
        train_examples = self._convert_to_dspy_examples(dataset.examples)
        
        # Create DSPy module
        module = DSPyModule(task_spec)
        
        # Create metric function
        metric_fn = self._create_metric_function(task_spec)
        
        # Create optimizer based on config
        optimizer = self._create_optimizer(metric_fn)
        
        # Compile the module
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            metric=metric_fn
        )
        
        # Extract optimized prompt
        optimized_prompt = self._extract_optimized_prompt(optimized_module, task_spec)
        
        # Track optimization cost
        total_cost = self.llm_client.get_cost_summary()["total_cost"]
        optimized_prompt.metadata["optimization_cost"] = total_cost
        
        return optimized_prompt
    
    def _convert_to_dspy_examples(self, examples: List[Example]) -> List[Any]:
        """Convert our examples to DSPy format."""
        dspy_examples = []
        for ex in examples:
            # Create DSPy example with dynamic fields
            dspy_ex = dspy.Example(
                input=ex.input,
                output=ex.output
            ).with_inputs("input")
            dspy_examples.append(dspy_ex)
        return dspy_examples
    
    def _create_optimizer(self, metric_fn: Callable) -> Any:
        """Create DSPy optimizer based on configuration."""
        if self.config.optimizer_type == "BootstrapFewShot":
            return BootstrapFewShot(
                metric=metric_fn,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                max_rounds=self.config.max_rounds
            )
        
        elif self.config.optimizer_type == "BootstrapFewShotWithRandomSearch":
            return BootstrapFewShotWithRandomSearch(
                metric=metric_fn,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                num_candidate_programs=self.config.num_candidates
            )
        
        elif self.config.optimizer_type == "MIPROv2":
            return MIPROv2(
                metric=metric_fn,
                num_candidates=self.config.num_candidates,
                init_temperature=self.config.temperature
            )
        
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_metric_function(self, task_spec: TaskSpec) -> Callable:
        """Create metric function for DSPy optimization."""
        def metric(example, prediction, trace=None):
            # Basic accuracy check
            if prediction.output == example.output:
                accuracy_score = 1.0
            else:
                # Fuzzy matching for partial credit
                pred_tokens = set(prediction.output.lower().split())
                true_tokens = set(example.output.lower().split())
                if pred_tokens and true_tokens:
                    overlap = len(pred_tokens & true_tokens)
                    accuracy_score = overlap / max(len(pred_tokens), len(true_tokens))
                else:
                    accuracy_score = 0.0
            
            # Constraint checking
            if task_spec.constraints:
                validation_result = task_spec.validate_response(prediction.output)
                constraint_score = validation_result.overall_score
            else:
                constraint_score = 1.0
            
            # Combine scores
            final_score = 0.7 * accuracy_score + 0.3 * constraint_score
            
            return final_score >= self.config.metric_threshold
        
        return metric
    
    def _extract_optimized_prompt(self, optimized_module: Any, 
                                 task_spec: TaskSpec) -> OptimizedPrompt:
        """Extract optimized prompt from DSPy module."""
        # Get the prompt from the optimized module
        if hasattr(optimized_module, 'predictor') and hasattr(optimized_module.predictor, 'demos'):
            examples = []
            for demo in optimized_module.predictor.demos:
                examples.append(Example(
                    input=demo.get('input', ''),
                    output=demo.get('output', '')
                ))
        else:
            examples = []
        
        # Extract the instruction prompt
        prompt_text = self._extract_prompt_text(optimized_module, task_spec)
        
        return OptimizedPrompt(
            text=prompt_text,
            examples=examples,
            metadata={
                "optimizer": "dspy",
                "optimizer_type": self.config.optimizer_type,
                "num_examples": len(examples),
                "task": task_spec.name
            }
        )
    
    def _extract_prompt_text(self, module: Any, task_spec: TaskSpec) -> str:
        """Extract the actual prompt text from DSPy module."""
        # This is a simplified extraction - DSPy modules can be complex
        prompt_parts = []
        
        # Add task description
        prompt_parts.append(f"Task: {task_spec.description}")
        
        # Add input/output format
        prompt_parts.append(f"\nInput format: {json.dumps(task_spec.input_format)}")
        prompt_parts.append(f"Output format: {json.dumps(task_spec.output_format)}")
        
        # Add constraints if any
        if task_spec.constraints:
            prompt_parts.append("\nConstraints:")
            for constraint in task_spec.constraints:
                prompt_parts.append(f"- {constraint.description}")
        
        return "\n".join(prompt_parts)
    
    def evaluate(self, prompt: OptimizedPrompt, test_set: Dataset) -> Metrics:
        """Evaluate optimized prompt on test set."""
        predictions = []
        costs = []
        
        for example in test_set.examples:
            # Format prompt with example
            full_prompt = self._format_prompt_with_example(prompt, example)
            
            # Get prediction
            response = self.llm_client.generate(full_prompt, temperature=0.1)
            predictions.append(response.text)
            costs.append(response.cost)
        
        # Calculate metrics
        ground_truth = [ex.output for ex in test_set.examples]
        
        accuracy_metric = AccuracyMetric()
        accuracy_score = accuracy_metric.compute(predictions, ground_truth)
        
        scores = {
            "accuracy": accuracy_score,
            "average_cost": sum(costs) / len(costs) if costs else 0
        }
        
        # Add constraint adherence if applicable
        if hasattr(self, 'task_spec') and self.task_spec.constraints:
            constraint_metric = ConstraintAdherenceMetric(self.task_spec.constraints)
            scores["constraint_adherence"] = constraint_metric.compute(predictions, ground_truth)
        
        return Metrics(
            scores=scores,
            metadata={
                "test_set_size": len(test_set),
                "total_cost": sum(costs)
            }
        )
    
    def _format_prompt_with_example(self, prompt: OptimizedPrompt, 
                                   example: Example) -> str:
        """Format prompt with few-shot examples and test input."""
        formatted_parts = [prompt.text]
        
        # Add few-shot examples
        if prompt.examples:
            formatted_parts.append("\nExamples:")
            for i, ex in enumerate(prompt.examples, 1):
                formatted_parts.append(f"\nExample {i}:")
                formatted_parts.append(f"Input: {ex.input}")
                formatted_parts.append(f"Output: {ex.output}")
        
        # Add test input
        formatted_parts.append(f"\nNow process this input:")
        formatted_parts.append(f"Input: {example.input}")
        formatted_parts.append("Output:")
        
        return "\n".join(formatted_parts)


class EnterpriseDSPyAdapter(DSPyAdapter):
    """Enterprise-ready DSPy adapter with business constraints."""
    
    def __init__(self, llm_client: BaseLLMClient, 
                 business_context: BusinessContext,
                 config: Optional[DSPyConfig] = None):
        super().__init__(llm_client, config)
        self.business_context = business_context
    
    def optimize_for_business_use(self, task_spec: TaskSpec, 
                                 synthetic_dataset: Dataset) -> OptimizedPrompt:
        """Optimize with business constraints."""
        # Add business-specific constraints to task spec
        enhanced_task_spec = self._enhance_with_business_constraints(task_spec)
        
        # Run optimization
        optimized_prompt = self.optimize(enhanced_task_spec, synthetic_dataset)
        
        # Add business metadata
        optimized_prompt.metadata.update({
            "industry": self.business_context.industry,
            "company_size": self.business_context.company_size,
            "use_case": self.business_context.use_case,
            "compliance": self.business_context.compliance_requirements
        })
        
        return optimized_prompt
    
    def _enhance_with_business_constraints(self, task_spec: TaskSpec) -> TaskSpec:
        """Add business-specific constraints to task spec."""
        from ..core.constraints import create_tone_constraint, create_business_constraint
        
        # Copy existing task spec
        enhanced = TaskSpec(
            name=task_spec.name,
            description=task_spec.description,
            input_format=task_spec.input_format,
            output_format=task_spec.output_format,
            constraints=list(task_spec.constraints),
            examples=task_spec.examples,
            metadata=dict(task_spec.metadata)
        )
        
        # Add brand voice constraint
        if self.business_context.brand_voice:
            enhanced.constraints.append(
                create_tone_constraint(self.business_context.brand_voice)
            )
        
        # Add compliance constraints
        for compliance in self.business_context.compliance_requirements or []:
            if compliance == "SOC2":
                enhanced.constraints.append(
                    create_business_constraint("compliance", weight=0.9)
                )
        
        return enhanced
    
    def create_team_deployment_assets(self, optimized_prompt: OptimizedPrompt) -> Dict[str, Any]:
        """Create deployment assets for team use."""
        return {
            "chatgpt_template": self._format_for_chatgpt(optimized_prompt),
            "slack_template": self._format_for_slack(optimized_prompt),
            "reference_card": self._create_quick_reference(optimized_prompt),
            "training_materials": self._create_training_guide(optimized_prompt)
        }
    
    def _format_for_chatgpt(self, prompt: OptimizedPrompt) -> str:
        """Format optimized prompt for ChatGPT usage."""
        template = f"""System Prompt:
{prompt.text}

Instructions:
1. Follow the format specified above
2. Maintain {self.business_context.brand_voice} tone
3. Ensure compliance with {', '.join(self.business_context.compliance_requirements or [])}

Examples to follow:
"""
        
        for i, example in enumerate(prompt.examples[:3], 1):
            template += f"\nExample {i}:\nInput: {example.input}\nOutput: {example.output}\n"
        
        return template
    
    def _format_for_slack(self, prompt: OptimizedPrompt) -> str:
        """Format for Slack bot implementation."""
        return f"""Slack Bot Response Template:

{prompt.text}

Quick responses for common scenarios:
""" + "\n".join([f"- {ex.input[:50]}... → {ex.output[:50]}..." 
                 for ex in prompt.examples[:3]])
    
    def _create_quick_reference(self, prompt: OptimizedPrompt) -> Dict[str, Any]:
        """Create quick reference card."""
        return {
            "purpose": prompt.metadata.get("task", "General"),
            "key_points": [
                "Use provided examples as templates",
                f"Maintain {self.business_context.brand_voice} tone",
                "Follow output format strictly"
            ],
            "examples": [{"input": ex.input, "output": ex.output} 
                        for ex in prompt.examples[:2]]
        }
    
    def _create_training_guide(self, prompt: OptimizedPrompt) -> str:
        """Create training guide for team."""
        return f"""Team Training Guide: {prompt.metadata.get('task', 'Optimized Prompt')}

Overview:
This prompt has been optimized for {self.business_context.use_case} in the {self.business_context.industry} industry.

Key Components:
1. Base Instructions: {prompt.text[:200]}...

2. Tone: {self.business_context.brand_voice}

3. Compliance: {', '.join(self.business_context.compliance_requirements or ['None'])}

Practice Examples:
""" + "\n".join([f"{i}. Input: {ex.input}\n   Expected Output: {ex.output}\n" 
                 for i, ex in enumerate(prompt.examples[:3], 1)])