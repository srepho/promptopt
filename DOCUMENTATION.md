# PromptOpt Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Optimizers](#optimizers)
6. [Synthetic Data Generation](#synthetic-data-generation)
7. [Cost Tracking](#cost-tracking)
8. [Enterprise Features](#enterprise-features)
9. [Colab Integration](#colab-integration)
10. [API Reference](#api-reference)

## Overview

PromptOpt is an enterprise-ready prompt optimization framework that combines DSPy and GRPO approaches to help teams standardize and optimize their LLM prompts. It's designed to work entirely through API calls, requiring no GPU resources.

### Key Features
- 🤖 Multiple optimization strategies (DSPy, GRPO, Hybrid)
- 💰 Built-in cost tracking and budget management
- 🏢 Enterprise-focused with compliance support
- 📊 Tournament-based evaluation system
- 🎯 Synthetic data generation for testing
- ☁️ Optimized for Google Colab environments

## Installation

### Using Conda (Recommended)
```bash
conda create -n promptopt python=3.9
conda activate promptopt
pip install promptopt
```

### Using pip
```bash
pip install promptopt
```

### Development Installation
```bash
git clone https://github.com/promptopt/promptopt
cd promptopt
pip install -e ".[dev]"
```

## Quick Start

```python
from promptopt import EnterprisePOC
from promptopt.core import TaskSpec, Dataset, Example
from promptopt.optimizers import create_hybrid_optimizer
from promptopt.utils import create_llm_client

# Set up LLM client
llm_client = create_llm_client("openai", "gpt-3.5-turbo")

# Define your task
task_spec = TaskSpec(
    name="customer_support",
    description="Generate helpful customer support responses",
    input_format={"query": "string"},
    output_format={"response": "string"}
)

# Create dataset
dataset = Dataset(examples=[
    Example(
        input="How do I reset my password?",
        output="To reset your password, click 'Forgot Password' on the login page."
    )
])

# Run optimization
optimizer = create_hybrid_optimizer("sequential", llm_client)
optimized_prompt = optimizer.optimize(task_spec, dataset)

print(f"Optimized prompt: {optimized_prompt.text}")
print(f"Cost: ${optimized_prompt.metadata['optimization_cost']:.2f}")
```

## Core Concepts

### TaskSpec
Defines the optimization task with constraints:

```python
from promptopt.core import TaskSpec, create_conciseness_constraint

task_spec = TaskSpec(
    name="email_writer",
    description="Write professional emails",
    input_format={"subject": "str", "context": "str"},
    output_format={"email": "str"},
    constraints=[
        create_conciseness_constraint(max_words=150),
        create_tone_constraint("professional")
    ]
)
```

### Dataset
Container for training/evaluation examples:

```python
from promptopt.core import Dataset, Example

dataset = Dataset(
    examples=[
        Example(input="input text", output="expected output"),
        # ... more examples
    ],
    name="training_data"
)

# Split for training/validation
train_set, val_set = dataset.split(train_ratio=0.8)
```

### Constraints
Programmatic requirements for responses:

```python
from promptopt.core import create_format_constraint, create_business_constraint

constraints = [
    create_format_constraint("Answer: [YES/NO]", weight=0.8),
    create_business_constraint("compliance", weight=0.9),
    create_no_pii_constraint(weight=1.0)
]
```

## Optimizers

### DSPy Adapter
Integrates with the DSPy framework for few-shot learning:

```python
from promptopt.optimizers import DSPyAdapter, DSPyConfig

config = DSPyConfig(
    optimizer_type="BootstrapFewShot",
    max_bootstrapped_demos=4,
    max_labeled_demos=16
)

optimizer = DSPyAdapter(llm_client, config)
result = optimizer.optimize(task_spec, dataset)
```

### GRPO Adapter
Tournament-based genetic optimization:

```python
from promptopt.optimizers import GRPOAdapter, GRPOConfig

config = GRPOConfig(
    num_candidates=8,
    tournament_rounds=3,
    mutation_rate=0.3
)

optimizer = GRPOAdapter(llm_client, config=config)
result = optimizer.optimize(task_spec, dataset)
```

### Hybrid Strategies

#### Sequential Hybrid
DSPy optimization followed by GRPO refinement:

```python
optimizer = create_hybrid_optimizer("sequential", llm_client)
```

#### Ensemble Hybrid
Run multiple optimizers and select best:

```python
optimizer = create_hybrid_optimizer("ensemble", llm_client)
```

#### Feedback Hybrid
Iterative optimization with feedback loop:

```python
optimizer = create_hybrid_optimizer("feedback", llm_client)
```

#### Cost-Aware Hybrid
Optimize within budget constraints:

```python
from promptopt.optimizers import HybridConfig

config = HybridConfig(budget_limit=10.0)
optimizer = create_hybrid_optimizer("cost_aware", llm_client, config=config)
```

## Synthetic Data Generation

Generate realistic business scenarios for testing:

```python
from promptopt.data import EnterpriseDataGenerator
from promptopt.core import BusinessContext

# Define business context
context = BusinessContext(
    industry="technology",
    company_size="enterprise",
    use_case="customer_support",
    compliance_requirements=["SOC2", "GDPR"]
)

# Generate synthetic data
generator = EnterpriseDataGenerator(llm_client)
dataset = generator.create_customer_support_data(
    company_context=context.to_dict(),
    count=100
)

# Validate data quality
from promptopt.data import DataQualityValidator

validator = DataQualityValidator()
results = validator.validate_dataset(dataset)
print(f"Data quality score: {results['overall_score']:.2f}")
```

## Cost Tracking

All LLM API calls are automatically tracked:

```python
# Check optimization costs
print(f"Total cost: ${llm_client.get_cost_summary()['total_cost']:.2f}")

# Get detailed breakdown
summary = llm_client.get_cost_summary()
print(f"Requests: {summary['request_count']}")
print(f"Prompt tokens: {summary['total_prompt_tokens']}")
print(f"Average cost per request: ${summary['average_cost_per_request']:.4f}")
```

## Enterprise Features

### Business Context Integration
```python
from promptopt.optimizers import EnterpriseSequentialHybrid

optimizer = EnterpriseSequentialHybrid(
    llm_client=llm_client,
    business_context=business_context
)

# Optimize with business constraints
result = optimizer.optimize_for_enterprise(task_spec, dataset)

# Generate deployment package
deployment = optimizer.create_deployment_package(result)
```

### Deployment Assets
```python
# Get ready-to-use templates
print(deployment["prompt_template"])
print(deployment["integration_guides"]["chatgpt"])
print(deployment["training_materials"])
```

## Colab Integration

### Environment Setup
```python
from promptopt.colab import ColabEnvironment

env = ColabEnvironment()
env.setup()  # Mounts Drive, sets up directories, checks API keys
```

### Interactive Wizards
```python
from promptopt.colab import DataGenerationWizard, OptimizationWizard

# Data generation wizard
data_wizard = DataGenerationWizard(is_colab=True)
config = data_wizard.run()

# Optimization configuration
opt_wizard = OptimizationWizard(is_colab=True)
opt_config = opt_wizard.run()
```

### Results Sharing
```python
# Save to Google Drive
filepath = env.drive.save_results(results, "optimization_results")

# Create shareable exports
exports = env.sharing.export_for_team(optimized_prompt)
```

## API Reference

### Core Classes

#### TaskSpec
```python
TaskSpec(
    name: str,
    description: str,
    input_format: Dict[str, Any],
    output_format: Dict[str, Any],
    constraints: List[Constraint] = [],
    examples: Optional[List[Example]] = None
)
```

#### Dataset
```python
Dataset(
    examples: List[Example],
    name: str = "unnamed_dataset",
    metadata: Dict[str, Any] = {}
)
```

#### OptimizedPrompt
```python
OptimizedPrompt(
    text: str,
    examples: List[Example] = [],
    metadata: Dict[str, Any] = {},
    optimization_history: List[Dict] = []
)
```

### Optimizer Methods

All optimizers implement:
```python
def optimize(task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt
def evaluate(prompt: OptimizedPrompt, test_set: Dataset) -> Metrics
```

### Utility Functions

#### Create LLM Client
```python
create_llm_client(
    provider: str,  # "openai" or "anthropic"
    model: str,     # e.g., "gpt-4", "claude-3-haiku"
    **kwargs        # Additional parameters
) -> BaseLLMClient
```

#### Create Hybrid Optimizer
```python
create_hybrid_optimizer(
    strategy: str,  # "sequential", "ensemble", "feedback", "cost_aware"
    llm_client: BaseLLMClient,
    judge_client: Optional[BaseLLMClient] = None,
    config: Optional[HybridConfig] = None
) -> BaseOptimizer
```

## Best Practices

1. **Always use conda environments** for development
2. **Set budget limits** for production optimization
3. **Validate synthetic data** before using for optimization
4. **Test optimized prompts** with real data before deployment
5. **Monitor costs** continuously during optimization
6. **Use appropriate constraints** for your business needs
7. **Save results** to Drive when using Colab

## Troubleshooting

### No API Keys Found
```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

### DSPy Not Available
```bash
pip install dspy-ai
```

### Colab Drive Not Mounting
- Ensure you have permission to mount Drive
- Try restarting the runtime
- Check Google account permissions

### High Optimization Costs
- Use `cost_aware` hybrid strategy
- Reduce `num_candidates` in GRPO config
- Lower `max_iterations` in optimization config
- Use cheaper models (e.g., gpt-3.5-turbo instead of gpt-4)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.