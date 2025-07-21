"""Demo script comparing DSPy and GRPO optimizers."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptopt.core import (
    TaskSpec, Dataset, Example, BusinessContext,
    create_conciseness_constraint, create_format_constraint
)
from promptopt.optimizers import DSPyAdapter, GRPOAdapter, DSPyConfig, GRPOConfig
from promptopt.utils import create_llm_client, ResultsVisualizer
from promptopt.data import EnterpriseDataGenerator


def create_sample_task():
    """Create a sample task for demonstration."""
    task_spec = TaskSpec(
        name="customer_response",
        description="Generate professional customer support responses",
        input_format={"customer_query": "string", "sentiment": "string"},
        output_format={"response": "string", "tone": "string"},
        constraints=[
            create_conciseness_constraint(max_words=100, weight=0.3),
            create_format_constraint("Response: [TEXT]", weight=0.5)
        ]
    )
    
    # Create sample dataset
    examples = [
        Example(
            input="Customer query: My package hasn't arrived\nSentiment: frustrated",
            output="Response: I understand your frustration about the delayed package. Let me immediately check the tracking status for you and provide an update on its location and expected delivery date."
        ),
        Example(
            input="Customer query: How do I reset my password?\nSentiment: neutral",
            output="Response: I'd be happy to help you reset your password. Please click the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your inbox."
        ),
        Example(
            input="Customer query: Your product is amazing!\nSentiment: positive",
            output="Response: Thank you so much for your kind feedback! We're thrilled to hear you're enjoying our product. Is there anything specific you'd like to share about your experience?"
        )
    ]
    
    dataset = Dataset(examples=examples, name="customer_support_demo")
    return task_spec, dataset


def compare_optimizers_simple():
    """Simple comparison without LLM (using mock responses)."""
    print("=== Prompt Optimizer Comparison Demo ===\n")
    
    # Create task and dataset
    task_spec, dataset = create_sample_task()
    print(f"Task: {task_spec.name}")
    print(f"Dataset size: {len(dataset)} examples\n")
    
    # Since we don't have API keys, we'll demonstrate the structure
    print("Optimizer Comparison Structure:")
    print("1. DSPy Optimizer:")
    print("   - Uses few-shot learning with bootstrapping")
    print("   - Optimizes instruction and example selection")
    print("   - Supports multiple optimization strategies\n")
    
    print("2. GRPO Optimizer:")
    print("   - Uses tournament-based genetic optimization")
    print("   - Evolves prompts through mutation and crossover")
    print("   - Evaluates via head-to-head comparisons\n")
    
    # Show sample optimized prompt structure
    print("Sample Optimized Prompt Structure:")
    print("```")
    print("Task: Generate professional customer support responses")
    print("\nConstraints:")
    print("- Response must be under 100 words")
    print("- Response must follow format: Response: [TEXT]")
    print("\nExamples:")
    print("1. Input: Customer query: My package hasn't arrived\\nSentiment: frustrated")
    print("   Output: Response: I understand your frustration...")
    print("```")
    
    # Visualize theoretical results
    visualizer = ResultsVisualizer()
    
    # Mock results for visualization
    mock_results = {
        "DSPy (BootstrapFewShot)": {
            "accuracy": 0.85,
            "constraint_adherence": 0.92,
            "cost_efficiency": 0.78
        },
        "GRPO (Tournament)": {
            "accuracy": 0.82,
            "constraint_adherence": 0.95,
            "cost_efficiency": 0.85
        },
        "Baseline": {
            "accuracy": 0.65,
            "constraint_adherence": 0.70,
            "cost_efficiency": 0.90
        }
    }
    
    print("\n=== Theoretical Performance Comparison ===")
    for optimizer, metrics in mock_results.items():
        print(f"\n{optimizer}:")
        for metric, score in metrics.items():
            print(f"  - {metric}: {score:.2%}")


def demonstrate_enterprise_features():
    """Demonstrate enterprise-specific features."""
    print("\n=== Enterprise Features Demo ===\n")
    
    # Create business context
    business_context = BusinessContext(
        industry="technology",
        company_size="enterprise",
        use_case="customer_support",
        compliance_requirements=["SOC2", "GDPR"],
        brand_voice="professional yet friendly",
        target_audience="business customers"
    )
    
    print(f"Business Context:")
    print(f"  Industry: {business_context.industry}")
    print(f"  Use Case: {business_context.use_case}")
    print(f"  Compliance: {', '.join(business_context.compliance_requirements)}")
    print(f"  Brand Voice: {business_context.brand_voice}\n")
    
    # Show how synthetic data generation works
    generator = EnterpriseDataGenerator()
    print("Synthetic Data Generation:")
    print("  - Generates industry-specific scenarios")
    print("  - Ensures compliance with regulations")
    print("  - Maintains brand consistency")
    print("  - Creates diverse, realistic examples\n")
    
    # Show deployment assets
    print("Team Deployment Assets Generated:")
    print("  1. ChatGPT Template - Ready-to-use prompt for ChatGPT")
    print("  2. Slack Integration - Bot response templates")
    print("  3. Quick Reference Card - Key points for team")
    print("  4. Training Guide - Comprehensive usage instructions")


def show_cost_tracking():
    """Demonstrate cost tracking features."""
    print("\n=== Cost Tracking Demo ===\n")
    
    print("Cost Tracking Features:")
    print("- Tracks API usage per optimization run")
    print("- Monitors costs by provider and model")
    print("- Provides budget limits and warnings")
    print("- Generates ROI analysis reports\n")
    
    # Mock cost data
    print("Sample Cost Report:")
    print("```")
    print("Optimization Run Summary:")
    print("  Total API Calls: 157")
    print("  Total Tokens: 45,230")
    print("  Total Cost: $2.84")
    print("  ")
    print("Cost Breakdown:")
    print("  - DSPy Optimization: $1.45")
    print("  - GRPO Tournament: $1.12")
    print("  - Evaluation: $0.27")
    print("  ")
    print("Efficiency Metrics:")
    print("  - Cost per optimized prompt: $1.42")
    print("  - Estimated monthly savings: $850")
    print("  - ROI: 299%")
    print("```")


def main():
    """Run all demonstrations."""
    compare_optimizers_simple()
    demonstrate_enterprise_features()
    show_cost_tracking()
    
    print("\n=== Next Steps ===")
    print("1. Set up API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    print("2. Run actual optimization with: python examples/run_optimization.py")
    print("3. Try in Google Colab for easy setup")
    print("4. Explore hybrid optimization strategies")


if __name__ == "__main__":
    main()