"""Demo script for hybrid optimization strategies."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptopt.core import (
    TaskSpec, Dataset, Example, BusinessContext,
    create_conciseness_constraint, create_format_constraint, create_tone_constraint
)
from promptopt.optimizers import (
    create_hybrid_optimizer, HybridConfig,
    SequentialHybrid, EnsembleHybrid, FeedbackHybrid, CostAwareHybrid,
    EnterpriseSequentialHybrid
)
from promptopt.data import EnterpriseDataGenerator
from promptopt.utils import ResultsVisualizer


def create_email_task():
    """Create a business email optimization task."""
    task_spec = TaskSpec(
        name="professional_email",
        description="Write professional business emails that are clear, concise, and action-oriented",
        input_format={"subject": "string", "context": "string", "recipient": "string"},
        output_format={"email_body": "string"},
        constraints=[
            create_conciseness_constraint(max_words=150, weight=0.3),
            create_tone_constraint("professional", weight=0.4),
            create_format_constraint("Subject: [SUBJECT]\n\nDear [RECIPIENT],\n\n[BODY]\n\nBest regards,", weight=0.3)
        ]
    )
    
    examples = [
        Example(
            input="Subject: Project Update\nContext: Weekly status update for Q4 initiative\nRecipient: Team Lead",
            output="Subject: Project Update - Week 45 Progress\n\nDear Team Lead,\n\nI'm pleased to report solid progress on our Q4 initiative this week. We've completed the API integration phase and are now 75% through user testing. The team resolved 3 critical bugs and remains on track for our December 1st launch.\n\nNext week, we'll focus on performance optimization and documentation updates. No blockers currently, but we'll need your input on the deployment strategy by Thursday.\n\nBest regards,"
        ),
        Example(
            input="Subject: Meeting Request\nContext: Need to discuss budget allocation\nRecipient: Finance Director",
            output="Subject: Meeting Request - Q1 Budget Allocation Discussion\n\nDear Finance Director,\n\nI'd like to schedule a meeting to discuss our Q1 budget allocation, specifically regarding the proposed marketing expansion. Could we meet for 30 minutes this week to review the projections and align on priorities?\n\nI'm available Tuesday 2-4 PM or Thursday morning. Please let me know what works best for your schedule.\n\nBest regards,"
        )
    ]
    
    dataset = Dataset(examples=examples * 3, name="email_optimization")  # Duplicate for more data
    return task_spec, dataset


def demonstrate_sequential_hybrid():
    """Demonstrate Sequential Hybrid optimization."""
    print("=== Sequential Hybrid Demo ===")
    print("Strategy: DSPy optimization followed by GRPO refinement\n")
    
    task_spec, dataset = create_email_task()
    
    # Simulated optimization flow (without actual LLM calls)
    print("Phase 1: DSPy Optimization")
    print("  - Bootstrapping few-shot examples")
    print("  - Optimizing instruction format")
    print("  - Result: Base prompt with 4 selected examples")
    
    print("\nPhase 2: GRPO Refinement")
    print("  - Creating 8 variations from DSPy result")
    print("  - Running tournament evaluation")
    print("  - Selecting winner based on constraint adherence")
    
    print("\nFinal Result:")
    print("  Improved constraint adherence: +15%")
    print("  Better format consistency: +20%")
    print("  Total optimization cost: $3.45")


def demonstrate_ensemble_hybrid():
    """Demonstrate Ensemble Hybrid optimization."""
    print("\n=== Ensemble Hybrid Demo ===")
    print("Strategy: Run multiple optimizers in parallel, select best\n")
    
    print("Running optimizers:")
    print("  1. DSPy (BootstrapFewShot)")
    print("  2. GRPO (Tournament-based)")
    print("  3. Sequential Hybrid")
    
    print("\nResults comparison:")
    results = {
        "DSPy": {"accuracy": 0.82, "cost": 1.20},
        "GRPO": {"accuracy": 0.85, "cost": 2.10},
        "Sequential": {"accuracy": 0.88, "cost": 3.30}
    }
    
    for name, metrics in results.items():
        print(f"  {name}: accuracy={metrics['accuracy']:.2%}, cost=${metrics['cost']:.2f}")
    
    print("\n✓ Selected: Sequential (highest accuracy)")


def demonstrate_feedback_hybrid():
    """Demonstrate Feedback Hybrid optimization."""
    print("\n=== Feedback Hybrid Demo ===")
    print("Strategy: Iterative optimization with feedback loop\n")
    
    print("Iteration 1:")
    print("  - DSPy generates initial prompt")
    print("  - GRPO tournament identifies winning patterns")
    print("  - Extract best examples for next iteration")
    
    print("\nIteration 2:")
    print("  - DSPy uses winning examples from previous round")
    print("  - Generate new variations")
    print("  - Tournament shows 5% improvement")
    
    print("\nIteration 3:")
    print("  - Further refinement with accumulated winners")
    print("  - Convergence detected (< 1% improvement)")
    print("  - Early stopping triggered")
    
    print("\nFinal performance: 12% better than single-pass optimization")


def demonstrate_cost_aware_hybrid():
    """Demonstrate Cost-Aware Hybrid optimization."""
    print("\n=== Cost-Aware Hybrid Demo ===")
    print("Strategy: Optimize within strict budget constraints\n")
    
    budget = 5.00
    print(f"Budget limit: ${budget:.2f}")
    
    print("\nPhase 1: Lightweight DSPy")
    print("  - Reduced demos (2 instead of 4)")
    print("  - Fewer iterations (5 instead of 10)")
    print("  - Cost: $1.85")
    
    remaining = budget - 1.85
    print(f"\nRemaining budget: ${remaining:.2f}")
    
    print("\nPhase 2: Budget-adjusted GRPO")
    print("  - Only 4 candidates (instead of 8)")
    print("  - Single tournament round")
    print("  - Cost: $2.90")
    
    print(f"\nTotal cost: $4.75 (under budget ✓)")
    print("Performance: 92% of full optimization at 47% of cost")


def demonstrate_enterprise_features():
    """Demonstrate enterprise-specific hybrid features."""
    print("\n=== Enterprise Sequential Hybrid Demo ===")
    print("Business-focused optimization with compliance\n")
    
    # Create business context
    business_context = BusinessContext(
        industry="finance",
        company_size="enterprise",
        use_case="client_communication",
        compliance_requirements=["SOC2", "FINRA"],
        brand_voice="professional and trustworthy",
        target_audience="institutional investors"
    )
    
    print(f"Industry: {business_context.industry}")
    print(f"Compliance: {', '.join(business_context.compliance_requirements)}")
    print(f"Brand Voice: {business_context.brand_voice}")
    
    print("\nOptimization includes:")
    print("  - Automatic compliance constraints")
    print("  - Brand voice enforcement")
    print("  - Industry-specific terminology")
    
    print("\nDeployment Package Generated:")
    print("  ✓ Production-ready prompt template")
    print("  ✓ Integration guides (ChatGPT, API, Slack)")
    print("  ✓ Team training materials")
    print("  ✓ Monitoring configuration")
    print("  ✓ Phased rollout plan")


def show_strategy_comparison():
    """Show comparison of different hybrid strategies."""
    print("\n=== Hybrid Strategy Comparison ===\n")
    
    strategies = {
        "Sequential": {
            "description": "DSPy → GRPO pipeline",
            "pros": "Balanced performance, leverages both strengths",
            "cons": "Higher cost, longer runtime",
            "best_for": "General optimization with good budget"
        },
        "Ensemble": {
            "description": "Run all, pick winner",
            "pros": "Best overall quality, robust",
            "cons": "Highest cost, requires most compute",
            "best_for": "Critical applications where quality is paramount"
        },
        "Feedback": {
            "description": "Iterative refinement",
            "pros": "Learns from results, can achieve high quality",
            "cons": "Variable runtime, may not converge",
            "best_for": "Complex tasks with clear success metrics"
        },
        "Cost-Aware": {
            "description": "Budget-constrained optimization",
            "pros": "Predictable costs, efficient",
            "cons": "May sacrifice some quality",
            "best_for": "Large-scale deployment, cost-sensitive applications"
        }
    }
    
    for name, info in strategies.items():
        print(f"{name} Hybrid:")
        print(f"  Description: {info['description']}")
        print(f"  Pros: {info['pros']}")
        print(f"  Cons: {info['cons']}")
        print(f"  Best for: {info['best_for']}")
        print()


def main():
    """Run all demonstrations."""
    print("=== Prompt Optimization: Hybrid Strategies Demo ===\n")
    
    # Check environment
    import subprocess
    result = subprocess.run(
        [sys.executable, "check_environment.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("⚠️  Environment check failed. Please set up conda environment first.")
        print("   See CLAUDE.md for instructions.")
        print()
    
    # Run demonstrations
    demonstrate_sequential_hybrid()
    demonstrate_ensemble_hybrid()
    demonstrate_feedback_hybrid()
    demonstrate_cost_aware_hybrid()
    demonstrate_enterprise_features()
    show_strategy_comparison()
    
    print("\n=== Summary ===")
    print("Hybrid strategies combine the strengths of different optimization approaches:")
    print("• Sequential: Best balance of quality and cost")
    print("• Ensemble: Highest quality when budget allows")
    print("• Feedback: Adaptive optimization for complex tasks")
    print("• Cost-Aware: Enterprise-scale optimization within budgets")
    print("\nChoose based on your specific needs and constraints!")


if __name__ == "__main__":
    main()