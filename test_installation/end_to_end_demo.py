#!/usr/bin/env python3
"""End-to-end demonstration of PromptOpt features."""

import os
import sys

def demo_flexible_data_generation():
    """Demonstrate the flexible data generation system."""
    print("\n" + "="*60)
    print("DEMO 1: Flexible Data Generation")
    print("="*60)
    
    from promptopt.data import FlexibleDataGenerator, TemplateBuilder
    
    # Create a data generator
    generator = FlexibleDataGenerator()
    
    # Build a custom template for customer feedback
    template = (TemplateBuilder("customer_feedback")
        .with_description("Generate customer feedback data")
        .add_input_field("product", "enum:laptop,phone,tablet,headphones")
        .add_input_field("customer_segment", "enum:consumer,business,education")
        .add_input_field("purchase_date", "date", format="%Y-%m-%d")
        .add_input_field("order_id", "regex:ORD-\\d{8}")
        .add_output_field("rating", "enum:1,2,3,4,5", 
                         weight={5: 0.3, 4: 0.4, 3: 0.2, 2: 0.08, 1: 0.02})
        .add_output_field("sentiment", "enum:positive,neutral,negative")
        .add_output_field("would_recommend", "enum:yes,no,maybe")
        .add_output_field("key_points", "json")
        .with_input_template("""Order: {order_id}
Product: {product}
Customer Type: {customer_segment}
Purchase Date: {purchase_date}""")
        .with_output_template("""Rating: {rating}/5 stars
Sentiment: {sentiment}
Would Recommend: {would_recommend}
Key Points: {key_points}""")
        .build()
    )
    
    # Register the template
    generator.register_template(template)
    
    # Generate some examples
    dataset = generator.generate("customer_feedback", count=3)
    
    print(f"\nGenerated {len(dataset.examples)} customer feedback examples:")
    for i, example in enumerate(dataset.examples):
        print(f"\n--- Example {i+1} ---")
        print("INPUT:")
        print(example.input)
        print("\nOUTPUT:")
        print(example.output)
    
    return dataset

def demo_task_optimization():
    """Demonstrate task specification and optimization setup."""
    print("\n" + "="*60)
    print("DEMO 2: Task Specification and Constraints")
    print("="*60)
    
    from promptopt.core import TaskSpec, create_conciseness_constraint, create_tone_constraint
    
    # Define a customer support task
    task = TaskSpec(
        name="customer_support_response",
        description="Generate helpful and empathetic customer support responses",
        input_format={
            "customer_query": "string",
            "customer_sentiment": "enum:frustrated,neutral,happy",
            "product": "string",
            "history": "list[string]"
        },
        output_format={
            "response": "string",
            "action_items": "list[string]",
            "escalate": "boolean"
        },
        constraints=[
            create_conciseness_constraint(max_words=150, weight=0.7),
            create_tone_constraint("empathetic and professional", weight=0.9)
        ],
        examples=[
            {
                "input": {
                    "customer_query": "My order hasn't arrived yet",
                    "customer_sentiment": "frustrated",
                    "product": "laptop",
                    "history": []
                },
                "output": {
                    "response": "I sincerely apologize for the delay with your laptop order. I understand how frustrating this must be.",
                    "action_items": ["Check order status", "Contact shipping carrier"],
                    "escalate": False
                }
            }
        ]
    )
    
    print(f"Task: {task.name}")
    print(f"Description: {task.description}")
    print(f"Constraints: {len(task.constraints)}")
    print(f"Examples provided: {len(task.examples)}")
    
    # Validate the task
    print("\nValidating task specification...")
    is_valid = task.validate()
    print(f"Task valid: {'✅' if is_valid else '❌'}")
    
    return task

def demo_enterprise_features():
    """Demonstrate enterprise-focused features."""
    print("\n" + "="*60)
    print("DEMO 3: Enterprise Data Generation")
    print("="*60)
    
    from promptopt.data import EnterpriseDataGenerator
    from promptopt.core import BusinessContext
    
    # Create business context
    context = BusinessContext(
        industry="fintech",
        company_size="enterprise",
        use_case="customer_support",
        compliance_requirements=["PCI", "SOC2"],
        team_size=50,
        monthly_volume=10000
    )
    
    print(f"Business Context:")
    print(f"  Industry: {context.industry}")
    print(f"  Company Size: {context.company_size}")
    print(f"  Compliance: {', '.join(context.compliance_requirements)}")
    
    # Generate enterprise data
    generator = EnterpriseDataGenerator()
    dataset = generator.create_customer_support_data(
        company_context={
            "industry": context.industry,
            "products": ["payment_processing", "fraud_detection", "analytics"],
            "common_issues": ["transaction_failed", "account_locked", "integration_error"]
        },
        count=5
    )
    
    print(f"\nGenerated {len(dataset.examples)} enterprise examples")
    
    # Show first example
    if dataset.examples:
        print("\nFirst example:")
        print(f"Input: {dataset.examples[0].input[:100]}...")
        print(f"Output: {dataset.examples[0].output[:100]}...")
    
    return dataset, context

def demo_validation():
    """Demonstrate data validation."""
    print("\n" + "="*60)
    print("DEMO 4: Data Quality Validation")
    print("="*60)
    
    from promptopt.data import DataQualityValidator
    from promptopt.core import Dataset, Example
    
    # Create a mixed-quality dataset
    test_data = Dataset(
        examples=[
            Example(
                input="What's your refund policy?",
                output="Our refund policy allows returns within 30 days of purchase."
            ),
            Example(
                input="Password reset",
                output="To reset your password, click the 'Forgot Password' link on the login page and follow the instructions sent to your email."
            ),
            Example(
                input="How do I contact support?",
                output="Support"  # Too short
            ),
            Example(
                input="Cancel subscription",
                output="I understand you'd like to cancel your subscription. Before we proceed, may I ask what led to this decision? We'd love to address any concerns. To cancel, go to Settings > Subscription > Cancel. Your access will continue until the end of the billing period. If you change your mind, you can reactivate anytime. We're sorry to see you go and hope you'll consider us again in the future. Thank you for being a valued customer. Best regards, The Support Team"  # Too long
            )
        ],
        name="test_validation"
    )
    
    # Validate the dataset
    validator = DataQualityValidator()
    results = validator.validate_dataset(test_data)
    
    print(f"Validation Results:")
    print(f"  Overall Score: {results['overall_score']:.2f}/1.00")
    print(f"  Format Compliance: {results['format_compliance']['score']:.2f}")
    print(f"  Length Distribution: {results['length_distribution']['score']:.2f}")
    print(f"  Diversity Score: {results['diversity']['score']:.2f}")
    
    if results['issues']:
        print(f"\nIssues found: {len(results['issues'])}")
        for issue in results['issues'][:3]:  # Show first 3 issues
            print(f"  - {issue}")
    
    return results

def demo_hybrid_optimization_setup():
    """Demonstrate hybrid optimization setup (without running optimization)."""
    print("\n" + "="*60)
    print("DEMO 5: Hybrid Optimization Setup")
    print("="*60)
    
    from promptopt.optimizers import HybridConfig
    
    # Show available optimization strategies
    strategies = ["sequential", "ensemble", "feedback", "cost_aware"]
    
    print("Available Hybrid Strategies:")
    for strategy in strategies:
        print(f"  - {strategy}")
    
    # Create configuration for cost-aware optimization
    config = HybridConfig(
        budget_limit=10.0,
        max_iterations=3,
        improvement_threshold=0.1,
        enterprise_mode=True
    )
    
    print(f"\nCost-Aware Configuration:")
    print(f"  Budget Limit: ${config.budget_limit}")
    print(f"  Max Iterations: {config.max_iterations}")
    print(f"  Improvement Threshold: {config.improvement_threshold * 100}%")
    print(f"  Enterprise Mode: {config.enterprise_mode}")
    
    print("\nNote: Actual optimization requires API keys for OpenAI or Anthropic")
    
    return config

def main():
    """Run all demonstrations."""
    print("\n🚀 PromptOpt End-to-End Demonstration")
    print("="*60)
    print("This demo showcases key features of the PromptOpt package")
    print("installed from PyPI.")
    
    try:
        # Run demonstrations
        dataset1 = demo_flexible_data_generation()
        task = demo_task_optimization()
        dataset2, context = demo_enterprise_features()
        validation_results = demo_validation()
        config = demo_hybrid_optimization_setup()
        
        print("\n" + "="*60)
        print("✅ All demonstrations completed successfully!")
        print("\nKey Takeaways:")
        print("1. Flexible data generation with custom templates")
        print("2. Task specification with business constraints")
        print("3. Enterprise-focused data generation")
        print("4. Automatic data quality validation")
        print("5. Multiple optimization strategies available")
        print("\nTo use optimization features, set up API keys:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())