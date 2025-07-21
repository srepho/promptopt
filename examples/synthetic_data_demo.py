"""Demo script for synthetic data generation."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptopt.data import (
    EnterpriseDataGenerator,
    DataQualityValidator,
    InteractiveSyntheticWizard
)
from promptopt.core import BusinessContext
from promptopt.utils import create_llm_client, DatasetSaver


def main():
    """Run synthetic data generation demo."""
    print("=== Prompt Optimization Synthetic Data Demo ===\n")
    
    # Create business context
    business_context = BusinessContext(
        industry="technology",
        company_size="enterprise",
        use_case="customer_support",
        compliance_requirements=["SOC2"],
        brand_voice="professional yet friendly"
    )
    
    print(f"Business Context: {business_context.industry} - {business_context.use_case}")
    print(f"Compliance: {', '.join(business_context.compliance_requirements)}\n")
    
    # Initialize data generator (without LLM for demo)
    generator = EnterpriseDataGenerator(llm_client=None)
    
    # Generate customer support data
    print("Generating synthetic customer support data...")
    dataset = generator.create_customer_support_data(
        company_context=business_context.to_dict(),
        count=10  # Small count for demo
    )
    
    print(f"Generated {len(dataset)} examples\n")
    
    # Preview some examples
    print("=== Sample Examples ===")
    for i, example in enumerate(dataset.examples[:3], 1):
        print(f"\nExample {i}:")
        print(f"Input: {example.input}")
        print(f"Output: {example.output}")
        print(f"Metadata: {example.metadata.get('scenario', 'N/A')}")
    
    # Validate data quality
    print("\n=== Data Quality Validation ===")
    validator = DataQualityValidator()
    validation_results = validator.validate_dataset(dataset)
    
    print(f"Overall Score: {validation_results['overall_score']:.2f}")
    print(f"Quality Scores:")
    for metric, score in validation_results['quality_scores'].items():
        print(f"  - {metric}: {score:.2f}")
    
    if validation_results['warnings']:
        print(f"\nWarnings:")
        for warning in validation_results['warnings'][:3]:
            print(f"  - {warning}")
    
    # Save dataset
    output_path = "synthetic_customer_support.json"
    DatasetSaver.save(dataset, output_path)
    print(f"\nDataset saved to: {output_path}")
    
    # Interactive wizard demo
    print("\n=== Interactive Wizard Demo ===")
    wizard = InteractiveSyntheticWizard()
    wizard_config = wizard.run_scenario_builder()
    
    # Generate more scenarios
    print("\n=== Additional Scenario Types ===")
    scenarios = ["internal_email", "content_creation"]
    for scenario in scenarios:
        print(f"\nGenerating {scenario} data...")
        context = {
            **business_context.to_dict(),
            "scenario_type": scenario
        }
        examples = generator.generate(count=5, context=context)
        print(f"Generated {len(examples)} {scenario} examples")
        
        # Show first example
        if examples:
            print(f"Sample: {examples[0]['input'][:100]}...")


if __name__ == "__main__":
    main()