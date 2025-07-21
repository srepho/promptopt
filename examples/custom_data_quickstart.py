#!/usr/bin/env python3
"""Quick start guide for custom data generation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptopt.data import FlexibleDataGenerator, TemplateBuilder


def simple_custom_example():
    """Simple example of creating custom data for your specific needs."""
    
    # Initialize generator
    generator = FlexibleDataGenerator()
    
    # Example 1: Simple template with custom fields
    print("=== Example 1: Custom Support Ticket Data ===\n")
    
    # Define your custom template
    template = (TemplateBuilder("support_tickets")
        .with_description("Custom support ticket data")
        # Define input fields with your specific requirements
        .add_input_field("ticket_id", "regex:TICK-\\d{6}")
        .add_input_field("customer_type", "enum:free,premium,enterprise", 
                        weight={"free": 0.5, "premium": 0.3, "enterprise": 0.2})
        .add_input_field("issue_category", "enum:billing,technical,feature_request,bug")
        .add_input_field("priority", "enum:low,medium,high,critical")
        .add_input_field("product_version", "template:v{major}.{minor}.{patch}")
        # Define output fields
        .add_output_field("resolution_time_hours", "number", min_value=0.5, max_value=72)
        .add_output_field("satisfaction_score", "enum:1,2,3,4,5")
        .add_output_field("resolved_by", "enum:ai_agent,tier1,tier2,engineering")
        # Use templates for formatting
        .with_input_template("""Ticket: {ticket_id}
Customer Type: {customer_type}
Issue: {issue_category}
Priority: {priority}
Product Version: {product_version}""")
        .with_output_template("""Resolution Time: {resolution_time_hours} hours
Satisfaction: {satisfaction_score}/5
Resolved By: {resolved_by}""")
        .build()
    )
    
    # Register and generate
    generator.register_template(template)
    
    # Generate with context
    context = {"major": "2", "minor": "1", "patch": "5"}
    dataset = generator.generate("support_tickets", count=3, context=context)
    
    # Display results
    for i, example in enumerate(dataset.examples):
        print(f"Example {i+1}:")
        print(example.input)
        print(example.output)
        print("-" * 40)


def advanced_custom_example():
    """Advanced example with custom generators and validation."""
    
    print("\n=== Example 2: Complex Custom Data with Validation ===\n")
    
    generator = FlexibleDataGenerator()
    
    # Register custom generator functions
    generator.register_generator("user_segment", 
        lambda ctx, data: f"{data.get('region', 'NA')}-{data.get('tier', 'STD')}-{ctx.get('year', '2024')}")
    
    generator.register_generator("custom_id",
        lambda ctx, data: f"{ctx.get('prefix', 'ID')}-{hash(str(data)) % 100000:05d}")
    
    # Define validation rules
    def validate_score_consistency(input_data, output_data):
        # Ensure scores are consistent with input metrics
        if input_data.get("engagement_level") == "high" and output_data.get("quality_score", 0) < 7:
            return False
        return True
    
    # Create complex template
    template = (TemplateBuilder("user_analytics")
        .with_description("User behavior analytics data")
        # Nested input structure
        .add_input_field("user", "json", sub_fields={
            "id": {"type": "function", "generator": "custom_id"},
            "segment": {"type": "function", "generator": "user_segment"},
            "region": {"type": "enum", "options": ["NA", "EU", "APAC", "LATAM"]},
            "tier": {"type": "enum", "options": ["FREE", "PRO", "ENT"]}
        })
        .add_input_field("engagement_level", "enum:low,medium,high")
        .add_input_field("feature_usage", "json")
        # Output with dependencies
        .add_output_field("quality_score", "number", min_value=1, max_value=10)
        .add_output_field("churn_risk", "enum:low,medium,high")
        .add_output_field("recommendations", "json")
        # Add validation
        .add_validation(validate_score_consistency)
        .build()
    )
    
    generator.register_template(template)
    
    # Generate with specific context
    context = {"prefix": "USR", "year": "2024"}
    dataset = generator.generate("user_analytics", count=2, context=context)
    
    for example in dataset.examples:
        print("\nGenerated Data:")
        print(f"Input: {example.metadata['input_data']}")
        print(f"Output: {example.metadata['output_data']}")
        print("-" * 40)


def your_specific_use_case():
    """Template for your specific generation requirements."""
    
    print("\n=== Your Custom Use Case ===\n")
    print("To create data for your specific needs:")
    print("1. Define your input fields (what data you have)")
    print("2. Define your output fields (what you want to generate)")
    print("3. Add any validation rules")
    print("4. Generate!\n")
    
    generator = FlexibleDataGenerator()
    
    # Here's a template you can modify for your needs:
    template = (TemplateBuilder("your_custom_data")
        .with_description("Description of your data")
        # Add your input fields here
        .add_input_field("field1", "text")  # Simple text field
        .add_input_field("field2", "enum:option1,option2,option3")  # Enum field
        .add_input_field("field3", "number", min_value=0, max_value=100)  # Number with range
        .add_input_field("field4", "regex:[A-Z]{3}-\\d{4}")  # Pattern-based field
        # Add your output fields here
        .add_output_field("result1", "text")
        .add_output_field("result2", "json")
        # Optional: Add formatting templates
        .with_input_template("Input: {field1} | {field2} | {field3} | {field4}")
        .with_output_template("Output: {result1}\nData: {result2}")
        .build()
    )
    
    generator.register_template(template)
    dataset = generator.generate("your_custom_data", count=1)
    
    print("Generated example:")
    print(dataset.examples[0].input)
    print(dataset.examples[0].output)


if __name__ == "__main__":
    # Run examples
    simple_custom_example()
    advanced_custom_example()
    your_specific_use_case()
    
    print("\n✅ Examples completed!")
    print("\nTo create your own custom data generator:")
    print("1. Copy one of these examples")
    print("2. Modify the field definitions for your needs")
    print("3. Add any custom validation rules")
    print("4. Generate your data!")
    print("\nSee flexible_data_generation.py for more advanced examples.")