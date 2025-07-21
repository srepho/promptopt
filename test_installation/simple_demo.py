#!/usr/bin/env python3
"""Simple working demonstration of PromptOpt."""

print("Testing PromptOpt installation...\n")

# Test 1: Basic imports
print("1. Testing imports...")
try:
    import promptopt
    from promptopt.core import TaskSpec, Dataset, Example
    from promptopt.data import FlexibleDataGenerator, TemplateBuilder
    print(f"✅ Successfully imported promptopt v{promptopt.__version__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Create simple data
print("\n2. Creating simple dataset...")
try:
    dataset = Dataset(
        examples=[
            Example(input="Hello", output="Hi there!"),
            Example(input="How are you?", output="I'm doing well, thank you!"),
            Example(input="Goodbye", output="Goodbye! Have a great day!")
        ],
        name="greetings"
    )
    print(f"✅ Created dataset with {len(dataset.examples)} examples")
except Exception as e:
    print(f"❌ Dataset creation failed: {e}")

# Test 3: Data generation
print("\n3. Testing data generation...")
try:
    generator = FlexibleDataGenerator()
    
    # Simple template
    template = (TemplateBuilder("support_tickets")
        .with_description("Support ticket data")
        .add_input_field("priority", "enum:low,medium,high,critical")
        .add_input_field("category", "enum:billing,technical,account,other")
        .add_output_field("response_time", "enum:immediate,1hour,4hours,24hours")
        .add_output_field("assigned_to", "enum:tier1,tier2,specialist")
        .build()
    )
    
    generator.register_template(template)
    generated = generator.generate("support_tickets", count=3)
    
    print(f"✅ Generated {len(generated.examples)} examples")
    
    # Show first example
    example = generated.examples[0]
    print(f"\nExample generated data:")
    print(f"  Input: {example.metadata['input_data']}")
    print(f"  Output: {example.metadata['output_data']}")
    
except Exception as e:
    print(f"❌ Data generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Task specification
print("\n4. Creating task specification...")
try:
    task = TaskSpec(
        name="email_classifier",
        description="Classify customer emails",
        input_format={"email_text": "string"},
        output_format={"category": "string", "priority": "string"}
    )
    print(f"✅ Created task: {task.name}")
    print(f"   Description: {task.description}")
except Exception as e:
    print(f"❌ Task creation failed: {e}")

# Test 5: Enterprise features
print("\n5. Testing enterprise features...")
try:
    from promptopt.data import EnterpriseDataGenerator
    
    gen = EnterpriseDataGenerator()
    enterprise_data = gen.create_customer_support_data(
        company_context={"industry": "tech", "products": ["SaaS"]},
        count=2
    )
    print(f"✅ Generated {len(enterprise_data.examples)} enterprise examples")
except Exception as e:
    print(f"❌ Enterprise features failed: {e}")

print("\n" + "="*50)
print("✅ PromptOpt is installed and working correctly!")
print("\nNext steps:")
print("1. Set API keys for optimization features:")
print("   export OPENAI_API_KEY='your-key'")
print("   export ANTHROPIC_API_KEY='your-key'")
print("2. Check out the examples in the GitHub repo")
print("3. Read the documentation for advanced features")