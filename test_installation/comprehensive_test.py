#!/usr/bin/env python3
"""Comprehensive test of the installed promptopt package."""

def test_imports():
    """Test all major imports."""
    print("\n1. Testing imports...")
    
    try:
        import promptopt
        print(f"   ✅ promptopt version: {promptopt.__version__}")
    except Exception as e:
        print(f"   ❌ Failed to import promptopt: {e}")
        return False
    
    # Test core imports
    try:
        from promptopt.core import TaskSpec, Dataset, Example, Constraint
        from promptopt.core import EvaluationMetric, Metrics
        print("   ✅ Core imports successful")
    except Exception as e:
        print(f"   ❌ Core imports failed: {e}")
        return False
    
    # Test data imports
    try:
        from promptopt.data import (
            EnterpriseDataGenerator,
            FlexibleDataGenerator,
            TemplateBuilder,
            FieldType,
            DataQualityValidator
        )
        print("   ✅ Data generation imports successful")
    except Exception as e:
        print(f"   ❌ Data imports failed: {e}")
        return False
    
    # Test optimizer imports
    try:
        from promptopt.optimizers import create_hybrid_optimizer
        print("   ✅ Optimizer imports successful")
    except Exception as e:
        print(f"   ❌ Optimizer imports failed: {e}")
        return False
    
    # Test utils imports
    try:
        from promptopt.utils import create_llm_client, ResultVisualizer
        print("   ✅ Utils imports successful")
    except Exception as e:
        print(f"   ❌ Utils imports failed: {e}")
        return False
    
    return True

def test_core_functionality():
    """Test core functionality."""
    print("\n2. Testing core functionality...")
    
    try:
        from promptopt.core import TaskSpec, Dataset, Example
        
        # Create TaskSpec
        task = TaskSpec(
            name="test_task",
            description="Test task for verification",
            input_format={"query": "string"},
            output_format={"response": "string"}
        )
        print("   ✅ TaskSpec creation successful")
        
        # Create Dataset
        dataset = Dataset(
            examples=[
                Example(input="What is AI?", output="AI is artificial intelligence"),
                Example(input="Hello", output="Hi there!")
            ],
            name="test_dataset"
        )
        print(f"   ✅ Dataset creation successful ({len(dataset.examples)} examples)")
        
        # Test dataset split
        train, val = dataset.split(train_ratio=0.5)
        print(f"   ✅ Dataset split successful (train: {len(train.examples)}, val: {len(val.examples)})")
        
    except Exception as e:
        print(f"   ❌ Core functionality test failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation features."""
    print("\n3. Testing data generation...")
    
    try:
        from promptopt.data import FlexibleDataGenerator, TemplateBuilder
        
        # Create generator
        generator = FlexibleDataGenerator()
        
        # Create simple template
        template = (TemplateBuilder("test_emails")
            .with_description("Test email generation")
            .add_input_field("sender", "enum:customer,support,sales")
            .add_input_field("priority", "enum:low,medium,high")
            .add_output_field("subject", "text")
            .add_output_field("urgency_score", "number", min_value=1, max_value=10)
            .build()
        )
        
        generator.register_template(template)
        print("   ✅ Template registration successful")
        
        # Generate data
        dataset = generator.generate("test_emails", count=5)
        print(f"   ✅ Data generation successful ({len(dataset.examples)} examples)")
        
        # Test custom generator
        generator.register_generator("test_id", 
            lambda ctx, data: f"TEST-{hash(str(data)) % 1000:03d}")
        
        template2 = (TemplateBuilder("test_with_custom")
            .with_description("Test with custom generator")
            .add_input_field("id", "generator:test_id")
            .add_input_field("value", "number", min_value=0, max_value=100)
            .add_output_field("processed", "text")
            .build()
        )
        
        generator.register_template(template2)
        dataset2 = generator.generate("test_with_custom", count=3)
        print(f"   ✅ Custom generator successful ({len(dataset2.examples)} examples)")
        
    except Exception as e:
        print(f"   ❌ Data generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_enterprise_features():
    """Test enterprise features."""
    print("\n4. Testing enterprise features...")
    
    try:
        from promptopt.data import EnterpriseDataGenerator
        from promptopt.core import BusinessContext
        
        # Create business context
        context = BusinessContext(
            industry="technology",
            company_size="startup",
            use_case="customer_support"
        )
        print("   ✅ BusinessContext creation successful")
        
        # Test enterprise data generator (without LLM)
        generator = EnterpriseDataGenerator(llm_client=None)
        dataset = generator.create_customer_support_data(
            company_context={"industry": "tech", "products": ["SaaS", "API"]},
            count=10
        )
        print(f"   ✅ Enterprise data generation successful ({len(dataset.examples)} examples)")
        
    except Exception as e:
        print(f"   ❌ Enterprise features test failed: {e}")
        return False
    
    return True

def test_validation():
    """Test validation features."""
    print("\n5. Testing validation features...")
    
    try:
        from promptopt.data import DataQualityValidator
        from promptopt.core import Dataset, Example
        
        # Create test dataset
        dataset = Dataset(
            examples=[
                Example(input="Test input 1", output="Test output 1"),
                Example(input="Test input 2", output="Test output 2"),
                Example(input="Short", output="This is a much longer output than expected"),
            ]
        )
        
        # Validate dataset
        validator = DataQualityValidator()
        results = validator.validate_dataset(dataset)
        
        print(f"   ✅ Validation successful")
        print(f"      - Overall score: {results.get('overall_score', 0):.2f}")
        print(f"      - Format compliance: {results.get('format_compliance', {}).get('score', 0):.2f}")
        
    except Exception as e:
        print(f"   ❌ Validation test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Comprehensive PromptOpt Package Test")
    print("="*50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_core_functionality()
    all_passed &= test_data_generation()
    all_passed &= test_enterprise_features()
    all_passed &= test_validation()
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed! PromptOpt is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())