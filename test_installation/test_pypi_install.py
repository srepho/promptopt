#!/usr/bin/env python3
"""Test script to verify PyPI installation works correctly."""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code: {result.returncode}")
        return False
    else:
        print(f"✅ Command succeeded")
        return True

def test_imports():
    """Test importing all major components."""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    test_code = '''
import promptopt
print(f"✅ promptopt version: {promptopt.__version__}")

# Core imports
from promptopt.core import TaskSpec, Dataset, Example, BaseOptimizer
print("✅ Core imports successful")

# Data generation imports
from promptopt.data import (
    EnterpriseDataGenerator,
    FlexibleDataGenerator,
    TemplateBuilder,
    DataQualityValidator
)
print("✅ Data generation imports successful")

# Optimizer imports
from promptopt.optimizers import (
    DSPyAdapter,
    GRPOAdapter,
    create_hybrid_optimizer
)
print("✅ Optimizer imports successful")

# Utils imports
from promptopt.utils import create_llm_client, ResultVisualizer
print("✅ Utils imports successful")

# Enterprise imports
from promptopt.enterprise import EnterprisePOC
print("✅ Enterprise imports successful")

# Colab imports
from promptopt.colab import ColabManager
print("✅ Colab imports successful")

print("\\n🎉 All imports successful!")
'''
    
    result = subprocess.run([sys.executable, '-c', test_code], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

def test_basic_functionality():
    """Test basic functionality."""
    print("\n" + "="*60)
    print("Testing basic functionality...")
    print("="*60)
    
    test_code = '''
from promptopt.core import TaskSpec, Dataset, Example
from promptopt.data import FlexibleDataGenerator, TemplateBuilder

# Test 1: Create a simple TaskSpec
task = TaskSpec(
    name="test_task",
    description="Test task",
    input_format={"text": "string"},
    output_format={"response": "string"}
)
print("✅ TaskSpec creation successful")

# Test 2: Create a simple Dataset
dataset = Dataset(
    examples=[
        Example(input="Hello", output="World"),
        Example(input="Test", output="Response")
    ]
)
print(f"✅ Dataset creation successful with {len(dataset.examples)} examples")

# Test 3: Test flexible data generator
generator = FlexibleDataGenerator()
template = (TemplateBuilder("test_template")
    .with_description("Test template")
    .add_input_field("field1", "text")
    .add_output_field("result", "text")
    .build()
)
generator.register_template(template)
test_data = generator.generate("test_template", count=2)
print(f"✅ Data generation successful with {len(test_data.examples)} examples")

print("\\n🎉 Basic functionality tests passed!")
'''
    
    result = subprocess.run([sys.executable, '-c', test_code], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

def main():
    """Main test function."""
    print("🧪 Testing PromptOpt PyPI Installation")
    print("="*60)
    
    # First, let's check current environment
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    
    # Install from PyPI
    success = run_command(
        f"{sys.executable} -m pip install promptopt",
        "Installing promptopt from PyPI"
    )
    
    if not success:
        print("\n❌ Installation failed!")
        return 1
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return 1
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed!")
        return 1
    
    # Show installed package info
    run_command(
        f"{sys.executable} -m pip show promptopt",
        "Package information"
    )
    
    print("\n" + "="*60)
    print("🎉 All tests passed! PromptOpt is working correctly!")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())