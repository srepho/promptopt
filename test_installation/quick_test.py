#!/usr/bin/env python3
"""Quick test of PyPI installation."""

import subprocess
import sys

# Create a fresh virtual environment and test
print("Creating fresh virtual environment for testing...")

commands = [
    "python -m venv test_env",
    "source test_env/bin/activate && pip install promptopt && python -c 'import promptopt; print(f\"Successfully installed promptopt {promptopt.__version__}\")'",
    "source test_env/bin/activate && python -c 'from promptopt.data import FlexibleDataGenerator; print(\"✅ Data generation imports work\")'",
    "source test_env/bin/activate && python -c 'from promptopt.optimizers import create_hybrid_optimizer; print(\"✅ Optimizer imports work\")'",
]

for cmd in commands:
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    if result.returncode != 0:
        print(f"❌ Command failed")
        break
else:
    print("\n✅ All tests passed!")