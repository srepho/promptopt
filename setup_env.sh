#!/bin/bash
# Setup script for PromptOpt development environment

echo "=== PromptOpt Environment Setup ==="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed!"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if promptopt environment exists
if conda env list | grep -q "promptopt"; then
    echo "✓ Found existing promptopt environment"
    echo "  To activate: conda activate promptopt"
else
    echo "Creating new conda environment 'promptopt'..."
    conda create -n promptopt python=3.9 -y
    echo "✓ Environment created"
fi

echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate promptopt"
echo "2. Install the package: pip install -e ."
echo "3. Install dev dependencies: pip install -r requirements-dev.txt"
echo ""
echo "To verify setup:"
echo "  conda activate promptopt"
echo "  python -c 'import promptopt; print(promptopt.__version__)'"