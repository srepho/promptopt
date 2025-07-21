#!/usr/bin/env python
"""Check if running in the correct conda environment."""

import sys
import os


def check_conda_environment():
    """Check if we're running in a conda environment and if it's the right one."""
    # Check if we're in any conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if not conda_env:
        print("❌ WARNING: Not running in a conda environment!")
        print("   Please activate the promptopt environment:")
        print("   $ conda activate promptopt")
        print()
        print("   If the environment doesn't exist, create it with:")
        print("   $ conda create -n promptopt python=3.9 -y")
        return False
    
    # Check if it's the correct environment
    if conda_env != 'promptopt':
        print(f"⚠️  WARNING: Running in '{conda_env}' environment, not 'promptopt'!")
        print("   Switch to the correct environment:")
        print("   $ conda activate promptopt")
        return False
    
    # All good!
    print(f"✓ Running in correct conda environment: {conda_env}")
    print(f"  Python: {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    return True


def check_package_installed():
    """Check if promptopt package is installed."""
    try:
        import promptopt
        print(f"✓ PromptOpt package is installed (version {promptopt.__version__})")
        return True
    except ImportError:
        print("❌ PromptOpt package not installed!")
        print("   Install in development mode:")
        print("   $ pip install -e .")
        return False


def check_dependencies():
    """Check if key dependencies are available."""
    dependencies = {
        'numpy': 'Core numerical operations',
        'pandas': 'Data handling',
        'openai': 'OpenAI API client',
        'anthropic': 'Anthropic API client',
    }
    
    print("\nChecking dependencies:")
    all_good = True
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"  ✓ {package} - {description}")
        except ImportError:
            print(f"  ❌ {package} - {description} (NOT INSTALLED)")
            all_good = False
    
    if not all_good:
        print("\nInstall missing dependencies with:")
        print("  $ pip install -r requirements.txt")
    
    return all_good


def main():
    """Run all environment checks."""
    print("=== PromptOpt Environment Check ===\n")
    
    checks = [
        check_conda_environment(),
        check_package_installed(),
        check_dependencies()
    ]
    
    if all(checks):
        print("\n✅ Environment is properly set up!")
    else:
        print("\n❌ Please fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()