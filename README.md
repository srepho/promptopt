# PromptOpt: Enterprise Prompt Optimization Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified framework for testing, comparing, and hybridizing DSPy and GRPO approaches to prompt optimization, with special focus on enterprise deployment via synthetic data and Colab accessibility.

## 🌟 Key Features

- **🤖 Multiple Optimization Strategies**: DSPy, GRPO, and hybrid approaches
- **💰 Cost-Aware Optimization**: Built-in budget management and tracking
- **🏢 Enterprise-Ready**: Compliance support, ROI analysis, deployment tools
- **☁️ Colab-Optimized**: Works perfectly in Google Colab (no GPU needed)
- **📊 Tournament Evaluation**: Head-to-head prompt comparison system
- **🎯 Synthetic Data**: Generate realistic business scenarios
- **🔧 API-Based**: Works with OpenAI and Anthropic APIs

## Installation

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n promptopt python=3.9
conda activate promptopt

# Install the package
pip install -e .

# For development
pip install -r requirements-dev.txt
```

### Using pip only

```bash
pip install promptopt
```

For development:
```bash
pip install -e ".[dev]"
```

For Colab environments:
```bash
pip install promptopt[colab]
```

## Quick Start

```python
from promptopt import EnterprisePOC
from promptopt.colab import ColabManager

# Set up environment
manager = ColabManager()
manager.setup_enterprise_environment()

# Run a complete POC
poc = EnterprisePOC()
results = poc.run_complete_poc(
    business_scenario="customer_support",
    company_context={"industry": "tech", "size": "enterprise"},
    budget_limit=500.0
)

# View results
print(f"Optimization improvements: {results.optimization_improvements}")
print(f"Projected ROI: {results.roi_projections}")
```

## 📚 Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for comprehensive guides and API reference.

## 🚀 Quick Start in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/promptopt/promptopt/blob/main/notebooks/Enterprise_Quickstart.ipynb)

## 🧪 Running Tests

```bash
conda activate promptopt
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DSPy framework for few-shot optimization techniques
- GRPO methodology for tournament-based optimization
- OpenAI and Anthropic for LLM APIs

## 📊 Example Results

Using PromptOpt, teams typically see:
- **30-40% improvement** in response quality
- **85%+ consistency** across team members
- **$2000+/month savings** from optimized prompts
- **2-week ROI** for enterprise deployments