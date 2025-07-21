# PromptOpt v0.1.0 Release Notes

## 🎉 Initial Release

PromptOpt is now available on PyPI! Install with:
```bash
pip install promptopt
```

## ✅ What's Working

1. **Core Framework**
   - TaskSpec and Dataset creation
   - Example-based learning
   - Constraint system
   - Cost tracking infrastructure

2. **Data Generation** 
   - FlexibleDataGenerator with custom templates
   - EnterpriseDataGenerator for business scenarios
   - TemplateBuilder for easy template creation
   - Field types: text, number, date, json, composite

3. **Enterprise Features**
   - BusinessContext for industry-specific generation
   - Compliance-aware data generation
   - Multiple business scenario templates

4. **Package Distribution**
   - Published to PyPI: https://pypi.org/project/promptopt/
   - Source on GitHub: https://github.com/srepho/promptopt
   - Comprehensive documentation included

## 🐛 Known Issues

1. **Enum Generation**: Enum fields in FlexibleDataGenerator currently return placeholder text instead of selecting from options
2. **Import Name**: Documentation references `ResultVisualizer` but actual class is `ResultsVisualizer`
3. **Regex Fields**: Regex field generation returns simple placeholders

## 🚀 Quick Start

```python
from promptopt import TaskSpec, Dataset, Example
from promptopt.data import FlexibleDataGenerator, TemplateBuilder

# Create a simple dataset
dataset = Dataset(examples=[
    Example(input="Hello", output="Hi there!"),
    Example(input="Help", output="How can I assist you?")
])

# Generate synthetic data
generator = FlexibleDataGenerator()
template = (TemplateBuilder("my_template")
    .add_input_field("query", "text")
    .add_output_field("response", "text")
    .build()
)
generator.register_template(template)
data = generator.generate("my_template", count=10)
```

## 📝 TODO for v0.1.1

1. Fix enum field generation in FlexibleDataGenerator
2. Improve regex pattern generation
3. Add more examples to documentation
4. Create jupyter notebook tutorials
5. Add CI/CD with GitHub Actions

## 🙏 Acknowledgments

Thanks to the DSPy and GRPO communities for inspiration and methodology.