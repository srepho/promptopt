"""Tests for data generation and validation."""

import pytest
from promptopt.data import (
    EnterpriseDataGenerator, DataQualityValidator,
    BusinessScenarioTemplates, SyntheticDataQualityValidator
)
from promptopt.core import Dataset, BusinessContext


class TestEnterpriseDataGenerator:
    def test_data_generator_creation(self):
        generator = EnterpriseDataGenerator()
        assert generator.llm_client is None
        assert len(generator.scenario_templates) > 0
    
    def test_generate_customer_support_data(self):
        generator = EnterpriseDataGenerator()
        context = {
            "industry": "technology",
            "company_size": "enterprise",
            "products": ["Software", "Hardware"]
        }
        
        dataset = generator.create_customer_support_data(context, count=5)
        
        assert len(dataset) == 5
        assert dataset.name == "customer_support_synthetic"
        assert dataset.metadata["synthetic"] is True
        
        # Check first example
        example = dataset.examples[0]
        assert "Customer inquiry:" in example.input
        assert "Product:" in example.input
        assert len(example.output) > 0
    
    def test_scenario_templates(self):
        generator = EnterpriseDataGenerator()
        templates = generator.scenario_templates
        
        assert "customer_support" in templates
        assert "internal_email" in templates
        assert "content_creation" in templates
        
        # Check template structure
        cs_template = templates["customer_support"]
        assert cs_template.name == "Customer Support"
        assert len(cs_template.variables) > 0
        assert len(cs_template.constraints) > 0


class TestDataQualityValidator:
    def test_validator_creation(self):
        validator = DataQualityValidator()
        assert validator.validation_results == {}
    
    def test_validate_empty_dataset(self):
        validator = DataQualityValidator()
        dataset = Dataset(examples=[], name="empty")
        
        results = validator.validate_dataset(dataset)
        
        assert not results["passed"]
        assert "Dataset is empty" in results["errors"]
        assert results["total_examples"] == 0
    
    def test_validate_good_dataset(self):
        from promptopt.core import Example
        
        validator = DataQualityValidator()
        examples = [
            Example(input="What is AI?", output="AI is artificial intelligence."),
            Example(input="How does ML work?", output="ML uses algorithms to learn patterns."),
            Example(input="What is deep learning?", output="Deep learning uses neural networks.")
        ]
        dataset = Dataset(examples=examples, name="test")
        
        results = validator.validate_dataset(dataset)
        
        assert results["passed"]
        assert results["total_examples"] == 3
        assert len(results["errors"]) == 0
        assert results["overall_score"] > 0.7
    
    def test_statistics_calculation(self):
        from promptopt.core import Example
        
        validator = DataQualityValidator()
        examples = [
            Example(input="Short", output="Brief"),
            Example(input="A medium length input", output="A medium length output"),
            Example(input="This is a much longer input with many words", output="Long output here")
        ]
        dataset = Dataset(examples=examples)
        
        results = validator.validate_dataset(dataset)
        stats = results["statistics"]
        
        assert stats["input_length"]["mean"] > 0
        assert stats["input_length"]["min"] == 1  # "Short"
        assert stats["input_length"]["max"] == 9  # Longest input
        assert stats["unique_inputs"] == 3
        assert stats["duplicate_rate"] == 0.0


class TestBusinessScenarioTemplates:
    def test_templates_creation(self):
        templates = BusinessScenarioTemplates()
        
        # Test customer support templates
        cs_templates = templates.get_customer_support_templates()
        assert len(cs_templates) > 0
        assert cs_templates[0]["category"] == "technical_issue"
        
        # Test email templates
        email_templates = templates.get_internal_email_templates()
        assert len(email_templates) > 0
        assert any(t["category"] == "meeting_request" for t in email_templates)
    
    def test_fill_template(self):
        templates = BusinessScenarioTemplates()
        
        template = {
            "subject": "Update on {project}",
            "details": "Status: {status}"
        }
        
        values = {
            "project": "Q4 Initiative",
            "status": "On Track"
        }
        
        filled = templates.fill_template(template, values)
        
        assert filled["subject"] == "Update on Q4 Initiative"
        assert filled["details"] == "Status: On Track"


class TestSyntheticDataQualityValidator:
    def test_quality_validation(self):
        from promptopt.core import Example
        
        validator = SyntheticDataQualityValidator()
        
        examples = [
            Example(
                input="Customer: My order hasn't arrived. Sentiment: frustrated",
                output="I understand your frustration. Let me check that for you."
            ),
            Example(
                input="Customer: Great service! Sentiment: happy", 
                output="Thank you for your feedback! We appreciate it."
            )
        ]
        dataset = Dataset(examples=examples)
        
        scores = validator.validate_dataset(dataset)
        
        assert "diversity" in scores
        assert "realism" in scores
        assert "consistency" in scores
        assert "completeness" in scores
        assert "overall" in scores
        
        # Should have decent scores for this small dataset
        assert scores["overall"] > 0.5
        assert scores["completeness"] == 1.0  # No placeholders