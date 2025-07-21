"""Tests for core functionality."""

import pytest
from promptopt.core import (
    Example, Dataset, TaskSpec, Constraint, OptimizedPrompt,
    create_conciseness_constraint, create_format_constraint
)


class TestExample:
    def test_example_creation(self):
        ex = Example(input="test input", output="test output")
        assert ex.input == "test input"
        assert ex.output == "test output"
        assert ex.metadata == {}
    
    def test_example_with_metadata(self):
        ex = Example(
            input="test", 
            output="result",
            metadata={"source": "synthetic"}
        )
        assert ex.metadata["source"] == "synthetic"


class TestDataset:
    def test_dataset_creation(self):
        examples = [
            Example(input="q1", output="a1"),
            Example(input="q2", output="a2")
        ]
        dataset = Dataset(examples=examples, name="test_dataset")
        
        assert len(dataset) == 2
        assert dataset.name == "test_dataset"
        assert dataset[0].input == "q1"
    
    def test_dataset_split(self):
        examples = [Example(input=f"q{i}", output=f"a{i}") for i in range(10)]
        dataset = Dataset(examples=examples)
        
        train, val = dataset.split(train_ratio=0.8)
        
        assert len(train) == 8
        assert len(val) == 2
        assert train.metadata["split"] == "train"
        assert val.metadata["split"] == "val"


class TestConstraints:
    def test_conciseness_constraint(self):
        constraint = create_conciseness_constraint(max_words=10)
        
        assert constraint.name == "conciseness"
        assert constraint.validator("This is a short response")  # 5 words
        assert not constraint.validator("This is a very long response that exceeds the maximum word limit")  # >10 words
    
    def test_format_constraint(self):
        constraint = create_format_constraint("Answer: [YES/NO]")
        
        assert constraint.validator("Answer: YES")
        assert constraint.validator("Answer: NO")
        assert not constraint.validator("Maybe")


class TestTaskSpec:
    def test_task_spec_creation(self):
        task = TaskSpec(
            name="test_task",
            description="Test task description",
            input_format={"question": "string"},
            output_format={"answer": "string"}
        )
        
        assert task.name == "test_task"
        assert task.description == "Test task description"
        assert len(task.constraints) == 0
    
    def test_task_spec_with_constraints(self):
        constraint = create_conciseness_constraint(max_words=50)
        task = TaskSpec(
            name="test_task",
            description="Test task",
            input_format={},
            output_format={},
            constraints=[constraint]
        )
        
        result = task.validate_response("Short response")
        assert result.passed
        assert result.overall_score == 1.0
        
        long_response = " ".join(["word"] * 60)
        result = task.validate_response(long_response)
        assert not result.passed
        assert result.overall_score < 1.0


class TestOptimizedPrompt:
    def test_optimized_prompt_creation(self):
        prompt = OptimizedPrompt(
            text="Test prompt text",
            examples=[Example(input="q", output="a")]
        )
        
        assert prompt.text == "Test prompt text"
        assert len(prompt.examples) == 1
        assert prompt.id is not None
        assert prompt.created_at is not None
    
    def test_optimized_prompt_to_dict(self):
        prompt = OptimizedPrompt(
            text="Test prompt",
            metadata={"optimizer": "test"}
        )
        
        data = prompt.to_dict()
        assert data["text"] == "Test prompt"
        assert data["metadata"]["optimizer"] == "test"
        assert "id" in data
        assert "created_at" in data