"""Tests for utility functions."""

import pytest
import os
from unittest.mock import Mock, patch
from promptopt.utils import (
    LLMResponse, CostTracker, BaseLLMClient,
    DatasetLoader, DatasetSaver, DatasetTransformer
)
from promptopt.core import Example, Dataset


class TestCostTracker:
    def test_cost_calculation_openai(self):
        tracker = CostTracker("openai", "gpt-3.5-turbo")
        
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500
        }
        
        cost = tracker.calculate_cost(usage)
        
        # Based on pricing: input=$0.0005/1K, output=$0.0015/1K
        expected = (1000/1000 * 0.0005) + (500/1000 * 0.0015)
        assert cost == expected
    
    def test_cost_tracking_update(self):
        tracker = CostTracker("openai", "gpt-4")
        
        usage1 = {"prompt_tokens": 100, "completion_tokens": 50}
        cost1 = tracker.calculate_cost(usage1)
        tracker.update(usage1, cost1)
        
        usage2 = {"prompt_tokens": 200, "completion_tokens": 100}
        cost2 = tracker.calculate_cost(usage2)
        tracker.update(usage2, cost2)
        
        summary = tracker.get_summary()
        
        assert summary["total_prompt_tokens"] == 300
        assert summary["total_completion_tokens"] == 150
        assert summary["request_count"] == 2
        assert summary["total_cost"] == round(cost1 + cost2, 4)


class TestLLMResponse:
    def test_llm_response_creation(self):
        response = LLMResponse(
            text="Generated text",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            cost=0.05,
            latency=1.5
        )
        
        assert response.text == "Generated text"
        assert response.usage["prompt_tokens"] == 10
        assert response.cost == 0.05
        assert response.latency == 1.5
        assert response.timestamp is not None


class TestDatasetLoader:
    def test_load_from_dict(self):
        data = [
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2", "metadata": {"source": "test"}}
        ]
        
        dataset = DatasetLoader.load_from_dict(data, name="test_dataset")
        
        assert len(dataset) == 2
        assert dataset.name == "test_dataset"
        assert dataset[0].input == "q1"
        assert dataset[1].metadata["source"] == "test"
    
    def test_load_from_json(self, tmp_path):
        import json
        
        data = [
            {"input": "question", "output": "answer"},
            {"input": "query", "output": "response"}
        ]
        
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        dataset = DatasetLoader.load_from_json(json_file)
        
        assert len(dataset) == 2
        assert dataset.name == "test"
        assert dataset.metadata["format"] == "json"


class TestDatasetSaver:
    def test_save_to_json(self, tmp_path):
        import json
        
        examples = [
            Example(input="q1", output="a1"),
            Example(input="q2", output="a2", metadata={"id": 1})
        ]
        dataset = Dataset(examples=examples, name="test")
        
        json_file = tmp_path / "output.json"
        DatasetSaver.save_to_json(dataset, json_file)
        
        assert json_file.exists()
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert data[0]["input"] == "q1"
        assert data[1]["metadata"]["id"] == 1


class TestDatasetTransformer:
    def test_filter_by_length(self):
        examples = [
            Example(input="short", output="brief"),
            Example(input="This is a medium length input", output="Medium response"),
            Example(input=" ".join(["word"] * 20), output="Long output")
        ]
        dataset = Dataset(examples=examples)
        
        filtered = DatasetTransformer.filter_by_length(dataset, min_length=2, max_length=10)
        
        assert len(filtered) == 1  # Only medium example
        assert "medium" in filtered[0].input.lower()
    
    def test_sample_dataset(self):
        examples = [Example(input=f"q{i}", output=f"a{i}") for i in range(10)]
        dataset = Dataset(examples=examples)
        
        sampled = DatasetTransformer.sample(dataset, n=3, seed=42)
        
        assert len(sampled) == 3
        assert sampled.metadata["sample_size"] == 3
    
    def test_merge_datasets(self):
        ds1 = Dataset(
            examples=[Example(input="q1", output="a1")],
            name="dataset1"
        )
        ds2 = Dataset(
            examples=[Example(input="q2", output="a2")],
            name="dataset2"
        )
        
        merged = DatasetTransformer.merge_datasets([ds1, ds2], name="combined")
        
        assert len(merged) == 2
        assert merged.name == "combined"
        assert merged.metadata["merged_from"] == ["dataset1", "dataset2"]