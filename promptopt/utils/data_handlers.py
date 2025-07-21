"""Utilities for handling datasets and data formats."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import pandas as pd

from ..core.base import Example, Dataset


class DatasetLoader:
    """Load datasets from various formats."""
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> Dataset:
        """Load dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            example = Example(
                input=item.get("input", item.get("question", "")),
                output=item.get("output", item.get("answer", "")),
                metadata=item.get("metadata", {})
            )
            examples.append(example)
        
        return Dataset(
            examples=examples,
            name=Path(file_path).stem,
            metadata={"source": str(file_path), "format": "json"}
        )
    
    @staticmethod
    def load_from_jsonl(file_path: Union[str, Path]) -> Dataset:
        """Load dataset from JSONL file."""
        examples = []
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                example = Example(
                    input=item.get("input", item.get("question", "")),
                    output=item.get("output", item.get("answer", "")),
                    metadata=item.get("metadata", {})
                )
                examples.append(example)
        
        return Dataset(
            examples=examples,
            name=Path(file_path).stem,
            metadata={"source": str(file_path), "format": "jsonl"}
        )
    
    @staticmethod
    def load_from_csv(file_path: Union[str, Path], 
                     input_column: str = "input",
                     output_column: str = "output") -> Dataset:
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path)
        
        examples = []
        for _, row in df.iterrows():
            example = Example(
                input=str(row[input_column]),
                output=str(row[output_column]),
                metadata={k: v for k, v in row.items() 
                         if k not in [input_column, output_column]}
            )
            examples.append(example)
        
        return Dataset(
            examples=examples,
            name=Path(file_path).stem,
            metadata={"source": str(file_path), "format": "csv"}
        )
    
    @staticmethod
    def load_from_dict(data: List[Dict[str, Any]], name: str = "dict_dataset") -> Dataset:
        """Load dataset from list of dictionaries."""
        examples = []
        for item in data:
            example = Example(
                input=item.get("input", ""),
                output=item.get("output", ""),
                metadata=item.get("metadata", {})
            )
            examples.append(example)
        
        return Dataset(
            examples=examples,
            name=name,
            metadata={"format": "dict"}
        )
    
    @classmethod
    def load(cls, source: Union[str, Path, List[Dict]], **kwargs) -> Dataset:
        """Load dataset from various sources."""
        if isinstance(source, list):
            return cls.load_from_dict(source, **kwargs)
        
        path = Path(source)
        if path.suffix == ".json":
            return cls.load_from_json(path)
        elif path.suffix == ".jsonl":
            return cls.load_from_jsonl(path)
        elif path.suffix == ".csv":
            return cls.load_from_csv(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


class DatasetSaver:
    """Save datasets to various formats."""
    
    @staticmethod
    def save_to_json(dataset: Dataset, file_path: Union[str, Path]):
        """Save dataset to JSON file."""
        data = []
        for example in dataset.examples:
            item = {
                "input": example.input,
                "output": example.output,
            }
            if example.metadata:
                item["metadata"] = example.metadata
            data.append(item)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def save_to_jsonl(dataset: Dataset, file_path: Union[str, Path]):
        """Save dataset to JSONL file."""
        with open(file_path, 'w') as f:
            for example in dataset.examples:
                item = {
                    "input": example.input,
                    "output": example.output,
                }
                if example.metadata:
                    item["metadata"] = example.metadata
                f.write(json.dumps(item) + '\n')
    
    @staticmethod
    def save_to_csv(dataset: Dataset, file_path: Union[str, Path]):
        """Save dataset to CSV file."""
        rows = []
        for example in dataset.examples:
            row = {
                "input": example.input,
                "output": example.output,
            }
            row.update(example.metadata)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
    
    @classmethod
    def save(cls, dataset: Dataset, file_path: Union[str, Path]):
        """Save dataset based on file extension."""
        path = Path(file_path)
        if path.suffix == ".json":
            cls.save_to_json(dataset, path)
        elif path.suffix == ".jsonl":
            cls.save_to_jsonl(dataset, path)
        elif path.suffix == ".csv":
            cls.save_to_csv(dataset, path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


class DatasetTransformer:
    """Transform and manipulate datasets."""
    
    @staticmethod
    def filter_by_length(dataset: Dataset, min_length: int = 0, 
                        max_length: int = float('inf')) -> Dataset:
        """Filter examples by input/output length."""
        filtered_examples = []
        for example in dataset.examples:
            input_len = len(example.input.split())
            output_len = len(example.output.split())
            
            if min_length <= input_len <= max_length and \
               min_length <= output_len <= max_length:
                filtered_examples.append(example)
        
        return Dataset(
            examples=filtered_examples,
            name=f"{dataset.name}_filtered",
            metadata={**dataset.metadata, "filter": "length"}
        )
    
    @staticmethod
    def sample(dataset: Dataset, n: int, seed: Optional[int] = None) -> Dataset:
        """Sample n examples from dataset."""
        import random
        if seed is not None:
            random.seed(seed)
        
        sampled_examples = random.sample(dataset.examples, 
                                       min(n, len(dataset.examples)))
        
        return Dataset(
            examples=sampled_examples,
            name=f"{dataset.name}_sample{n}",
            metadata={**dataset.metadata, "sample_size": n}
        )
    
    @staticmethod
    def augment_with_metadata(dataset: Dataset, 
                            metadata: Dict[str, Any]) -> Dataset:
        """Add metadata to all examples in dataset."""
        augmented_examples = []
        for example in dataset.examples:
            new_example = Example(
                input=example.input,
                output=example.output,
                metadata={**example.metadata, **metadata}
            )
            augmented_examples.append(new_example)
        
        return Dataset(
            examples=augmented_examples,
            name=dataset.name,
            metadata={**dataset.metadata, "augmented": True}
        )
    
    @staticmethod
    def merge_datasets(datasets: List[Dataset], name: str = "merged") -> Dataset:
        """Merge multiple datasets into one."""
        all_examples = []
        for dataset in datasets:
            all_examples.extend(dataset.examples)
        
        return Dataset(
            examples=all_examples,
            name=name,
            metadata={
                "merged_from": [d.name for d in datasets],
                "total_examples": len(all_examples)
            }
        )