"""GRPO (Generative Reward-based Prompt Optimization) adapter."""

from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from ..core.base import BaseOptimizer, Dataset, TaskSpec, OptimizedPrompt, Metrics, Example
from ..core.interfaces import TournamentResult, ComparisonResult
from ..utils.llm_clients import BaseLLMClient, LLMJudge
from ..evaluation.tournaments import TournamentManager


@dataclass
class GRPOConfig:
    """Configuration for GRPO optimization."""
    num_candidates: int = 8
    tournament_rounds: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    elite_ratio: float = 0.25
    temperature_schedule: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.5])
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "performance": 0.7,
        "format_adherence": 0.2,
        "cost_efficiency": 0.1
    })


@dataclass
class PromptCandidate:
    """Represents a prompt candidate in GRPO."""
    id: str
    text: str
    examples: List[Example]
    generation: int = 0
    fitness_score: float = 0.0
    win_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GRPOAdapter(BaseOptimizer):
    """GRPO adapter for API-based prompt optimization."""
    
    def __init__(self, llm_client: BaseLLMClient, 
                 judge_client: Optional[BaseLLMClient] = None,
                 config: Optional[GRPOConfig] = None):
        super().__init__()
        self.llm_client = llm_client
        self.judge_client = judge_client or llm_client
        self.judge = LLMJudge(self.judge_client)
        self.config = config or GRPOConfig()
        self.tournament_manager = TournamentManager(self.judge)
        self.generation_count = 0
    
    def optimize(self, task_spec: TaskSpec, dataset: Dataset) -> OptimizedPrompt:
        """Optimize prompt using GRPO with tournament selection."""
        # Initialize population
        population = self._initialize_population(task_spec, dataset)
        
        # Evolution loop
        for round_idx in range(self.config.tournament_rounds):
            logging.info(f"GRPO Round {round_idx + 1}/{self.config.tournament_rounds}")
            
            # Run tournament
            tournament_results = self._run_tournament(population, dataset, task_spec)
            
            # Update fitness scores
            self._update_fitness_scores(population, tournament_results)
            
            # Select elite candidates
            elite = self._select_elite(population)
            
            # Generate new candidates through mutation and crossover
            new_candidates = self._evolve_population(elite, task_spec, dataset)
            
            # Update population
            population = elite + new_candidates
            self.generation_count += 1
        
        # Final tournament to select best
        final_results = self._run_tournament(population, dataset, task_spec)
        best_candidate = self._select_best_candidate(population, final_results)
        
        # Convert to OptimizedPrompt
        return self._candidate_to_optimized_prompt(best_candidate, task_spec)
    
    def _initialize_population(self, task_spec: TaskSpec, 
                             dataset: Dataset) -> List[PromptCandidate]:
        """Initialize diverse population of prompt candidates."""
        candidates = []
        
        # Strategy 1: Basic template
        basic_prompt = self._create_basic_prompt(task_spec)
        candidates.append(PromptCandidate(
            id=f"candidate_0",
            text=basic_prompt,
            examples=self._select_examples(dataset, 3),
            metadata={"strategy": "basic"}
        ))
        
        # Strategy 2: Detailed instructions
        detailed_prompt = self._create_detailed_prompt(task_spec)
        candidates.append(PromptCandidate(
            id=f"candidate_1",
            text=detailed_prompt,
            examples=self._select_examples(dataset, 5),
            metadata={"strategy": "detailed"}
        ))
        
        # Strategy 3: Constraint-focused
        constraint_prompt = self._create_constraint_focused_prompt(task_spec)
        candidates.append(PromptCandidate(
            id=f"candidate_2",
            text=constraint_prompt,
            examples=self._select_examples(dataset, 4),
            metadata={"strategy": "constraint_focused"}
        ))
        
        # Strategy 4-N: Variations
        for i in range(3, self.config.num_candidates):
            variation = self._create_prompt_variation(task_spec, dataset)
            candidates.append(PromptCandidate(
                id=f"candidate_{i}",
                text=variation["text"],
                examples=variation["examples"],
                metadata={"strategy": "variation"}
            ))
        
        return candidates
    
    def _create_basic_prompt(self, task_spec: TaskSpec) -> str:
        """Create a basic prompt template."""
        return f"""{task_spec.description}

Input will be provided in the following format:
{task_spec.input_format}

Please provide output in this format:
{task_spec.output_format}"""
    
    def _create_detailed_prompt(self, task_spec: TaskSpec) -> str:
        """Create a detailed prompt with step-by-step instructions."""
        prompt = f"""Task: {task_spec.description}

Step-by-step instructions:
1. Carefully read the input provided
2. Analyze what is being asked
3. Consider the output format requirements
4. Generate a response that matches the format exactly

Input format: {task_spec.input_format}
Output format: {task_spec.output_format}"""
        
        if task_spec.constraints:
            prompt += "\n\nImportant constraints:"
            for constraint in task_spec.constraints:
                prompt += f"\n- {constraint.description}"
        
        return prompt
    
    def _create_constraint_focused_prompt(self, task_spec: TaskSpec) -> str:
        """Create a prompt that emphasizes constraints."""
        prompt = f"""Complete this task: {task_spec.description}

CRITICAL REQUIREMENTS:"""
        
        for constraint in task_spec.constraints:
            prompt += f"\n- {constraint.description} (MUST follow)"
        
        prompt += f"\n\nInput format: {task_spec.input_format}"
        prompt += f"\nOutput format: {task_spec.output_format}"
        prompt += "\n\nEnsure ALL requirements are met in your response."
        
        return prompt
    
    def _create_prompt_variation(self, task_spec: TaskSpec, 
                               dataset: Dataset) -> Dict[str, Any]:
        """Create a prompt variation using different strategies."""
        strategies = [
            self._chain_of_thought_variation,
            self._role_based_variation,
            self._example_heavy_variation,
            self._concise_variation
        ]
        
        strategy = random.choice(strategies)
        return strategy(task_spec, dataset)
    
    def _chain_of_thought_variation(self, task_spec: TaskSpec, 
                                   dataset: Dataset) -> Dict[str, Any]:
        """Create chain-of-thought style prompt."""
        prompt = f"""{task_spec.description}

Let's approach this step-by-step:
1. First, understand the input: {task_spec.input_format}
2. Then, process according to the requirements
3. Finally, format the output as: {task_spec.output_format}

Think through your response before providing the final answer."""
        
        return {
            "text": prompt,
            "examples": self._select_examples(dataset, 2)
        }
    
    def _role_based_variation(self, task_spec: TaskSpec, 
                            dataset: Dataset) -> Dict[str, Any]:
        """Create role-based prompt."""
        roles = [
            "You are an expert assistant",
            "As a professional",
            "You are a helpful AI"
        ]
        
        prompt = f"""{random.choice(roles)} tasked with: {task_spec.description}

Given input in format: {task_spec.input_format}
Provide output in format: {task_spec.output_format}

Be accurate and follow all requirements."""
        
        return {
            "text": prompt,
            "examples": self._select_examples(dataset, 4)
        }
    
    def _example_heavy_variation(self, task_spec: TaskSpec, 
                                dataset: Dataset) -> Dict[str, Any]:
        """Create prompt relying heavily on examples."""
        prompt = f"""{task_spec.description}

Follow the pattern shown in the examples below."""
        
        return {
            "text": prompt,
            "examples": self._select_examples(dataset, 6)
        }
    
    def _concise_variation(self, task_spec: TaskSpec, 
                         dataset: Dataset) -> Dict[str, Any]:
        """Create very concise prompt."""
        prompt = f"""Task: {task_spec.description}
Input: {task_spec.input_format}
Output: {task_spec.output_format}"""
        
        return {
            "text": prompt,
            "examples": self._select_examples(dataset, 3)
        }
    
    def _select_examples(self, dataset: Dataset, count: int) -> List[Example]:
        """Select diverse examples from dataset."""
        if len(dataset) <= count:
            return dataset.examples
        
        # Select diverse examples based on length and content
        examples = list(dataset.examples)
        selected = []
        
        # Sort by length to get variety
        examples.sort(key=lambda x: len(x.input) + len(x.output))
        
        # Pick examples from different parts of the distribution
        indices = np.linspace(0, len(examples) - 1, count, dtype=int)
        for idx in indices:
            selected.append(examples[idx])
        
        return selected
    
    def _run_tournament(self, population: List[PromptCandidate], 
                       dataset: Dataset, task_spec: TaskSpec) -> TournamentResult:
        """Run tournament evaluation on population."""
        # Sample test cases for evaluation
        test_samples = random.sample(dataset.examples, 
                                   min(10, len(dataset.examples)))
        
        matchup_results = defaultdict(dict)
        win_counts = defaultdict(int)
        total_matches = defaultdict(int)
        
        # Round-robin tournament
        for i, candidate_a in enumerate(population):
            for j, candidate_b in enumerate(population):
                if i >= j:  # Skip self and already compared pairs
                    continue
                
                # Compare on multiple test cases
                a_wins = 0
                b_wins = 0
                
                for test_case in test_samples:
                    # Generate responses
                    response_a = self._generate_response(candidate_a, test_case)
                    response_b = self._generate_response(candidate_b, test_case)
                    
                    # Judge comparison
                    criteria = self._create_judgment_criteria(task_spec)
                    judgment = self.judge.judge_comparison(
                        response_a, response_b, criteria, test_case.input
                    )
                    
                    if judgment["winner"] == "A":
                        a_wins += 1
                    elif judgment["winner"] == "B":
                        b_wins += 1
                
                # Determine overall winner
                if a_wins > b_wins:
                    matchup_results[candidate_a.id][candidate_b.id] = ComparisonResult.A_BETTER
                    matchup_results[candidate_b.id][candidate_a.id] = ComparisonResult.B_BETTER
                    win_counts[candidate_a.id] += 1
                elif b_wins > a_wins:
                    matchup_results[candidate_a.id][candidate_b.id] = ComparisonResult.B_BETTER
                    matchup_results[candidate_b.id][candidate_a.id] = ComparisonResult.A_BETTER
                    win_counts[candidate_b.id] += 1
                else:
                    matchup_results[candidate_a.id][candidate_b.id] = ComparisonResult.TIE
                    matchup_results[candidate_b.id][candidate_a.id] = ComparisonResult.TIE
                
                total_matches[candidate_a.id] += 1
                total_matches[candidate_b.id] += 1
        
        # Calculate win rates
        rankings = []
        for candidate in population:
            win_rate = win_counts[candidate.id] / total_matches[candidate.id] if total_matches[candidate.id] > 0 else 0
            rankings.append((candidate.id, win_rate))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return TournamentResult(
            rankings=rankings,
            matchup_results=dict(matchup_results),
            total_matches=len(population) * (len(population) - 1) // 2,
            metadata={"generation": self.generation_count}
        )
    
    def _generate_response(self, candidate: PromptCandidate, 
                         test_case: Example) -> str:
        """Generate response using candidate prompt."""
        # Format prompt with examples
        full_prompt = candidate.text + "\n\nExamples:\n"
        for ex in candidate.examples:
            full_prompt += f"\nInput: {ex.input}\nOutput: {ex.output}\n"
        
        full_prompt += f"\nNow process:\nInput: {test_case.input}\nOutput:"
        
        # Generate response
        response = self.llm_client.generate(
            full_prompt, 
            temperature=self.config.temperature_schedule[
                min(self.generation_count, len(self.config.temperature_schedule) - 1)
            ]
        )
        
        return response.text
    
    def _create_judgment_criteria(self, task_spec: TaskSpec) -> str:
        """Create criteria for judging responses."""
        criteria_parts = [
            f"Task accuracy: Does the response correctly complete the task '{task_spec.description}'?",
            "Format adherence: Does the response follow the specified output format?",
        ]
        
        if task_spec.constraints:
            criteria_parts.append("Constraint satisfaction: Does the response meet all specified constraints?")
        
        criteria_parts.append("Overall quality: Which response is better overall?")
        
        return " ".join(criteria_parts)
    
    def _update_fitness_scores(self, population: List[PromptCandidate], 
                             tournament_results: TournamentResult):
        """Update fitness scores based on tournament results."""
        win_rate_dict = dict(tournament_results.rankings)
        
        for candidate in population:
            candidate.win_rate = win_rate_dict.get(candidate.id, 0.0)
            
            # Calculate composite fitness score
            fitness_components = {
                "win_rate": candidate.win_rate * self.config.reward_weights["performance"],
                "format_score": candidate.metadata.get("format_score", 0.5) * self.config.reward_weights["format_adherence"],
                "efficiency": (1.0 - candidate.metadata.get("avg_cost", 0.5)) * self.config.reward_weights["cost_efficiency"]
            }
            
            candidate.fitness_score = sum(fitness_components.values())
    
    def _select_elite(self, population: List[PromptCandidate]) -> List[PromptCandidate]:
        """Select elite candidates based on fitness scores."""
        sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        elite_count = int(len(population) * self.config.elite_ratio)
        return sorted_population[:elite_count]
    
    def _evolve_population(self, elite: List[PromptCandidate], 
                         task_spec: TaskSpec, dataset: Dataset) -> List[PromptCandidate]:
        """Generate new candidates through evolution."""
        new_candidates = []
        candidate_id = len(elite) + self.generation_count * 100
        
        while len(new_candidates) < self.config.num_candidates - len(elite):
            if random.random() < self.config.crossover_rate and len(elite) >= 2:
                # Crossover
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover(parent1, parent2, candidate_id)
            else:
                # Mutation
                parent = random.choice(elite)
                child = self._mutate(parent, candidate_id, task_spec, dataset)
            
            new_candidates.append(child)
            candidate_id += 1
        
        return new_candidates
    
    def _crossover(self, parent1: PromptCandidate, parent2: PromptCandidate, 
                  candidate_id: int) -> PromptCandidate:
        """Crossover two parent prompts."""
        # Mix prompt components
        if random.random() < 0.5:
            # Take structure from parent1, examples from parent2
            text = parent1.text
            examples = parent2.examples
        else:
            # Take structure from parent2, examples from parent1
            text = parent2.text
            examples = parent1.examples
        
        # Potentially mix examples
        if len(parent1.examples) > 0 and len(parent2.examples) > 0:
            mixed_examples = []
            for i in range(max(len(parent1.examples), len(parent2.examples))):
                if i < len(parent1.examples) and i < len(parent2.examples):
                    mixed_examples.append(
                        random.choice([parent1.examples[i], parent2.examples[i]])
                    )
                elif i < len(parent1.examples):
                    mixed_examples.append(parent1.examples[i])
                else:
                    mixed_examples.append(parent2.examples[i])
            examples = mixed_examples
        
        return PromptCandidate(
            id=f"candidate_{candidate_id}",
            text=text,
            examples=examples,
            generation=self.generation_count + 1,
            metadata={"strategy": "crossover", "parents": [parent1.id, parent2.id]}
        )
    
    def _mutate(self, parent: PromptCandidate, candidate_id: int, 
               task_spec: TaskSpec, dataset: Dataset) -> PromptCandidate:
        """Mutate a parent prompt."""
        mutation_type = random.choice(["rephrase", "add_detail", "simplify", "reorder", "change_examples"])
        
        if mutation_type == "rephrase":
            # Rephrase the prompt
            mutated_text = self._rephrase_prompt(parent.text, task_spec)
            examples = parent.examples
        
        elif mutation_type == "add_detail":
            # Add more detail
            mutated_text = parent.text + f"\n\nRemember: {random.choice(task_spec.constraints).description if task_spec.constraints else 'Be precise and accurate.'}"
            examples = parent.examples
        
        elif mutation_type == "simplify":
            # Simplify the prompt
            lines = parent.text.split('\n')
            if len(lines) > 2:
                mutated_text = '\n'.join([lines[0], lines[-1]])
            else:
                mutated_text = parent.text
            examples = parent.examples
        
        elif mutation_type == "reorder":
            # Reorder prompt components
            lines = parent.text.split('\n')
            random.shuffle(lines)
            mutated_text = '\n'.join(lines)
            examples = parent.examples
        
        else:  # change_examples
            # Change examples
            mutated_text = parent.text
            examples = self._select_examples(dataset, len(parent.examples))
        
        return PromptCandidate(
            id=f"candidate_{candidate_id}",
            text=mutated_text,
            examples=examples,
            generation=self.generation_count + 1,
            metadata={"strategy": "mutation", "mutation_type": mutation_type, "parent": parent.id}
        )
    
    def _rephrase_prompt(self, original: str, task_spec: TaskSpec) -> str:
        """Rephrase prompt using LLM."""
        rephrase_prompt = f"""Rephrase the following prompt instruction while maintaining its meaning and requirements:

Original: {original}

Provide only the rephrased version, no explanation."""
        
        response = self.llm_client.generate(rephrase_prompt, temperature=0.8, max_tokens=300)
        return response.text.strip()
    
    def _select_best_candidate(self, population: List[PromptCandidate], 
                             final_results: TournamentResult) -> PromptCandidate:
        """Select the best candidate from final tournament."""
        # Get candidate with highest win rate
        best_id = final_results.rankings[0][0]
        for candidate in population:
            if candidate.id == best_id:
                return candidate
        
        # Fallback to highest fitness score
        return max(population, key=lambda x: x.fitness_score)
    
    def _candidate_to_optimized_prompt(self, candidate: PromptCandidate, 
                                     task_spec: TaskSpec) -> OptimizedPrompt:
        """Convert GRPO candidate to OptimizedPrompt."""
        return OptimizedPrompt(
            text=candidate.text,
            examples=candidate.examples,
            metadata={
                "optimizer": "grpo",
                "generation": candidate.generation,
                "fitness_score": candidate.fitness_score,
                "win_rate": candidate.win_rate,
                "strategy": candidate.metadata.get("strategy", "unknown"),
                "task": task_spec.name,
                "optimization_cost": self.llm_client.get_cost_summary()["total_cost"]
            },
            optimization_history=[{
                "generation": candidate.generation,
                "fitness": candidate.fitness_score,
                "win_rate": candidate.win_rate
            }]
        )
    
    def evaluate(self, prompt: OptimizedPrompt, test_set: Dataset) -> Metrics:
        """Evaluate optimized prompt on test set."""
        predictions = []
        costs = []
        
        for example in test_set.examples:
            # Format prompt with examples
            full_prompt = prompt.text + "\n\nExamples:\n"
            for ex in prompt.examples:
                full_prompt += f"\nInput: {ex.input}\nOutput: {ex.output}\n"
            full_prompt += f"\nNow process:\nInput: {example.input}\nOutput:"
            
            # Get prediction
            response = self.llm_client.generate(full_prompt, temperature=0.1)
            predictions.append(response.text)
            costs.append(response.cost)
        
        # Calculate metrics
        from ..core.metrics import AccuracyMetric, F1Metric
        
        ground_truth = [ex.output for ex in test_set.examples]
        
        accuracy_metric = AccuracyMetric()
        f1_metric = F1Metric()
        
        scores = {
            "accuracy": accuracy_metric.compute(predictions, ground_truth),
            "f1": f1_metric.compute(predictions, ground_truth),
            "average_cost": sum(costs) / len(costs) if costs else 0,
            "total_cost": sum(costs)
        }
        
        return Metrics(
            scores=scores,
            metadata={
                "test_set_size": len(test_set),
                "optimizer": "grpo"
            }
        )