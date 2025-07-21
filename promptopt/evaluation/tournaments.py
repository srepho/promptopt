"""Tournament evaluation system for prompt optimization."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
from collections import defaultdict
import logging

from ..core.interfaces import ComparisonResult, JudgmentResult, TournamentResult
from ..utils.llm_clients import LLMJudge


@dataclass
class TournamentConfig:
    """Configuration for tournament evaluation."""
    tournament_type: str = "round_robin"  # or "single_elimination", "swiss"
    matches_per_pair: int = 3
    judgment_criteria: Optional[str] = None
    confidence_threshold: float = 0.6
    parallel_matches: bool = True


class TournamentManager:
    """Manages tournament-style evaluation of prompts."""
    
    def __init__(self, judge: LLMJudge, config: Optional[TournamentConfig] = None):
        self.judge = judge
        self.config = config or TournamentConfig()
        self.match_history = []
    
    def run_round_robin(self, prompts: List[Any], test_cases: List[Any],
                       criteria: Optional[str] = None) -> TournamentResult:
        """Run round-robin tournament where every prompt faces every other."""
        matchup_results = defaultdict(dict)
        detailed_results = []
        
        # Generate all pairings
        pairings = []
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                pairings.append((i, j))
        
        # Run matches
        for i, j in pairings:
            prompt_a = prompts[i]
            prompt_b = prompts[j]
            
            # Run multiple matches with different test cases
            match_results = []
            test_sample = random.sample(test_cases, 
                                      min(self.config.matches_per_pair, len(test_cases)))
            
            for test_case in test_sample:
                result = self._run_single_match(
                    prompt_a, prompt_b, test_case, 
                    criteria or self.config.judgment_criteria
                )
                match_results.append(result)
            
            # Aggregate match results
            aggregate_result = self._aggregate_match_results(match_results)
            
            # Store results
            prompt_a_id = getattr(prompt_a, 'id', f'prompt_{i}')
            prompt_b_id = getattr(prompt_b, 'id', f'prompt_{j}')
            
            matchup_results[prompt_a_id][prompt_b_id] = aggregate_result
            matchup_results[prompt_b_id][prompt_a_id] = self._reverse_result(aggregate_result)
            
            detailed_results.append({
                'prompt_a': prompt_a_id,
                'prompt_b': prompt_b_id,
                'result': aggregate_result,
                'matches': match_results
            })
        
        # Calculate rankings
        rankings = self._calculate_rankings(prompts, matchup_results)
        
        return TournamentResult(
            rankings=rankings,
            matchup_results=dict(matchup_results),
            total_matches=len(pairings) * self.config.matches_per_pair,
            metadata={
                'tournament_type': 'round_robin',
                'detailed_results': detailed_results
            }
        )
    
    def run_single_elimination(self, prompts: List[Any], test_cases: List[Any],
                             criteria: Optional[str] = None) -> TournamentResult:
        """Run single elimination tournament."""
        if len(prompts) < 2:
            raise ValueError("Need at least 2 prompts for tournament")
        
        # Initialize bracket
        current_round = list(prompts)
        round_num = 0
        bracket_results = []
        
        while len(current_round) > 1:
            next_round = []
            round_results = []
            
            # Pair up competitors
            for i in range(0, len(current_round), 2):
                if i + 1 < len(current_round):
                    prompt_a = current_round[i]
                    prompt_b = current_round[i + 1]
                    
                    # Run match
                    match_results = []
                    test_sample = random.sample(test_cases, 
                                              min(self.config.matches_per_pair, len(test_cases)))
                    
                    for test_case in test_sample:
                        result = self._run_single_match(
                            prompt_a, prompt_b, test_case,
                            criteria or self.config.judgment_criteria
                        )
                        match_results.append(result)
                    
                    # Determine winner
                    winner = self._determine_match_winner(prompt_a, prompt_b, match_results)
                    next_round.append(winner)
                    
                    round_results.append({
                        'match': (getattr(prompt_a, 'id', str(i)), 
                                 getattr(prompt_b, 'id', str(i+1))),
                        'winner': getattr(winner, 'id', 'unknown'),
                        'results': match_results
                    })
                else:
                    # Bye - advance directly
                    next_round.append(current_round[i])
            
            bracket_results.append({
                'round': round_num,
                'matches': round_results
            })
            
            current_round = next_round
            round_num += 1
        
        # Create final rankings
        champion = current_round[0] if current_round else None
        rankings = [(getattr(champion, 'id', 'unknown'), 1.0)] if champion else []
        
        return TournamentResult(
            rankings=rankings,
            matchup_results={},
            total_matches=sum(len(r['matches']) for r in bracket_results) * self.config.matches_per_pair,
            metadata={
                'tournament_type': 'single_elimination',
                'bracket': bracket_results,
                'champion': getattr(champion, 'id', 'unknown') if champion else None
            }
        )
    
    def run_swiss(self, prompts: List[Any], test_cases: List[Any],
                  rounds: int = 3, criteria: Optional[str] = None) -> TournamentResult:
        """Run Swiss-system tournament."""
        # Track scores and opponents faced
        scores = {getattr(p, 'id', f'prompt_{i}'): 0.0 for i, p in enumerate(prompts)}
        opponents_faced = defaultdict(set)
        all_matchups = defaultdict(dict)
        
        for round_num in range(rounds):
            # Pair prompts based on current scores
            pairings = self._swiss_pairing(prompts, scores, opponents_faced)
            
            # Run matches for this round
            for prompt_a, prompt_b in pairings:
                prompt_a_id = getattr(prompt_a, 'id', 'unknown')
                prompt_b_id = getattr(prompt_b, 'id', 'unknown')
                
                # Skip if already played
                if prompt_b_id in opponents_faced[prompt_a_id]:
                    continue
                
                # Run match
                match_results = []
                test_sample = random.sample(test_cases,
                                          min(self.config.matches_per_pair, len(test_cases)))
                
                for test_case in test_sample:
                    result = self._run_single_match(
                        prompt_a, prompt_b, test_case,
                        criteria or self.config.judgment_criteria
                    )
                    match_results.append(result)
                
                # Update scores
                aggregate_result = self._aggregate_match_results(match_results)
                if aggregate_result == ComparisonResult.A_BETTER:
                    scores[prompt_a_id] += 1.0
                elif aggregate_result == ComparisonResult.B_BETTER:
                    scores[prompt_b_id] += 1.0
                else:  # Tie
                    scores[prompt_a_id] += 0.5
                    scores[prompt_b_id] += 0.5
                
                # Track opponents
                opponents_faced[prompt_a_id].add(prompt_b_id)
                opponents_faced[prompt_b_id].add(prompt_a_id)
                
                # Store matchup results
                all_matchups[prompt_a_id][prompt_b_id] = aggregate_result
                all_matchups[prompt_b_id][prompt_a_id] = self._reverse_result(aggregate_result)
        
        # Create final rankings
        rankings = [(pid, score / rounds) for pid, score in sorted(scores.items(), 
                                                                  key=lambda x: x[1], 
                                                                  reverse=True)]
        
        return TournamentResult(
            rankings=rankings,
            matchup_results=dict(all_matchups),
            total_matches=sum(len(opponents) for opponents in opponents_faced.values()) // 2 * self.config.matches_per_pair,
            metadata={
                'tournament_type': 'swiss',
                'rounds': rounds,
                'final_scores': scores
            }
        )
    
    def _run_single_match(self, prompt_a: Any, prompt_b: Any, 
                         test_case: Any, criteria: str) -> Dict[str, Any]:
        """Run a single match between two prompts."""
        # Generate responses (assuming prompts have a generate method or similar)
        if hasattr(prompt_a, 'generate'):
            response_a = prompt_a.generate(test_case)
        else:
            response_a = f"Response from {getattr(prompt_a, 'id', 'A')} for {test_case}"
        
        if hasattr(prompt_b, 'generate'):
            response_b = prompt_b.generate(test_case)
        else:
            response_b = f"Response from {getattr(prompt_b, 'id', 'B')} for {test_case}"
        
        # Judge comparison
        judgment = self.judge.judge_comparison(
            response_a, response_b, criteria,
            context=str(test_case) if test_case else None
        )
        
        # Store match details
        match_result = {
            'test_case': str(test_case)[:100],  # Truncate for storage
            'winner': judgment['winner'],
            'confidence': judgment['confidence'],
            'reasoning': judgment['reasoning'],
            'cost': judgment.get('cost', 0)
        }
        
        self.match_history.append(match_result)
        
        return match_result
    
    def _aggregate_match_results(self, match_results: List[Dict[str, Any]]) -> ComparisonResult:
        """Aggregate multiple match results into overall result."""
        a_wins = sum(1 for r in match_results if r['winner'] == 'A')
        b_wins = sum(1 for r in match_results if r['winner'] == 'B')
        
        # Consider confidence scores
        a_confidence = sum(r['confidence'] for r in match_results if r['winner'] == 'A')
        b_confidence = sum(r['confidence'] for r in match_results if r['winner'] == 'B')
        
        # Weighted decision
        if a_wins > b_wins:
            return ComparisonResult.A_BETTER
        elif b_wins > a_wins:
            return ComparisonResult.B_BETTER
        else:
            # Tie breaker by confidence
            if a_confidence > b_confidence:
                return ComparisonResult.A_BETTER
            elif b_confidence > a_confidence:
                return ComparisonResult.B_BETTER
            else:
                return ComparisonResult.TIE
    
    def _reverse_result(self, result: ComparisonResult) -> ComparisonResult:
        """Reverse a comparison result."""
        if result == ComparisonResult.A_BETTER:
            return ComparisonResult.B_BETTER
        elif result == ComparisonResult.B_BETTER:
            return ComparisonResult.A_BETTER
        else:
            return ComparisonResult.TIE
    
    def _calculate_rankings(self, prompts: List[Any], 
                          matchup_results: Dict[str, Dict[str, ComparisonResult]]) -> List[Tuple[str, float]]:
        """Calculate rankings based on matchup results."""
        win_counts = defaultdict(int)
        total_matches = defaultdict(int)
        
        for prompt_id, opponents in matchup_results.items():
            for opponent_id, result in opponents.items():
                total_matches[prompt_id] += 1
                if result == ComparisonResult.A_BETTER:
                    win_counts[prompt_id] += 1
                elif result == ComparisonResult.TIE:
                    win_counts[prompt_id] += 0.5
        
        # Calculate win rates
        rankings = []
        for i, prompt in enumerate(prompts):
            prompt_id = getattr(prompt, 'id', f'prompt_{i}')
            if total_matches[prompt_id] > 0:
                win_rate = win_counts[prompt_id] / total_matches[prompt_id]
            else:
                win_rate = 0.0
            rankings.append((prompt_id, win_rate))
        
        # Sort by win rate
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _determine_match_winner(self, prompt_a: Any, prompt_b: Any,
                              match_results: List[Dict[str, Any]]) -> Any:
        """Determine winner from match results."""
        aggregate = self._aggregate_match_results(match_results)
        
        if aggregate == ComparisonResult.A_BETTER:
            return prompt_a
        elif aggregate == ComparisonResult.B_BETTER:
            return prompt_b
        else:
            # Tie - could use additional criteria or random selection
            return random.choice([prompt_a, prompt_b])
    
    def _swiss_pairing(self, prompts: List[Any], scores: Dict[str, float],
                      opponents_faced: Dict[str, set]) -> List[Tuple[Any, Any]]:
        """Create pairings for Swiss tournament round."""
        # Sort by current score
        sorted_prompts = sorted(prompts, 
                              key=lambda p: scores[getattr(p, 'id', 'unknown')], 
                              reverse=True)
        
        pairings = []
        paired = set()
        
        for i, prompt_a in enumerate(sorted_prompts):
            if prompt_a in paired:
                continue
            
            prompt_a_id = getattr(prompt_a, 'id', f'prompt_{i}')
            
            # Find best opponent (closest score, not yet faced)
            for j, prompt_b in enumerate(sorted_prompts[i+1:], i+1):
                if prompt_b in paired:
                    continue
                
                prompt_b_id = getattr(prompt_b, 'id', f'prompt_{j}')
                
                if prompt_b_id not in opponents_faced[prompt_a_id]:
                    pairings.append((prompt_a, prompt_b))
                    paired.add(prompt_a)
                    paired.add(prompt_b)
                    break
        
        return pairings
    
    def compute_win_rates(self, results: TournamentResult) -> Dict[str, float]:
        """Extract win rates from tournament results."""
        return dict(results.rankings)
    
    def get_match_statistics(self) -> Dict[str, Any]:
        """Get statistics about matches run."""
        if not self.match_history:
            return {}
        
        total_cost = sum(m.get('cost', 0) for m in self.match_history)
        avg_confidence = sum(m['confidence'] for m in self.match_history) / len(self.match_history)
        
        winner_distribution = defaultdict(int)
        for match in self.match_history:
            winner_distribution[match['winner']] += 1
        
        return {
            'total_matches': len(self.match_history),
            'total_cost': total_cost,
            'average_confidence': avg_confidence,
            'winner_distribution': dict(winner_distribution),
            'tie_rate': winner_distribution.get('tie', 0) / len(self.match_history) if self.match_history else 0
        }