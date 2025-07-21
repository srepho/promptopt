"""Visualization utilities for optimization results."""

from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime


class ResultsVisualizer:
    """Visualize optimization results and comparisons."""
    
    def __init__(self, style: str = "default"):
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        try:
            sns.set_palette("husl")
        except:
            pass
    
    def plot_optimizer_comparison(self, results: Dict[str, Dict[str, float]], 
                                 metrics: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create multi-metric comparison across optimizers."""
        if not metrics:
            metrics = list(next(iter(results.values())).keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            optimizers = list(results.keys())
            scores = [results[opt].get(metric, 0) for opt in optimizers]
            
            bars = ax.bar(optimizers, scores)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel('Score')
            ax.set_xlabel('Optimizer')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_curves(self, optimization_history: Dict[str, List[Dict]], 
                           metric: str = "score",
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot optimization progress over iterations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for optimizer_name, history in optimization_history.items():
            iterations = range(len(history))
            scores = [h.get(metric, 0) for h in history]
            ax.plot(iterations, scores, marker='o', label=optimizer_name)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title('Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_tournament_results(self, tournament_results: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Visualize tournament bracket and results."""
        rankings = tournament_results.get('rankings', [])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Win rates bar chart
        if rankings:
            names = [r[0] for r in rankings]
            win_rates = [r[1] for r in rankings]
            
            bars = ax1.barh(names, win_rates)
            ax1.set_xlabel('Win Rate')
            ax1.set_title('Tournament Rankings')
            ax1.set_xlim(0, 1)
            
            # Add value labels
            for bar, rate in zip(bars, win_rates):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{rate:.2%}', ha='left', va='center')
        
        # Head-to-head matrix
        matchup_results = tournament_results.get('matchup_results', {})
        if matchup_results:
            participants = list(matchup_results.keys())
            matrix = np.zeros((len(participants), len(participants)))
            
            for i, p1 in enumerate(participants):
                for j, p2 in enumerate(participants):
                    if p2 in matchup_results[p1]:
                        result = matchup_results[p1][p2]
                        if result == 'win':
                            matrix[i, j] = 1
                        elif result == 'loss':
                            matrix[i, j] = 0
                        else:
                            matrix[i, j] = 0.5
            
            sns.heatmap(matrix, xticklabels=participants, yticklabels=participants,
                       annot=True, fmt='.1f', cmap='RdYlGn', center=0.5,
                       ax=ax2, cbar_kws={'label': 'Win Rate'})
            ax2.set_title('Head-to-Head Results')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cost_analysis(self, cost_data: Dict[str, Dict[str, float]],
                          save_path: Optional[str] = None) -> plt.Figure:
        """Plot cost analysis across optimizers."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Total costs
        optimizers = list(cost_data.keys())
        total_costs = [cost_data[opt].get('total_cost', 0) for opt in optimizers]
        
        bars = ax1.bar(optimizers, total_costs)
        ax1.set_ylabel('Total Cost ($)')
        ax1.set_title('Optimization Costs')
        
        for bar, cost in zip(bars, total_costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${cost:.2f}', ha='center', va='bottom')
        
        # Cost vs Performance scatter
        if all('performance_score' in cost_data[opt] for opt in optimizers):
            performances = [cost_data[opt]['performance_score'] for opt in optimizers]
            ax2.scatter(total_costs, performances, s=100)
            
            for opt, cost, perf in zip(optimizers, total_costs, performances):
                ax2.annotate(opt, (cost, perf), xytext=(5, 5), 
                           textcoords='offset points')
            
            ax2.set_xlabel('Total Cost ($)')
            ax2.set_ylabel('Performance Score')
            ax2.set_title('Cost vs Performance Trade-off')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_constraint_satisfaction(self, constraint_results: Dict[str, Dict[str, float]],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot constraint satisfaction across optimizers."""
        df_data = []
        for optimizer, constraints in constraint_results.items():
            for constraint, score in constraints.items():
                df_data.append({
                    'Optimizer': optimizer,
                    'Constraint': constraint,
                    'Score': score
                })
        
        df = pd.DataFrame(df_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create grouped bar chart
        pivot_df = df.pivot(index='Constraint', columns='Optimizer', values='Score')
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_ylabel('Satisfaction Score')
        ax.set_title('Constraint Satisfaction by Optimizer')
        ax.legend(title='Optimizer')
        ax.set_ylim(0, 1.1)
        
        # Add horizontal line at 1.0
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, 
                  label='Perfect Satisfaction')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_optimization_report(self, results: Dict[str, Any],
                                 save_dir: str = './reports') -> Dict[str, str]:
        """Create comprehensive visual report of optimization results."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        report_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Optimizer comparison
        if 'optimizer_scores' in results:
            path = f"{save_dir}/optimizer_comparison_{timestamp}.png"
            self.plot_optimizer_comparison(results['optimizer_scores'], save_path=path)
            report_paths['optimizer_comparison'] = path
        
        # Learning curves
        if 'optimization_history' in results:
            path = f"{save_dir}/learning_curves_{timestamp}.png"
            self.plot_learning_curves(results['optimization_history'], save_path=path)
            report_paths['learning_curves'] = path
        
        # Tournament results
        if 'tournament_results' in results:
            path = f"{save_dir}/tournament_results_{timestamp}.png"
            self.plot_tournament_results(results['tournament_results'], save_path=path)
            report_paths['tournament_results'] = path
        
        # Cost analysis
        if 'cost_data' in results:
            path = f"{save_dir}/cost_analysis_{timestamp}.png"
            self.plot_cost_analysis(results['cost_data'], save_path=path)
            report_paths['cost_analysis'] = path
        
        # Constraint satisfaction
        if 'constraint_results' in results:
            path = f"{save_dir}/constraint_satisfaction_{timestamp}.png"
            self.plot_constraint_satisfaction(results['constraint_results'], save_path=path)
            report_paths['constraint_satisfaction'] = path
        
        return report_paths


class InteractiveDashboard:
    """Create interactive dashboard for Colab/Jupyter environments."""
    
    @staticmethod
    def create_progress_display(optimizer_name: str, iteration: int, 
                              total_iterations: int, current_score: float):
        """Create progress display for notebook environments."""
        from IPython.display import clear_output, display, HTML
        
        clear_output(wait=True)
        
        progress = iteration / total_iterations
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        html = f"""
        <div style="font-family: monospace;">
            <h3>{optimizer_name} Optimization Progress</h3>
            <p>Iteration: {iteration}/{total_iterations}</p>
            <p>Progress: [{bar}] {progress:.1%}</p>
            <p>Current Score: {current_score:.4f}</p>
        </div>
        """
        
        display(HTML(html))