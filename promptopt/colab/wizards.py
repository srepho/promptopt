"""Interactive wizards for Colab environments."""

from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass


@dataclass
class WizardStep:
    """Represents a step in a wizard."""
    name: str
    prompt: str
    options: Optional[List[str]] = None
    default: Optional[Any] = None
    validator: Optional[callable] = None


class DataGenerationWizard:
    """Interactive wizard for synthetic data generation."""
    
    def __init__(self, is_colab: bool = False):
        self.is_colab = is_colab
        self.steps = self._define_steps()
    
    def _define_steps(self) -> List[WizardStep]:
        """Define wizard steps."""
        return [
            WizardStep(
                name="scenario_type",
                prompt="What type of business scenario?",
                options=["customer_support", "internal_email", "content_creation", "data_analysis"],
                default="customer_support"
            ),
            WizardStep(
                name="industry",
                prompt="What industry?",
                options=["technology", "healthcare", "finance", "retail", "education", "other"],
                default="technology"
            ),
            WizardStep(
                name="company_size",
                prompt="Company size?",
                options=["startup", "smb", "enterprise"],
                default="enterprise"
            ),
            WizardStep(
                name="data_count",
                prompt="How many examples to generate?",
                default=50,
                validator=lambda x: 1 <= int(x) <= 1000
            ),
            WizardStep(
                name="compliance",
                prompt="Compliance requirements? (comma-separated)",
                options=["SOC2", "GDPR", "HIPAA", "PCI", "None"],
                default="SOC2"
            )
        ]
    
    def run(self) -> Dict[str, Any]:
        """Run the wizard."""
        print("🧙 Synthetic Data Generation Wizard\n")
        
        config = {}
        
        for step in self.steps:
            value = self._process_step(step)
            config[step.name] = value
        
        # Process special fields
        if "compliance" in config and config["compliance"] != "None":
            config["compliance_requirements"] = [c.strip() for c in config["compliance"].split(",")]
        else:
            config["compliance_requirements"] = []
        
        print("\n✅ Configuration complete!")
        return config
    
    def _process_step(self, step: WizardStep) -> Any:
        """Process a single wizard step."""
        if self.is_colab:
            return self._process_colab_step(step)
        else:
            return self._process_cli_step(step)
    
    def _process_colab_step(self, step: WizardStep) -> Any:
        """Process step in Colab environment."""
        # In real Colab, would use forms/widgets
        # For now, simulate with defaults
        print(f"{step.prompt}")
        if step.options:
            print(f"Options: {', '.join(step.options)}")
        print(f"Selected: {step.default}")
        return step.default
    
    def _process_cli_step(self, step: WizardStep) -> Any:
        """Process step in CLI environment."""
        print(f"\n{step.prompt}")
        if step.options:
            print(f"Options: {', '.join(step.options)}")
            print(f"Default: {step.default}")
        
        # For demo, use default
        return step.default


class OptimizationWizard:
    """Interactive wizard for optimization configuration."""
    
    def __init__(self, is_colab: bool = False):
        self.is_colab = is_colab
    
    def run(self) -> Dict[str, Any]:
        """Run optimization configuration wizard."""
        print("⚙️  Optimization Configuration Wizard\n")
        
        config = {
            "optimizer": self._select_optimizer(),
            "budget_limit": self._set_budget(),
            "quality_threshold": self._set_quality_threshold(),
            "constraints": self._configure_constraints()
        }
        
        print("\n✅ Optimization configured!")
        return config
    
    def _select_optimizer(self) -> str:
        """Select optimization strategy."""
        strategies = {
            "1": "dspy",
            "2": "grpo", 
            "3": "sequential_hybrid",
            "4": "ensemble_hybrid",
            "5": "feedback_hybrid",
            "6": "cost_aware_hybrid"
        }
        
        print("Select optimization strategy:")
        for key, value in strategies.items():
            print(f"  {key}. {value}")
        
        # Default to sequential hybrid
        return strategies.get("3", "sequential_hybrid")
    
    def _set_budget(self) -> float:
        """Set optimization budget."""
        print("\nOptimization budget ($):")
        print("  Suggested: $5-10 for testing, $20-50 for production")
        return 10.0  # Default
    
    def _set_quality_threshold(self) -> float:
        """Set quality threshold."""
        print("\nMinimum quality threshold (0-1):")
        print("  0.8 = Good quality, 0.9 = High quality")
        return 0.85  # Default
    
    def _configure_constraints(self) -> List[str]:
        """Configure optimization constraints."""
        print("\nOptimization constraints:")
        constraints = [
            "maintain_format",
            "ensure_conciseness",
            "preserve_tone",
            "minimize_cost"
        ]
        
        # Default to all constraints
        return constraints


class DeploymentWizard:
    """Interactive wizard for deployment configuration."""
    
    def __init__(self, is_colab: bool = False):
        self.is_colab = is_colab
    
    def run(self, optimized_prompt: Any) -> Dict[str, Any]:
        """Run deployment configuration wizard."""
        print("🚀 Deployment Configuration Wizard\n")
        
        config = {
            "deployment_target": self._select_target(),
            "team_size": self._set_team_size(),
            "rollout_phases": self._configure_rollout(),
            "monitoring": self._configure_monitoring()
        }
        
        # Generate deployment assets
        assets = self._generate_deployment_assets(optimized_prompt, config)
        config["assets"] = assets
        
        print("\n✅ Deployment package ready!")
        return config
    
    def _select_target(self) -> List[str]:
        """Select deployment targets."""
        targets = ["chatgpt", "api", "slack", "teams", "custom"]
        print("Deployment targets:")
        for t in targets:
            print(f"  - {t}")
        
        # Default to ChatGPT and API
        return ["chatgpt", "api"]
    
    def _set_team_size(self) -> int:
        """Set team size for deployment."""
        print("\nTeam size for deployment:")
        return 10  # Default
    
    def _configure_rollout(self) -> List[Dict[str, Any]]:
        """Configure rollout phases."""
        return [
            {"phase": 1, "coverage": "10%", "duration": "1 week"},
            {"phase": 2, "coverage": "50%", "duration": "2 weeks"},
            {"phase": 3, "coverage": "100%", "duration": "ongoing"}
        ]
    
    def _configure_monitoring(self) -> Dict[str, Any]:
        """Configure monitoring settings."""
        return {
            "metrics": ["quality_score", "response_time", "cost_per_query"],
            "alerts": {"quality_below": 0.8, "cost_above": 0.10},
            "reporting": "weekly"
        }
    
    def _generate_deployment_assets(self, prompt: Any, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate deployment assets."""
        assets = {}
        
        if "chatgpt" in config["deployment_target"]:
            assets["chatgpt_instructions"] = self._create_chatgpt_instructions(prompt)
        
        if "api" in config["deployment_target"]:
            assets["api_code"] = self._create_api_code(prompt)
        
        if "slack" in config["deployment_target"]:
            assets["slack_config"] = self._create_slack_config(prompt)
        
        return assets
    
    def _create_chatgpt_instructions(self, prompt: Any) -> str:
        """Create ChatGPT custom instructions."""
        return f"""Custom Instructions for ChatGPT:

{getattr(prompt, 'text', 'Optimized prompt text here')}

Examples to follow:
[Include 2-3 examples from optimization]

Remember to maintain consistency and quality in all responses."""
    
    def _create_api_code(self, prompt: Any) -> str:
        """Create API integration code."""
        return """# API Integration Code

import openai

def get_optimized_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": OPTIMIZED_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content
"""
    
    def _create_slack_config(self, prompt: Any) -> str:
        """Create Slack bot configuration."""
        return """# Slack Bot Configuration

1. Create Slack App at api.slack.com
2. Add Bot Token Scopes: chat:write, im:history
3. Install to workspace
4. Use optimized prompt as bot's system message
"""


class ResultsAnalysisWizard:
    """Interactive wizard for results analysis."""
    
    def __init__(self, is_colab: bool = False):
        self.is_colab = is_colab
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results interactively."""
        print("📊 Results Analysis Wizard\n")
        
        analysis = {
            "summary": self._generate_summary(results),
            "insights": self._extract_insights(results),
            "recommendations": self._generate_recommendations(results),
            "next_steps": self._suggest_next_steps(results)
        }
        
        if self.is_colab:
            self._create_interactive_report(analysis)
        else:
            self._print_analysis(analysis)
        
        return analysis
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate results summary."""
        return {
            "optimizer_used": results.get("optimizer", "Unknown"),
            "final_score": results.get("final_score", 0.0),
            "improvement": results.get("improvement_percentage", 0.0),
            "total_cost": results.get("total_cost", 0.0)
        }
    
    def _extract_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract key insights from results."""
        insights = []
        
        if results.get("improvement_percentage", 0) > 20:
            insights.append("Significant improvement achieved (>20%)")
        
        if results.get("total_cost", 0) < 5.0:
            insights.append("Cost-effective optimization completed")
        
        if results.get("constraint_adherence", 0) > 0.9:
            insights.append("Excellent constraint adherence achieved")
        
        return insights
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if results.get("final_score", 0) < 0.8:
            recommendations.append("Consider running ensemble optimization for better quality")
        
        if results.get("total_cost", 0) > 20:
            recommendations.append("Use cost-aware optimization for production deployment")
        
        recommendations.append("Test with real user data before full deployment")
        
        return recommendations
    
    def _suggest_next_steps(self, results: Dict[str, Any]) -> List[str]:
        """Suggest next steps."""
        return [
            "Run A/B test with optimized prompt",
            "Create deployment package for team",
            "Set up monitoring dashboard",
            "Schedule follow-up optimization in 30 days"
        ]
    
    def _create_interactive_report(self, analysis: Dict[str, Any]):
        """Create interactive report for Colab."""
        try:
            from IPython.display import display, Markdown
            
            report = f"""
# 📊 Optimization Results Analysis

## Summary
- **Optimizer**: {analysis['summary']['optimizer_used']}
- **Final Score**: {analysis['summary']['final_score']:.2%}
- **Improvement**: {analysis['summary']['improvement']:.1%}
- **Total Cost**: ${analysis['summary']['total_cost']:.2f}

## Key Insights
{chr(10).join(f"- {insight}" for insight in analysis['insights'])}

## Recommendations
{chr(10).join(f"1. {rec}" for i, rec in enumerate(analysis['recommendations'], 1))}

## Next Steps
{chr(10).join(f"- [ ] {step}" for step in analysis['next_steps'])}
"""
            display(Markdown(report))
        except ImportError:
            self._print_analysis(analysis)
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis for CLI."""
        print("\nSummary:")
        for key, value in analysis['summary'].items():
            print(f"  {key}: {value}")
        
        print("\nKey Insights:")
        for insight in analysis['insights']:
            print(f"  - {insight}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\nNext Steps:")
        for step in analysis['next_steps']:
            print(f"  - {step}")