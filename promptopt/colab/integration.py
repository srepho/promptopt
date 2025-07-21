"""Integration utilities for Google Colab environments."""

import os
import json
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
import datetime


class DriveIntegration:
    """Google Drive integration for Colab."""
    
    def __init__(self, base_path: str = "/content/drive/MyDrive/PromptOpt"):
        self.base_path = base_path
        self.is_mounted = self._check_drive_mounted()
    
    def _check_drive_mounted(self) -> bool:
        """Check if Google Drive is mounted."""
        return os.path.exists("/content/drive")
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        if not self.is_mounted:
            print("⚠️  Google Drive not mounted")
            return
        
        directories = [
            self.base_path,
            f"{self.base_path}/data",
            f"{self.base_path}/results",
            f"{self.base_path}/prompts",
            f"{self.base_path}/deployments",
            f"{self.base_path}/backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("✓ Drive directories ready")
    
    def save_results(self, results: Dict[str, Any], name: str) -> str:
        """Save results to Drive."""
        if not self.is_mounted:
            return self._save_local(results, name)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = f"{self.base_path}/results/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Saved to Drive: {filepath}")
        return filepath
    
    def _save_local(self, results: Dict[str, Any], name: str) -> str:
        """Save locally when Drive not available."""
        Path("./local_results").mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./local_results/{name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Saved locally: {filename}")
        return filename
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from Drive."""
        filepath = f"{self.base_path}/results/{filename}"
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Results file not found: {filepath}")
    
    def list_results(self) -> List[str]:
        """List available results files."""
        results_dir = f"{self.base_path}/results"
        
        if os.path.exists(results_dir):
            files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            return sorted(files, reverse=True)
        else:
            return []
    
    def backup_optimization(self, optimizer_state: Any, name: str) -> str:
        """Backup optimizer state."""
        if not self.is_mounted:
            print("⚠️  Drive not mounted, skipping backup")
            return ""
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        filepath = f"{self.base_path}/backups/{filename}"
        
        with open(filepath, 'wb') as f:
            pickle.dump(optimizer_state, f)
        
        print(f"✓ Backed up to: {filepath}")
        return filepath
    
    def restore_optimization(self, filepath: str) -> Any:
        """Restore optimizer state from backup."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class SharingUtilities:
    """Utilities for sharing results and prompts."""
    
    def __init__(self, drive_integration: Optional[DriveIntegration] = None):
        self.drive = drive_integration
    
    def create_shareable_link(self, filepath: str) -> str:
        """Create shareable link for file."""
        # In real implementation, would use Google Drive API
        # For now, return instruction
        return f"To share: Right-click '{filepath}' in Drive → Get link"
    
    def export_for_team(self, optimized_prompt: Any, format: str = "all") -> Dict[str, str]:
        """Export optimized prompt for team use."""
        exports = {}
        
        if format in ["all", "json"]:
            exports["json"] = self._export_json(optimized_prompt)
        
        if format in ["all", "markdown"]:
            exports["markdown"] = self._export_markdown(optimized_prompt)
        
        if format in ["all", "python"]:
            exports["python"] = self._export_python(optimized_prompt)
        
        return exports
    
    def _export_json(self, prompt: Any) -> str:
        """Export as JSON."""
        data = {
            "prompt_text": getattr(prompt, 'text', ''),
            "examples": [{"input": ex.input, "output": ex.output} 
                        for ex in getattr(prompt, 'examples', [])],
            "metadata": getattr(prompt, 'metadata', {})
        }
        return json.dumps(data, indent=2)
    
    def _export_markdown(self, prompt: Any) -> str:
        """Export as Markdown."""
        md = f"""# Optimized Prompt

## Instructions
{getattr(prompt, 'text', 'Prompt text here')}

## Examples
"""
        
        for i, ex in enumerate(getattr(prompt, 'examples', [])[:3], 1):
            md += f"""
### Example {i}
**Input:** {ex.input}
**Output:** {ex.output}
"""
        
        return md
    
    def _export_python(self, prompt: Any) -> str:
        """Export as Python code."""
        return f'''"""Optimized prompt for production use."""

SYSTEM_PROMPT = """{getattr(prompt, 'text', '')}"""

EXAMPLES = {[{{"input": ex.input, "output": ex.output}} for ex in getattr(prompt, 'examples', [])[:3]]}

def get_response(user_input: str) -> str:
    """Get response using optimized prompt."""
    # Add your LLM integration here
    pass
'''


class SecretsManager:
    """Manage secrets in Colab environment."""
    
    def __init__(self):
        self.secrets = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from various sources."""
        # Try Colab secrets first
        try:
            from google.colab import userdata
            for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                try:
                    self.secrets[key] = userdata.get(key)
                except:
                    pass
        except ImportError:
            pass
        
        # Fall back to environment variables
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if key not in self.secrets and os.getenv(key):
                self.secrets[key] = os.getenv(key)
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value."""
        return self.secrets.get(key)
    
    def set_secret(self, key: str, value: str):
        """Set a secret value."""
        self.secrets[key] = value
        os.environ[key] = value
    
    def has_llm_keys(self) -> bool:
        """Check if any LLM API keys are available."""
        return any(key in self.secrets for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        providers = []
        if "OPENAI_API_KEY" in self.secrets:
            providers.append("openai")
        if "ANTHROPIC_API_KEY" in self.secrets:
            providers.append("anthropic")
        return providers


class NotebookUtilities:
    """Utilities specific to notebook environments."""
    
    @staticmethod
    def display_progress(current: int, total: int, message: str = ""):
        """Display progress bar in notebook."""
        try:
            from IPython.display import clear_output, display, HTML
            
            progress = current / total
            bar_length = 50
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            html = f"""
            <div style="font-family: monospace; margin: 10px 0;">
                <div>{message}</div>
                <div>[{bar}] {progress:.1%}</div>
                <div>Step {current}/{total}</div>
            </div>
            """
            
            clear_output(wait=True)
            display(HTML(html))
        except ImportError:
            # Fallback for non-notebook environments
            print(f"{message} - {current}/{total} ({current/total:.1%})")
    
    @staticmethod
    def create_download_link(content: str, filename: str, link_text: str = "Download"):
        """Create download link in notebook."""
        try:
            from IPython.display import display, HTML
            import base64
            
            b64 = base64.b64encode(content.encode()).decode()
            href = f'<a download="{filename}" href="data:text/plain;base64,{b64}">{link_text}</a>'
            
            display(HTML(href))
        except ImportError:
            print(f"Content saved to: {filename}")
            with open(filename, 'w') as f:
                f.write(content)
    
    @staticmethod
    def display_results_summary(results: Dict[str, Any]):
        """Display nice results summary in notebook."""
        try:
            from IPython.display import display, HTML
            
            html = """
            <style>
                .results-box {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    background-color: #f9f9f9;
                }
                .metric {
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }
                .metric-label {
                    font-size: 14px;
                    color: #666;
                }
            </style>
            <div class="results-box">
                <h2>🎉 Optimization Complete!</h2>
            """
            
            # Add metrics
            metrics = results.get('metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    html += f"""
                    <div class="metric">
                        <div class="metric-value">{value:.2%}</div>
                        <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                    </div>
                    """
            
            # Add cost if available
            if 'total_cost' in results:
                html += f"""
                <div class="metric">
                    <div class="metric-value">${results['total_cost']:.2f}</div>
                    <div class="metric-label">Total Cost</div>
                </div>
                """
            
            html += "</div>"
            
            display(HTML(html))
        except ImportError:
            # Fallback for non-notebook
            print("\n🎉 Optimization Complete!")
            print("\nResults:")
            for key, value in results.get('metrics', {}).items():
                print(f"  {key}: {value}")
            if 'total_cost' in results:
                print(f"  Total Cost: ${results['total_cost']:.2f}")


class ColabEnvironment:
    """Complete Colab environment setup and management."""
    
    def __init__(self):
        self.drive = DriveIntegration()
        self.secrets = SecretsManager()
        self.sharing = SharingUtilities(self.drive)
        self.notebook = NotebookUtilities()
    
    def setup(self):
        """Complete environment setup."""
        print("🚀 Setting up PromptOpt in Colab...\n")
        
        # Check environment
        if self._is_colab():
            print("✓ Running in Google Colab")
            
            # Mount Drive
            self._mount_drive()
            
            # Set up directories
            self.drive.ensure_directories()
            
            # Check secrets
            if self.secrets.has_llm_keys():
                providers = self.secrets.get_available_providers()
                print(f"✓ LLM providers available: {', '.join(providers)}")
            else:
                print("⚠️  No API keys found. Add them to Colab secrets.")
        else:
            print("✓ Running in local environment")
        
        print("\n✅ Setup complete! Ready to optimize prompts.")
    
    def _is_colab(self) -> bool:
        """Check if running in Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _mount_drive(self):
        """Mount Google Drive."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive mounted")
        except Exception as e:
            print(f"⚠️  Could not mount Drive: {e}")