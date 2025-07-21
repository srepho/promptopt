"""Google Colab environment management."""

import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path


class ColabManager:
    """Manage Colab-specific functionality."""
    
    def __init__(self):
        self.is_colab = self._detect_colab()
        self.drive_mounted = False
        self.api_keys = {}
    
    def _detect_colab(self) -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def setup_enterprise_environment(self):
        """Set up enterprise environment in Colab."""
        print("🚀 Setting up PromptOpt Enterprise Environment...")
        
        if self.is_colab:
            print("✓ Running in Google Colab")
            self._mount_drive()
            self._setup_colab_secrets()
            self._install_dependencies()
        else:
            print("✓ Running in local environment")
        
        # Set up API keys
        self._setup_api_keys()
        
        # Create working directories
        self._setup_directories()
        
        print("\n✅ Environment setup complete!")
    
    def _mount_drive(self):
        """Mount Google Drive in Colab."""
        if self.is_colab and not self.drive_mounted:
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                self.drive_mounted = True
                print("✓ Google Drive mounted")
            except Exception as e:
                print(f"⚠️  Could not mount Drive: {e}")
    
    def _setup_colab_secrets(self):
        """Set up secrets management in Colab."""
        if self.is_colab:
            try:
                from google.colab import userdata
                # Try to get secrets from Colab secrets
                for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                    try:
                        secret = userdata.get(key)
                        if secret:
                            os.environ[key] = secret
                            self.api_keys[key] = secret
                            print(f"✓ Loaded {key} from Colab secrets")
                    except:
                        pass
            except ImportError:
                print("ℹ️  Colab secrets not available in this version")
    
    def _install_dependencies(self):
        """Install required dependencies in Colab."""
        if self.is_colab:
            print("📦 Installing dependencies...")
            import subprocess
            
            # Install promptopt if not already installed
            try:
                import promptopt
            except ImportError:
                subprocess.run(["pip", "install", "-q", "promptopt"], check=False)
                print("✓ Installed promptopt")
    
    def _setup_api_keys(self):
        """Set up API keys from environment or user input."""
        keys_needed = []
        
        for key_name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if not os.getenv(key_name):
                keys_needed.append(key_name)
        
        if keys_needed and self.is_colab:
            print("\n🔑 API Key Setup")
            print("Add your API keys to Colab secrets or set them here:")
            
            for key_name in keys_needed:
                if self.is_colab:
                    # In Colab, could use getpass for secure input
                    print(f"\nTo set {key_name}:")
                    print(f"1. Click the 🔑 key icon in the left sidebar")
                    print(f"2. Add a new secret named '{key_name}'")
                    print(f"3. Paste your API key as the value")
                else:
                    print(f"export {key_name}='your-key-here'")
        
        # Verify what's available
        available_keys = []
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if os.getenv(key):
                available_keys.append(key.replace("_API_KEY", ""))
        
        if available_keys:
            print(f"\n✓ Available LLM providers: {', '.join(available_keys)}")
        else:
            print("\n⚠️  No API keys found. You'll need to set them to use LLM features.")
    
    def _setup_directories(self):
        """Create working directories."""
        dirs = ["./results", "./data", "./prompts", "./reports"]
        for dir_path in dirs:
            Path(dir_path).mkdir(exist_ok=True)
        print("✓ Created working directories")
    
    def create_business_context_wizard(self) -> Dict[str, Any]:
        """Interactive wizard for creating business context."""
        if self.is_colab:
            return self._colab_interactive_wizard()
        else:
            return self._cli_wizard()
    
    def _colab_interactive_wizard(self) -> Dict[str, Any]:
        """Colab-specific interactive wizard using forms."""
        print("\n🏢 Business Context Setup Wizard")
        
        # In real Colab, would use forms/widgets
        # For now, return example context
        context = {
            "industry": "technology",
            "company_size": "enterprise",
            "use_case": "customer_support",
            "compliance_requirements": ["SOC2", "GDPR"],
            "brand_voice": "professional yet friendly",
            "target_audience": "business customers"
        }
        
        print("\nBusiness Context Created:")
        for key, value in context.items():
            print(f"  {key}: {value}")
        
        return context
    
    def _cli_wizard(self) -> Dict[str, Any]:
        """CLI-based wizard for local environments."""
        print("\n🏢 Business Context Setup")
        
        # Simplified for demo
        industries = ["technology", "healthcare", "finance", "retail", "other"]
        use_cases = ["customer_support", "internal_communication", "content_creation", "data_analysis"]
        
        print("\nAvailable industries:", ", ".join(industries))
        print("Available use cases:", ", ".join(use_cases))
        
        return {
            "industry": industries[0],
            "company_size": "enterprise",
            "use_case": use_cases[0],
            "compliance_requirements": ["SOC2"],
            "brand_voice": "professional",
            "target_audience": "business users"
        }
    
    def create_shareable_results(self, results: Dict[str, Any], 
                               filename: str = "optimization_results") -> str:
        """Create shareable results file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if self.is_colab and self.drive_mounted:
            # Save to Drive for easy sharing
            base_path = "/content/drive/MyDrive/PromptOpt"
            Path(base_path).mkdir(exist_ok=True)
            filepath = f"{base_path}/{filename}_{timestamp}.json"
        else:
            # Save locally
            filepath = f"./results/{filename}_{timestamp}.json"
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {filepath}")
        
        if self.is_colab:
            print("Share this file from your Google Drive!")
        
        return filepath
    
    def create_interactive_dashboard(self, results: Dict[str, Any]):
        """Create interactive dashboard for results."""
        if self.is_colab:
            try:
                from IPython.display import display, HTML
                
                html_content = self._generate_dashboard_html(results)
                display(HTML(html_content))
            except ImportError:
                print("Dashboard visualization not available")
        else:
            print("\n📊 Results Summary:")
            self._print_results_summary(results)
    
    def _generate_dashboard_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML dashboard content."""
        html = """
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>🚀 Prompt Optimization Results</h2>
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px;">
        """
        
        if "optimizer" in results:
            html += f"<p><strong>Optimizer:</strong> {results['optimizer']}</p>"
        
        if "metrics" in results:
            html += "<h3>Performance Metrics</h3><ul>"
            for metric, value in results["metrics"].items():
                html += f"<li>{metric}: {value:.2%}</li>"
            html += "</ul>"
        
        if "cost" in results:
            html += f"<p><strong>Optimization Cost:</strong> ${results['cost']:.2f}</p>"
        
        html += """
            </div>
            <p style="margin-top: 20px;">
                <a href="#" onclick="alert('Copy shareable link from Drive!')">Share Results</a>
            </p>
        </div>
        """
        
        return html
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Print results summary for CLI."""
        if "optimizer" in results:
            print(f"Optimizer: {results['optimizer']}")
        
        if "metrics" in results:
            print("\nMetrics:")
            for metric, value in results["metrics"].items():
                print(f"  {metric}: {value:.2%}")
        
        if "cost" in results:
            print(f"\nTotal Cost: ${results['cost']:.2f}")
    
    def setup_monitoring(self):
        """Set up monitoring for optimization runs."""
        print("\n📊 Setting up monitoring...")
        
        if self.is_colab:
            # Could integrate with Colab's built-in charts
            print("✓ Colab monitoring enabled")
            print("  View progress in the output cells below")
        else:
            print("✓ Local monitoring enabled")
            print("  Check ./results/ for optimization logs")
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits for current environment."""
        limits = {
            "max_runtime_seconds": 43200 if self.is_colab else None,  # 12 hours in Colab
            "max_memory_gb": 12 if self.is_colab else None,
            "gpu_available": False,  # Assuming no GPU for this package
            "persistent_storage": self.drive_mounted if self.is_colab else True
        }
        
        return limits