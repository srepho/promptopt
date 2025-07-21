"""LLM client implementations with cost tracking."""

import time
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

import openai
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.interfaces import LLMProvider


@dataclass
class LLMResponse:
    """Response from an LLM with metadata."""
    text: str
    usage: Dict[str, int]
    cost: float
    latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostTracker:
    """Track costs across different LLM providers."""
    provider: str
    model: str
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    request_count: int = 0
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "openai": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
        "anthropic": {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-2.1": {"input": 0.008, "output": 0.024},
        }
    }
    
    def calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost for a single request."""
        pricing = self.PRICING.get(self.provider, {}).get(self.model, {})
        if not pricing:
            return 0.0
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        prompt_cost = (prompt_tokens / 1000) * pricing.get("input", 0)
        completion_cost = (completion_tokens / 1000) * pricing.get("output", 0)
        
        return prompt_cost + completion_cost
    
    def update(self, usage: Dict[str, int], cost: float):
        """Update tracking statistics."""
        self.total_cost += cost
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.request_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_cost": round(self.total_cost, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "request_count": self.request_count,
            "average_cost_per_request": round(self.total_cost / self.request_count, 4) 
                                       if self.request_count > 0 else 0
        }


class BaseLLMClient(LLMProvider):
    """Base class for LLM clients with common functionality."""
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1000):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_tracker = CostTracker(provider, model)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response with retry logic."""
        return self._generate_impl(prompt, **kwargs)
    
    def _generate_impl(self, prompt: str, **kwargs) -> LLMResponse:
        """Implementation-specific generation method."""
        raise NotImplementedError
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses
    
    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the cost for given token counts."""
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        return self.cost_tracker.calculate_cost(usage)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return self.cost_tracker.get_summary()


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__("openai", model, api_key, **kwargs)
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _generate_impl(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using OpenAI API."""
        start_time = time.time()
        
        # Merge kwargs with defaults
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            params["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            params["presence_penalty"] = kwargs["presence_penalty"]
        
        response = self.client.chat.completions.create(**params)
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        cost = self.cost_tracker.calculate_cost(usage)
        self.cost_tracker.update(usage, cost)
        
        return LLMResponse(
            text=response.choices[0].message.content,
            usage=usage,
            cost=cost,
            latency=time.time() - start_time,
            metadata={
                "model": self.model,
                "provider": self.provider,
                "finish_reason": response.choices[0].finish_reason
            }
        )


class AnthropicClient(BaseLLMClient):
    """Anthropic API client implementation."""
    
    def __init__(self, model: str = "claude-3-haiku", api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__("anthropic", model, api_key, **kwargs)
        self.client = Anthropic(api_key=self.api_key)
    
    def _generate_impl(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using Anthropic API."""
        start_time = time.time()
        
        # Map model names to Anthropic's expected format
        model_map = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-2.1": "claude-2.1"
        }
        
        model_name = model_map.get(self.model, self.model)
        
        response = self.client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        
        # Estimate token usage for Anthropic (they don't provide exact counts)
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        completion_tokens = len(response.content[0].text.split()) * 1.3
        
        usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens)
        }
        
        cost = self.cost_tracker.calculate_cost(usage)
        self.cost_tracker.update(usage, cost)
        
        return LLMResponse(
            text=response.content[0].text,
            usage=usage,
            cost=cost,
            latency=time.time() - start_time,
            metadata={
                "model": model_name,
                "provider": self.provider,
                "stop_reason": response.stop_reason
            }
        )


class LLMJudge:
    """LLM-based judge for comparing responses."""
    
    def __init__(self, client: BaseLLMClient):
        self.client = client
    
    def judge_comparison(self, response_a: str, response_b: str, 
                        criteria: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Compare two responses using LLM judgment."""
        prompt = self._build_comparison_prompt(response_a, response_b, criteria, context)
        
        response = self.client.generate(prompt, temperature=0.1)  # Low temp for consistency
        
        # Parse the judgment
        judgment = self._parse_judgment(response.text)
        
        return {
            "winner": judgment["winner"],
            "confidence": judgment["confidence"],
            "reasoning": judgment["reasoning"],
            "cost": response.cost,
            "raw_response": response.text
        }
    
    def _build_comparison_prompt(self, response_a: str, response_b: str, 
                                criteria: str, context: Optional[str] = None) -> str:
        """Build prompt for comparison."""
        prompt = f"""Compare the following two responses based on the criteria: {criteria}

{f'Context: {context}' if context else ''}

Response A:
{response_a}

Response B:
{response_b}

Please evaluate which response is better. Respond in the following JSON format:
{{
    "winner": "A" or "B" or "tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your judgment"
}}
"""
        return prompt
    
    def _parse_judgment(self, response: str) -> Dict[str, Any]:
        """Parse judgment from LLM response."""
        try:
            # Try to parse as JSON
            import json
            return json.loads(response)
        except:
            # Fallback parsing
            winner = "tie"
            if "Response A" in response and "better" in response:
                winner = "A"
            elif "Response B" in response and "better" in response:
                winner = "B"
            
            return {
                "winner": winner,
                "confidence": 0.5,
                "reasoning": response
            }


def create_llm_client(provider: str, model: str, **kwargs) -> BaseLLMClient:
    """Factory function to create LLM clients."""
    if provider.lower() == "openai":
        return OpenAIClient(model=model, **kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicClient(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")