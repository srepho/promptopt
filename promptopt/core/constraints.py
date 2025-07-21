"""Pre-built constraint definitions for common use cases."""

import re
from typing import Optional

from .base import Constraint


def create_conciseness_constraint(max_words: int = 50, weight: float = 0.3) -> Constraint:
    """Create a constraint for response conciseness."""
    return Constraint(
        name="conciseness",
        description=f"Response must be under {max_words} words",
        validator=lambda response: len(response.split()) <= max_words,
        weight=weight
    )


def create_format_constraint(required_format: str, weight: float = 0.5) -> Constraint:
    """Create a constraint for response format.
    
    Args:
        required_format: Format pattern with placeholders, e.g. "Answer: [YES/NO]"
        weight: Importance weight for this constraint
    """
    # Convert format placeholders to regex
    pattern = required_format
    pattern = pattern.replace("[YES/NO]", "(YES|NO)")
    pattern = pattern.replace("[TRUE/FALSE]", "(TRUE|FALSE)")
    pattern = pattern.replace("[NUMBER]", r"\d+")
    pattern = pattern.replace("[TEXT]", r".+")
    
    return Constraint(
        name="format_adherence",
        description=f"Response must follow format: {required_format}",
        validator=lambda response: bool(re.search(pattern, response, re.IGNORECASE)),
        weight=weight
    )


def create_tone_constraint(tone: str, keywords: Optional[list] = None, 
                         weight: float = 0.3) -> Constraint:
    """Create a constraint for response tone."""
    tone_keywords = {
        "professional": ["please", "thank you", "would", "could", "appreciate"],
        "friendly": ["happy", "glad", "great", "wonderful", "excited"],
        "formal": ["therefore", "furthermore", "however", "consequently"],
        "casual": ["hey", "thanks", "cool", "awesome", "sure"]
    }
    
    keywords_to_check = keywords or tone_keywords.get(tone.lower(), [])
    
    def validator(response: str) -> bool:
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in keywords_to_check)
    
    return Constraint(
        name=f"{tone}_tone",
        description=f"Response must maintain {tone} tone",
        validator=validator,
        weight=weight
    )


def create_no_pii_constraint(weight: float = 1.0) -> Constraint:
    """Create a constraint to prevent PII in responses."""
    # Simple PII patterns - in production would use more sophisticated detection
    pii_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone
        r'\b\d{16}\b',  # Credit card
    ]
    
    def validator(response: str) -> bool:
        for pattern in pii_patterns:
            if re.search(pattern, response):
                return False
        return True
    
    return Constraint(
        name="no_pii",
        description="Response must not contain PII",
        validator=validator,
        weight=weight
    )


def create_json_format_constraint(weight: float = 0.8) -> Constraint:
    """Create a constraint for valid JSON responses."""
    import json
    
    def validator(response: str) -> bool:
        try:
            json.loads(response)
            return True
        except:
            return False
    
    return Constraint(
        name="valid_json",
        description="Response must be valid JSON",
        validator=validator,
        weight=weight
    )


def create_bullet_points_constraint(min_points: int = 3, max_points: int = 10,
                                   weight: float = 0.5) -> Constraint:
    """Create a constraint for bullet point format."""
    def validator(response: str) -> bool:
        # Count lines starting with bullet indicators
        bullet_patterns = [r'^\s*[-*•]\s', r'^\s*\d+\.\s']
        lines = response.split('\n')
        bullet_count = 0
        
        for line in lines:
            if any(re.match(pattern, line) for pattern in bullet_patterns):
                bullet_count += 1
        
        return min_points <= bullet_count <= max_points
    
    return Constraint(
        name="bullet_points",
        description=f"Response must have {min_points}-{max_points} bullet points",
        validator=validator,
        weight=weight
    )


def create_language_constraint(language: str = "english", weight: float = 0.7) -> Constraint:
    """Create a constraint for response language."""
    # Simple language detection based on common words
    language_indicators = {
        "english": ["the", "is", "and", "to", "of", "in"],
        "spanish": ["el", "la", "de", "que", "es", "en"],
        "french": ["le", "de", "et", "la", "les", "des"],
        "german": ["der", "die", "das", "und", "ist", "ein"],
    }
    
    indicators = language_indicators.get(language.lower(), language_indicators["english"])
    
    def validator(response: str) -> bool:
        words = response.lower().split()
        if len(words) < 10:  # Too short to determine
            return True
        
        indicator_count = sum(1 for word in words if word in indicators)
        return indicator_count / len(words) > 0.05  # At least 5% indicator words
    
    return Constraint(
        name=f"{language}_language",
        description=f"Response must be in {language}",
        validator=validator,
        weight=weight
    )


def create_business_constraint(constraint_type: str, **kwargs) -> Constraint:
    """Create business-specific constraints."""
    if constraint_type == "customer_service":
        return Constraint(
            name="customer_service_quality",
            description="Response must be helpful and empathetic",
            validator=lambda r: any(word in r.lower() for word in 
                                  ["help", "assist", "understand", "apologize", "sorry"]),
            weight=kwargs.get("weight", 0.6)
        )
    
    elif constraint_type == "compliance":
        return Constraint(
            name="compliance_adherence",
            description="Response must include disclaimer",
            validator=lambda r: "not financial advice" in r.lower() or 
                              "consult professional" in r.lower(),
            weight=kwargs.get("weight", 0.9)
        )
    
    elif constraint_type == "brand_voice":
        brand = kwargs.get("brand", "professional")
        return create_tone_constraint(brand, weight=kwargs.get("weight", 0.5))
    
    else:
        raise ValueError(f"Unknown business constraint type: {constraint_type}")