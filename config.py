"""
Configuration settings for Resume Screening System
"""

import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class ScoringWeights:
    """Default scoring weights for ATS calculation"""
    skills_match: float = 0.50      # 50%
    experience_relevance: float = 0.30  # 30%
    education_match: float = 0.10   # 10%
    extras: float = 0.10           # 10% (Certifications/Projects)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'skills_match': self.skills_match,
            'experience_relevance': self.experience_relevance,
            'education_match': self.education_match,
            'extras': self.extras
        }
    
    def validate(self) -> bool:
        """Ensure weights sum to 1.0"""
        total = sum(self.to_dict().values())
        return abs(total - 1.0) < 0.001

# Azure OpenAI Configuration
AZURE_OPENAI_CONFIG = {
    'api_key': os.getenv('AZURE_OPENAI_API_KEY', ''),
    'api_base': os.getenv('AZURE_OPENAI_API_BASE', ''),
    'api_version': os.getenv('AZURE_OPENAI_API_VERSION', '2023-12-01-preview'),
    'deployment_name': os.getenv('AZURE_OPENAI_API_NAME', 'gpt-4o'),
    'model_name': 'gpt-4o',
    'temperature': 0.0,  # For deterministic output
    'max_tokens': 2000
}

# Sentence-BERT Model Configuration
SBERT_MODEL = 'all-MiniLM-L6-v2'

# Text preprocessing settings
PREPROCESSING_CONFIG = {
    'remove_stopwords': True,
    'lemmatize': True,
    'lowercase': True,
    'normalize_whitespace': True
}

# Default scoring weights
DEFAULT_WEIGHTS = ScoringWeights()

# Output settings
OUTPUT_CONFIG = {
    'json_indent': 2,
    'include_raw_scores': True,
    'include_debug_info': False
}