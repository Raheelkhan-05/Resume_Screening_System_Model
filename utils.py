"""
Utility functions for Resume Screening System
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('resume_screening.log')
        ]
    )
    return logging.getLogger(__name__)

def save_results_to_json(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Save screening results to JSON file"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"screening_results_{timestamp}.json"
    
    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'system': 'Resume Screening System'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_path

def validate_file_exists(file_path: str) -> bool:
    """Check if file exists and is readable"""
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

def validate_file_type(file_path: str, allowed_types: list = None) -> bool:
    """Validate file type based on extension"""
    if allowed_types is None:
        allowed_types = ['.pdf', '.docx', '.txt']
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in allowed_types

def sanitize_text(text: str) -> str:
    """Basic text sanitization"""
    if not text:
        return ""
    
    # Remove null bytes and normalize whitespace
    text = text.replace('\x00', '').strip()
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove excessive whitespace
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    return text

def create_output_directory(base_dir: str = "output") -> str:
    """Create output directory if it doesn't exist"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

class ProgressTracker:
    """Simple progress tracking for CLI feedback"""
    
    def __init__(self, total_steps: int = 6):
        self.total_steps = total_steps
        self.current_step = 0
        self.steps = [
            "Extracting text from files",
            "Parsing resume with LLM",
            "Parsing job description with LLM", 
            "Preprocessing text",
            "Computing semantic similarity",
            "Generating recommendations"
        ]
    
    def next_step(self) -> str:
        """Move to next step and return description"""
        if self.current_step < self.total_steps:
            step_desc = self.steps[self.current_step]
            self.current_step += 1
            return f"[{self.current_step}/{self.total_steps}] {step_desc}..."
        return "Complete!"
    
    def get_progress(self) -> float:
        """Get progress percentage"""
        return (self.current_step / self.total_steps) * 100