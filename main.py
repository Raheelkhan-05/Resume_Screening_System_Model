#!/usr/bin/env python3
"""
Resume Screening System - Main Entry Point
Uses Azure OpenAI GPT-4o and Sentence-BERT for ATS scoring
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from cli import main

if __name__ == "__main__":
    main()