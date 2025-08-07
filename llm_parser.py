"""
LLM-based parsing module using Azure OpenAI GPT-4o
Handles structured parsing of resumes and job descriptions
"""

import logging
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from config import AZURE_OPENAI_CONFIG

logger = logging.getLogger(__name__)

class LLMParser:
    """Handles structured parsing using Azure OpenAI GPT-4o"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.resume_parsing_prompt = self._get_resume_parsing_prompt()
        self.jd_parsing_prompt = self._get_jd_parsing_prompt()
    
    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize Azure OpenAI client"""
        try:
            llm = AzureChatOpenAI(
                deployment_name=AZURE_OPENAI_CONFIG['deployment_name'],
                api_version=AZURE_OPENAI_CONFIG['api_version'],
                api_key=AZURE_OPENAI_CONFIG['api_key'],
                azure_endpoint=AZURE_OPENAI_CONFIG['api_base'],
                model_name=AZURE_OPENAI_CONFIG['model_name'],
                temperature=AZURE_OPENAI_CONFIG['temperature'],  # 0.0 for deterministic output
                max_tokens=AZURE_OPENAI_CONFIG['max_tokens']
            )
            
            logger.info("Azure OpenAI client initialized successfully")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def parse_with_llm(self, text: str, document_type: str) -> Dict[str, Any]:
        """
        Parse document text using LLM
        
        Args:
            text: Document text to parse
            document_type: 'resume' or 'job_description'
            
        Returns:
            Structured parsing results
        """
        try:
            if document_type == 'resume':
                return self._parse_resume(text)
            elif document_type == 'job_description':
                return self._parse_job_description(text)
            else:
                raise ValueError(f"Unsupported document type: {document_type}")
                
        except Exception as e:
            logger.error(f"Error parsing {document_type}: {e}")
            return {
                'parsing_successful': False,
                'error_message': str(e),
                'parsed_data': {}
            }
    
    def _parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Parse resume into structured sections"""
        try:
            prompt = self.resume_parsing_prompt.format(resume_text=resume_text)
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse JSON response
            parsed_data = json.loads(response.content)
            
            return {
                'parsing_successful': True,
                'error_message': None,
                'parsed_data': parsed_data,
                'document_type': 'resume'
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in resume parsing: {e}")
            return {
                'parsing_successful': False,
                'error_message': f"JSON decode error: {str(e)}",
                'parsed_data': {}
            }
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            return {
                'parsing_successful': False,
                'error_message': str(e),
                'parsed_data': {}
            }
    
    def _parse_job_description(self, jd_text: str) -> Dict[str, Any]:
        """Parse job description into structured sections"""
        try:
            prompt = self.jd_parsing_prompt.format(jd_text=jd_text)
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse JSON response
            parsed_data = json.loads(response.content)
            
            return {
                'parsing_successful': True,
                'error_message': None,
                'parsed_data': parsed_data,
                'document_type': 'job_description'
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in JD parsing: {e}")
            return {
                'parsing_successful': False,
                'error_message': f"JSON decode error: {str(e)}",
                'parsed_data': {}
            }
        except Exception as e:
            logger.error(f"Error parsing job description: {e}")
            return {
                'parsing_successful': False,
                'error_message': str(e),
                'parsed_data': {}
            }
    
    def _get_resume_parsing_prompt(self) -> str:
        """Get prompt template for resume parsing"""
        return """
You are an expert resume parser. Analyze the following resume text and extract structured information.

RESUME TEXT:
{resume_text}

INSTRUCTIONS:
1. Extract information into the JSON structure below
2. If a section is not found, use empty string or empty array
3. Be thorough but accurate - only include information actually present
4. For skills, include both technical and soft skills
5. For experience, extract key achievements and responsibilities
6. Ensure the JSON is valid and properly formatted

REQUIRED JSON OUTPUT FORMAT:
{{
    "personal_info": {{
        "name": "Full name of candidate",
        "email": "Email address",
        "phone": "Phone number",
        "location": "City, State/Country"
    }},
    "summary": "Professional summary or objective statement",
    "skills": {{
        "technical_skills": ["list", "of", "technical", "skills"],
        "soft_skills": ["list", "of", "soft", "skills"],
        "tools_and_technologies": ["tools", "technologies", "frameworks"],
        "programming_languages": ["languages", "if", "applicable"]
    }},
    "experience": [
        {{
            "job_title": "Position title",
            "company": "Company name",
            "duration": "Start - End dates",
            "responsibilities": ["key", "responsibilities", "and", "achievements"],
            "technologies_used": ["relevant", "technologies"]
        }}
    ],
    "education": [
        {{
            "degree": "Degree title",
            "institution": "School/University name",
            "year": "Graduation year",
            "gpa": "GPA if mentioned",
            "relevant_coursework": ["relevant", "courses"]
        }}
    ],
    "projects": [
        {{
            "project_name": "Project title",
            "description": "Brief description",
            "technologies": ["technologies", "used"],
            "duration": "Project timeline"
        }}
    ],
    "certifications": [
        {{
            "certification_name": "Certification title",
            "issuing_organization": "Issuer",
            "date_obtained": "Date",
            "expiry_date": "Expiry if applicable"
        }}
    ]
}}

Return ONLY the JSON response with no additional text or formatting.
"""
    
    def _get_jd_parsing_prompt(self) -> str:
        """Get prompt template for job description parsing"""
        return """
You are an expert job description parser. Analyze the following job description text and extract structured information.

JOB DESCRIPTION TEXT:
{jd_text}

INSTRUCTIONS:
1. Extract information into the JSON structure below
2. If a section is not found, use empty string or empty array
3. Be thorough but accurate - only include information actually present
4. Distinguish between required and preferred qualifications
5. Extract both hard and soft skill requirements
6. Ensure the JSON is valid and properly formatted

REQUIRED JSON OUTPUT FORMAT:
{{
    "job_info": {{
        "job_title": "Position title",
        "company": "Company name",
        "location": "Job location",
        "employment_type": "Full-time/Part-time/Contract/Remote",
        "department": "Department if mentioned"
    }},
    "job_summary": "Brief job description or summary",
    "responsibilities": ["key", "job", "responsibilities", "and", "duties"],
    "required_qualifications": {{
        "experience_level": "Years of experience required",
        "education": ["required", "education", "qualifications"],
        "technical_skills": ["required", "technical", "skills"],
        "soft_skills": ["required", "soft", "skills"],
        "certifications": ["required", "certifications"],
        "tools_and_technologies": ["required", "tools", "platforms"]
    }},
    "preferred_qualifications": {{
        "experience_level": "Preferred years of experience",
        "education": ["preferred", "education"],
        "technical_skills": ["preferred", "technical", "skills"],
        "soft_skills": ["preferred", "soft", "skills"],
        "certifications": ["preferred", "certifications"],
        "tools_and_technologies": ["preferred", "tools"]
    }},
    "benefits": ["company", "benefits", "if", "mentioned"],
    "company_info": {{
        "company_description": "Brief company description",
        "company_size": "Company size if mentioned",
        "industry": "Industry sector"
    }}
}}

Return ONLY the JSON response with no additional text or formatting.
"""