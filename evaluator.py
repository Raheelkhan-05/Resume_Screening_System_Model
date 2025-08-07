"""
LLM-based evaluation and recommendation module
Generates candidate summaries, detailed feedback, and improvement suggestions
"""

import logging
import json
from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

from config import AZURE_OPENAI_CONFIG

logger = logging.getLogger(__name__)

class CandidateEvaluator:
    """Handles LLM-based evaluation and recommendations"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
    
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
            
            logger.info("Azure OpenAI client for evaluation initialized successfully")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client for evaluation: {e}")
            raise
    
    def evaluate_with_llm(self, resume_data: Dict[str, Any], jd_data: Dict[str, Any], 
                         scoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation using LLM
        
        Args:
            resume_data: Parsed resume data
            jd_data: Parsed job description data
            scoring_results: ATS scoring results
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            # Generate candidate summary
            candidate_summary = self._generate_candidate_summary(resume_data, jd_data, scoring_results)
            
            # Generate section-wise evaluation
            section_evaluation = self._generate_section_evaluation(resume_data, jd_data, scoring_results)
            
            # Generate improvement recommendations
            recommendations = self._generate_recommendations(resume_data, jd_data, scoring_results)
            
            # Predict improved score
            predicted_score = self._predict_improved_score(scoring_results, recommendations)
            
            return {
                'candidate_summary': candidate_summary,
                'section_evaluation': section_evaluation,
                'recommendations': recommendations,
                'predicted_improved_score': predicted_score,
                'evaluation_successful': True,
                'error_message': None
            }
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return {
                'candidate_summary': '',
                'section_evaluation': {},
                'recommendations': [],
                'predicted_improved_score': scoring_results.get('ats_score', 0),
                'evaluation_successful': False,
                'error_message': str(e)
            }
    
    def _generate_candidate_summary(self, resume_data: Dict[str, Any], 
                                  jd_data: Dict[str, Any], 
                                  scoring_results: Dict[str, Any]) -> str:
        """Generate candidate summary"""
        prompt = self._get_summary_prompt()
        
        # Prepare context data
        context = {
            'resume_data': resume_data,
            'job_requirements': jd_data,
            'ats_score': scoring_results.get('ats_score', 0),
            'section_scores': scoring_results.get('section_scores', {})
        }
        
        try:
            full_prompt = prompt.format(
                candidate_name=resume_data.get('personal_info', {}).get('name', 'Candidate'),
                job_title=jd_data.get('job_info', {}).get('job_title', 'Position'),
                ats_score=scoring_results.get('ats_score', 0),
                context_json=json.dumps(context, indent=2)
            )
            
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating candidate summary: {e}")
            return "Unable to generate candidate summary due to processing error."
    
    def _generate_section_evaluation(self, resume_data: Dict[str, Any], 
                                   jd_data: Dict[str, Any], 
                                   scoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed section-wise evaluation"""
        prompt = self._get_section_evaluation_prompt()
        
        context = {
            'resume_data': resume_data,
            'job_requirements': jd_data,
            'section_scores': scoring_results.get('section_scores', {}),
            'detailed_analysis': scoring_results.get('detailed_analysis', {})
        }
        
        try:
            full_prompt = prompt.format(context_json=json.dumps(context, indent=2))
            
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            # Parse JSON response
            evaluation_data = json.loads(response.content)
            return evaluation_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in section evaluation: {e}")
            return self._get_default_section_evaluation()
        except Exception as e:
            logger.error(f"Error generating section evaluation: {e}")
            return self._get_default_section_evaluation()
    
    def _generate_recommendations(self, resume_data: Dict[str, Any], 
                                jd_data: Dict[str, Any], 
                                scoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable improvement recommendations"""
        prompt = self._get_recommendations_prompt()
        
        context = {
            'resume_data': resume_data,
            'job_requirements': jd_data,
            'ats_score': scoring_results.get('ats_score', 0),
            'section_scores': scoring_results.get('section_scores', {}),
            'detailed_analysis': scoring_results.get('detailed_analysis', {})
        }
        
        try:
            full_prompt = prompt.format(
                ats_score=scoring_results.get('ats_score', 0),
                context_json=json.dumps(context, indent=2)
            )
            
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            # Parse JSON response
            recommendations_data = json.loads(response.content)
            return recommendations_data.get('recommendations', [])
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in recommendations: {e}")
            return self._get_default_recommendations(scoring_results)
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_default_recommendations(scoring_results)
    
    def _predict_improved_score(self, scoring_results: Dict[str, Any], 
                               recommendations: List[Dict[str, Any]]) -> float:
        """Predict improved ATS score after implementing recommendations"""
        current_score = scoring_results.get('ats_score', 0)
        
        # Simple heuristic-based prediction
        # In a more sophisticated version, this could use ML models
        try:
            total_impact = 0
            for rec in recommendations:
                impact = rec.get('expected_impact', 0)
                total_impact += impact
            
            # Cap the improvement to be realistic
            max_improvement = min(total_impact, 30)  # Max 30 point improvement
            predicted_score = min(current_score + max_improvement, 100)
            
            return round(predicted_score, 2)
            
        except Exception as e:
            logger.error(f"Error predicting improved score: {e}")
            return current_score
    
    def _get_summary_prompt(self) -> str:
        """Get prompt template for candidate summary"""
        return """
You are an expert HR analyst. Generate a concise, professional summary of the candidate's profile and fit for the position.

CANDIDATE: {candidate_name}
POSITION: {job_title}
CURRENT ATS SCORE: {ats_score}/100

ANALYSIS DATA:
{context_json}

INSTRUCTIONS:
1. Write a 3-4 sentence professional summary
2. Highlight the candidate's key strengths relevant to the position
3. Mention any notable gaps or areas for improvement
4. Maintain a balanced, objective tone
5. Focus on job-relevant qualifications and experience

Write the summary in paragraph form without any JSON formatting or additional structure.
"""
    
    def _get_section_evaluation_prompt(self) -> str:
        """Get prompt template for section-wise evaluation"""
        return """
You are an expert ATS analyst. Provide detailed section-wise evaluation of the candidate's alignment with job requirements.

ANALYSIS DATA:
{context_json}

INSTRUCTIONS:
1. Evaluate each section: Skills, Experience, Education, Extras
2. For each section, provide specific feedback on alignment
3. Identify specific missing elements
4. Assess the quality and relevance of presented information
5. Return results in the exact JSON format below

REQUIRED JSON OUTPUT FORMAT:
{{
    "skills_evaluation": {{
        "alignment_score": "numerical score from section_scores",
        "strengths": ["specific", "technical", "skills", "that", "match"],
        "gaps": ["missing", "required", "skills"],
        "assessment": "2-3 sentence evaluation of skills section"
    }},
    "experience_evaluation": {{
        "alignment_score": "numerical score from section_scores", 
        "relevant_experience": ["matching", "experience", "areas"],
        "experience_gaps": ["missing", "experience", "types"],
        "assessment": "2-3 sentence evaluation of experience section"
    }},
    "education_evaluation": {{
        "alignment_score": "numerical score from section_scores",
        "matching_qualifications": ["relevant", "education", "aspects"],
        "education_gaps": ["missing", "educational", "requirements"],
        "assessment": "2-3 sentence evaluation of education section"
    }},
    "extras_evaluation": {{
        "alignment_score": "numerical score from section_scores",
        "valuable_additions": ["relevant", "certifications", "projects"],
        "recommended_additions": ["suggested", "certifications", "projects"],
        "assessment": "2-3 sentence evaluation of additional qualifications"
    }}
}}

Return ONLY the JSON response with no additional text or formatting.
"""
    
    def _get_recommendations_prompt(self) -> str:
        """Get prompt template for improvement recommendations"""
        return """
You are an expert career coach and ATS optimization specialist. Generate specific, actionable recommendations to improve the candidate's ATS score.

CURRENT ATS SCORE: {ats_score}/100

ANALYSIS DATA:
{context_json}

INSTRUCTIONS:
1. Focus on the lowest-scoring sections first
2. Provide specific, actionable recommendations
3. Estimate the potential impact of each recommendation
4. Prioritize recommendations by impact and feasibility
5. Include both content and formatting suggestions
6. Return results in the exact JSON format below

REQUIRED JSON OUTPUT FORMAT:
{{
    "recommendations": [
        {{
            "category": "Skills/Experience/Education/Formatting",
            "priority": "High/Medium/Low",
            "title": "Brief recommendation title",
            "description": "Detailed explanation of what to do",
            "specific_actions": ["action 1", "action 2", "action 3"],
            "expected_impact": 5,
            "difficulty": "Easy/Medium/Hard",
            "rationale": "Why this recommendation will help"
        }}
    ]
}}

GUIDELINES FOR EXPECTED IMPACT:
- High priority, easy difficulty: 8-15 points
- Medium priority: 3-8 points  
- Low priority: 1-5 points
- Ensure total expected impact is realistic (max 25-30 points improvement)

Return ONLY the JSON response with no additional text or formatting.
"""
    
    def _get_default_section_evaluation(self) -> Dict[str, Any]:
        """Return default section evaluation if LLM parsing fails"""
        return {
            "skills_evaluation": {
                "alignment_score": 0.0,
                "strengths": [],
                "gaps": ["Unable to analyze due to processing error"],
                "assessment": "Section evaluation unavailable due to processing error."
            },
            "experience_evaluation": {
                "alignment_score": 0.0,
                "relevant_experience": [],
                "experience_gaps": ["Unable to analyze due to processing error"],
                "assessment": "Experience evaluation unavailable due to processing error."
            },
            "education_evaluation": {
                "alignment_score": 0.0,
                "matching_qualifications": [],
                "education_gaps": ["Unable to analyze due to processing error"],
                "assessment": "Education evaluation unavailable due to processing error."
            },
            "extras_evaluation": {
                "alignment_score": 0.0,
                "valuable_additions": [],
                "recommended_additions": ["Unable to analyze due to processing error"],
                "assessment": "Additional qualifications evaluation unavailable due to processing error."
            }
        }
    
    def _get_default_recommendations(self, scoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return default recommendations if LLM parsing fails"""
        current_score = scoring_results.get('ats_score', 0)
        
        return [
            {
                "category": "General",
                "priority": "High",
                "title": "Review and optimize resume content",
                "description": "Unable to generate specific recommendations due to processing error. Please review resume manually.",
                "specific_actions": [
                    "Ensure all relevant skills are clearly listed",
                    "Quantify achievements with specific metrics",
                    "Align experience descriptions with job requirements"
                ],
                "expected_impact": max(5, min(15, 60 - current_score)),
                "difficulty": "Medium",
                "rationale": "General resume optimization typically improves ATS scores"
            }
        ]