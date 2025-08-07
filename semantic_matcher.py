"""
Semantic matching and scoring module using Sentence-BERT
Computes ATS scores based on resume-job description similarity
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

from config import SBERT_MODEL, DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)

class SemanticMatcher:
    """Handles semantic matching using Sentence-BERT embeddings"""
    
    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name
        self.model = self._load_model()
        self.scoring_weights = DEFAULT_WEIGHTS
    
    def _load_model(self) -> SentenceTransformer:
        """Load Sentence-BERT model"""
        try:
            model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded Sentence-BERT model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading Sentence-BERT model: {e}")
            raise
    
    def embed_and_score_with_sbert(self, resume_data: Dict[str, Any], 
                                  jd_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute semantic similarity and ATS score
        
        Args:
            resume_data: Parsed resume data
            jd_data: Parsed job description data
            
        Returns:
            Comprehensive scoring results
        """
        try:
            # Extract and prepare text sections
            resume_sections = self._extract_resume_sections(resume_data)
            jd_sections = self._extract_jd_sections(jd_data)
            
            # Compute section-wise similarities
            section_scores = self._compute_section_similarities(resume_sections, jd_sections)
            
            # Calculate weighted ATS score
            final_score = self._calculate_weighted_score(section_scores)
            
            # Generate detailed analysis
            analysis = self._generate_score_analysis(section_scores, resume_sections, jd_sections)
            
            return {
                'ats_score': final_score,
                'section_scores': section_scores,
                'detailed_analysis': analysis,
                'scoring_weights': self.scoring_weights.to_dict(),
                'embedding_model': self.model_name,
                'computation_successful': True,
                'error_message': None
            }
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return {
                'ats_score': 0.0,
                'section_scores': {},
                'detailed_analysis': {},
                'scoring_weights': self.scoring_weights.to_dict(),
                'embedding_model': self.model_name,
                'computation_successful': False,
                'error_message': str(e)
            }
    
    def _extract_resume_sections(self, resume_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract text sections from parsed resume"""
        sections = {}
        
        try:
            # Skills section
            skills = resume_data.get('skills', {})
            all_skills = []
            all_skills.extend(skills.get('technical_skills', []))
            all_skills.extend(skills.get('soft_skills', []))
            all_skills.extend(skills.get('tools_and_technologies', []))
            all_skills.extend(skills.get('programming_languages', []))
            sections['skills'] = ' '.join(all_skills)
            
            # Experience section
            experience = resume_data.get('experience', [])
            exp_text = []
            for exp in experience:
                exp_text.append(exp.get('job_title', ''))
                exp_text.extend(exp.get('responsibilities', []))
                exp_text.extend(exp.get('technologies_used', []))
            sections['experience'] = ' '.join(exp_text)
            
            # Education section
            education = resume_data.get('education', [])
            edu_text = []
            for edu in education:
                edu_text.append(edu.get('degree', ''))
                edu_text.append(edu.get('institution', ''))
                edu_text.extend(edu.get('relevant_coursework', []))
            sections['education'] = ' '.join(edu_text)
            
            # Projects and certifications (extras)
            projects = resume_data.get('projects', [])
            proj_text = []
            for proj in projects:
                proj_text.append(proj.get('project_name', ''))
                proj_text.append(proj.get('description', ''))
                proj_text.extend(proj.get('technologies', []))
            
            certs = resume_data.get('certifications', [])
            cert_text = []
            for cert in certs:
                cert_text.append(cert.get('certification_name', ''))
                cert_text.append(cert.get('issuing_organization', ''))
            
            sections['extras'] = ' '.join(proj_text + cert_text)
            
            # Overall resume text
            all_text = []
            all_text.append(resume_data.get('summary', ''))
            all_text.extend([sections['skills'], sections['experience'], 
                           sections['education'], sections['extras']])
            sections['overall'] = ' '.join(all_text)
            
        except Exception as e:
            logger.error(f"Error extracting resume sections: {e}")
            # Return empty sections if extraction fails
            sections = {
                'skills': '',
                'experience': '',
                'education': '',
                'extras': '',
                'overall': ''
            }
        
        return sections
    
    def _extract_jd_sections(self, jd_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract text sections from parsed job description"""
        sections = {}
        
        try:
            # Required skills
            req_quals = jd_data.get('required_qualifications', {})
            req_skills = []
            req_skills.extend(req_quals.get('technical_skills', []))
            req_skills.extend(req_quals.get('soft_skills', []))
            req_skills.extend(req_quals.get('tools_and_technologies', []))
            
            # Preferred skills
            pref_quals = jd_data.get('preferred_qualifications', {})
            pref_skills = []
            pref_skills.extend(pref_quals.get('technical_skills', []))
            pref_skills.extend(pref_quals.get('soft_skills', []))
            pref_skills.extend(pref_quals.get('tools_and_technologies', []))
            
            sections['skills'] = ' '.join(req_skills + pref_skills)
            
            # Experience requirements
            experience_text = []
            experience_text.append(req_quals.get('experience_level', ''))
            experience_text.append(pref_quals.get('experience_level', ''))
            experience_text.extend(jd_data.get('responsibilities', []))
            sections['experience'] = ' '.join(experience_text)
            
            # Education requirements
            education_text = []
            education_text.extend(req_quals.get('education', []))
            education_text.extend(pref_quals.get('education', []))
            sections['education'] = ' '.join(education_text)
            
            # Certifications (extras)
            extras_text = []
            extras_text.extend(req_quals.get('certifications', []))
            extras_text.extend(pref_quals.get('certifications', []))
            sections['extras'] = ' '.join(extras_text)
            
            # Overall job description
            all_text = []
            all_text.append(jd_data.get('job_summary', ''))
            all_text.extend([sections['skills'], sections['experience'], 
                           sections['education'], sections['extras']])
            sections['overall'] = ' '.join(all_text)
            
        except Exception as e:
            logger.error(f"Error extracting JD sections: {e}")
            # Return empty sections if extraction fails
            sections = {
                'skills': '',
                'experience': '',
                'education': '',
                'extras': '',
                'overall': ''
            }
        
        return sections
    
    def _compute_section_similarities(self, resume_sections: Dict[str, str], 
                                    jd_sections: Dict[str, str]) -> Dict[str, float]:
        """Compute cosine similarity for each section"""
        similarities = {}
        
        for section in ['skills', 'experience', 'education', 'extras', 'overall']:
            resume_text = resume_sections.get(section, '')
            jd_text = jd_sections.get(section, '')
            
            if not resume_text.strip() or not jd_text.strip():
                similarities[section] = 0.0
                continue
            
            try:
                # Generate embeddings
                resume_embedding = self.model.encode([resume_text])
                jd_embedding = self.model.encode([jd_text])
                
                # Compute cosine similarity
                similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
                similarities[section] = float(similarity)
                
            except Exception as e:
                logger.error(f"Error computing similarity for {section}: {e}")
                similarities[section] = 0.0
        
        return similarities
    
    def _calculate_weighted_score(self, section_scores: Dict[str, float]) -> float:
        """Calculate final weighted ATS score"""
        try:
            # Apply weights to section scores
            weighted_score = (
                section_scores.get('skills', 0.0) * self.scoring_weights.skills_match +
                section_scores.get('experience', 0.0) * self.scoring_weights.experience_relevance +
                section_scores.get('education', 0.0) * self.scoring_weights.education_match +
                section_scores.get('extras', 0.0) * self.scoring_weights.extras
            )
            
            # Convert to 0-100 scale
            final_score = weighted_score * 100
            
            # Ensure score is within valid range
            final_score = max(0.0, min(100.0, final_score))
            
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating weighted score: {e}")
            return 0.0
    
    def _generate_score_analysis(self, section_scores: Dict[str, float], 
                               resume_sections: Dict[str, str],
                               jd_sections: Dict[str, str]) -> Dict[str, Any]:
        """Generate detailed analysis of scoring results"""
        analysis = {
            'score_breakdown': {},
            'strengths': [],
            'weaknesses': [],
            'missing_skills': [],
            'recommendations': []
        }
        
        # Score breakdown with weights
        weights = self.scoring_weights.to_dict()
        for section, score in section_scores.items():
            if section in weights:
                weighted_contribution = score * weights[section] * 100
                analysis['score_breakdown'][section] = {
                    'similarity_score': round(score, 3),
                    'weight': weights[section],
                    'weighted_contribution': round(weighted_contribution, 2)
                }
        
        # Identify strengths (scores > 0.7)
        for section, score in section_scores.items():
            if score > 0.7 and section != 'overall':
                analysis['strengths'].append(f"{section.title()}: {score:.2f} similarity")
        
        # Identify weaknesses (scores < 0.4)
        for section, score in section_scores.items():
            if score < 0.4 and section != 'overall':
                analysis['weaknesses'].append(f"{section.title()}: {score:.2f} similarity")
        
        # Basic missing skills identification
        # This is a simplified version - more sophisticated analysis in evaluator.py
        if section_scores.get('skills', 0) < 0.5:
            analysis['missing_skills'].append("Technical skills alignment is low")
        
        if section_scores.get('experience', 0) < 0.5:
            analysis['recommendations'].append("Consider highlighting more relevant work experience")
        
        return analysis