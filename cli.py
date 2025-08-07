"""
Command Line Interface for Resume Screening System
Provides easy-to-use CLI for ATS scoring and analysis
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional
import logging

from utils import setup_logging, save_results_to_json, validate_file_exists, ProgressTracker
from file_processor import FileProcessor, TextPreprocessor
from llm_parser import LLMParser
from semantic_matcher import SemanticMatcher
from evaluator import CandidateEvaluator
from config import DEFAULT_WEIGHTS

# Setup logging
logger = setup_logging()

class ResumeScreeningSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self.text_preprocessor = TextPreprocessor()
        self.llm_parser = LLMParser()
        self.semantic_matcher = SemanticMatcher()
        self.evaluator = CandidateEvaluator()
        self.progress = ProgressTracker()
    
    def screen_resume(self, resume_path: str, jd_path: str = None, jd_text: str = None, 
                     output_path: str = None, verbose: bool = False) -> dict:
        """
        Complete resume screening pipeline
        
        Args:
            resume_path: Path to resume file
            jd_path: Path to job description file (optional if jd_text provided)
            jd_text: Job description text (optional if jd_path provided)
            output_path: Output file path for results
            verbose: Whether to show detailed progress
            
        Returns:
            Complete screening results
        """
        results = {
            'input_files': {
                'resume_path': resume_path,
                'jd_path': jd_path,
                'jd_text_provided': bool(jd_text)
            },
            'processing_steps': {},
            'final_results': {}
        }
        
        try:
            # Step 1: Extract text from files
            if verbose:
                click.echo(self.progress.next_step())
            
            resume_extraction = self.file_processor.extract_text_from_file(resume_path)
            results['processing_steps']['resume_extraction'] = resume_extraction
            
            if not resume_extraction['extraction_successful']:
                raise ValueError(f"Failed to extract resume: {resume_extraction['error_message']}")
            
            # Handle job description input
            if jd_path:
                jd_extraction = self.file_processor.extract_text_from_file(jd_path)
                results['processing_steps']['jd_extraction'] = jd_extraction
                
                if not jd_extraction['extraction_successful']:
                    raise ValueError(f"Failed to extract job description: {jd_extraction['error_message']}")
                
                jd_text = jd_extraction['cleaned_text']
            elif jd_text:
                jd_extraction = {
                    'cleaned_text': jd_text,
                    'word_count': len(jd_text.split()),
                    'extraction_successful': True
                }
                results['processing_steps']['jd_extraction'] = jd_extraction
            else:
                raise ValueError("Either jd_path or jd_text must be provided")
            
            # Step 2: Parse resume with LLM
            if verbose:
                click.echo(self.progress.next_step())
                
            resume_parsing = self.llm_parser.parse_with_llm(
                resume_extraction['cleaned_text'], 'resume'
            )
            results['processing_steps']['resume_parsing'] = resume_parsing
            
            if not resume_parsing['parsing_successful']:
                raise ValueError(f"Failed to parse resume: {resume_parsing['error_message']}")
            
            # Step 3: Parse job description with LLM
            if verbose:
                click.echo(self.progress.next_step())
                
            jd_parsing = self.llm_parser.parse_with_llm(jd_text, 'job_description')
            results['processing_steps']['jd_parsing'] = jd_parsing
            
            if not jd_parsing['parsing_successful']:
                raise ValueError(f"Failed to parse job description: {jd_parsing['error_message']}")
            
            # Step 4: Preprocess text
            if verbose:
                click.echo(self.progress.next_step())
                
            resume_preprocessing = self.text_preprocessor.preprocess_text(
                resume_extraction['cleaned_text']
            )
            jd_preprocessing = self.text_preprocessor.preprocess_text(jd_text)
            
            results['processing_steps']['preprocessing'] = {
                'resume': resume_preprocessing,
                'job_description': jd_preprocessing
            }
            
            # Step 5: Compute semantic similarity and ATS score
            if verbose:
                click.echo(self.progress.next_step())
                
            scoring_results = self.semantic_matcher.embed_and_score_with_sbert(
                resume_parsing['parsed_data'], 
                jd_parsing['parsed_data']
            )
            results['processing_steps']['scoring'] = scoring_results
            
            if not scoring_results['computation_successful']:
                raise ValueError(f"Failed to compute ATS score: {scoring_results['error_message']}")
            
            # Step 6: Generate recommendations
            if verbose:
                click.echo(self.progress.next_step())
                
            evaluation_results = self.evaluator.evaluate_with_llm(
                resume_parsing['parsed_data'],
                jd_parsing['parsed_data'],
                scoring_results
            )
            results['processing_steps']['evaluation'] = evaluation_results
            
            if not evaluation_results['evaluation_successful']:
                logger.warning(f"Evaluation partially failed: {evaluation_results['error_message']}")
            
            # Compile final results
            results['final_results'] = {
                'ats_score': scoring_results['ats_score'],
                'candidate_summary': evaluation_results.get('candidate_summary', ''),
                'section_evaluation': evaluation_results.get('section_evaluation', {}),
                'section_scores': scoring_results['section_scores'],
                'scoring_weights': scoring_results['scoring_weights'],
                'recommendations': evaluation_results.get('recommendations', []),
                'predicted_improved_score': evaluation_results.get('predicted_improved_score', 
                                                                 scoring_results['ats_score']),
                'detailed_analysis': scoring_results.get('detailed_analysis', {})
            }
            
            # Save results if output path provided
            if output_path:
                saved_path = save_results_to_json(results, output_path)
                results['output_file'] = saved_path
                if verbose:
                    click.echo(f"Results saved to: {saved_path}")
            
            if verbose:
                click.echo(self.progress.next_step())  # Complete!
            
            return results
            
        except Exception as e:
            logger.error(f"Resume screening failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results


@click.group()
@click.version_option(version='1.0.0')
def main():
    """Resume Screening System - ATS scoring using Azure OpenAI GPT-4o and Sentence-BERT"""
    pass

@main.command()
@click.argument('resume_path', type=click.Path(exists=True))
@click.option('--jd-path', '-j', type=click.Path(exists=True), 
              help='Path to job description file')
@click.option('--jd-text', '-t', type=str, 
              help='Job description as text (alternative to --jd-path)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path for results (JSON format)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Show detailed progress information')
@click.option('--format', '-f', type=click.Choice(['json', 'summary']), default='summary',
              help='Output format (json for full results, summary for concise output)')
def screen(resume_path, jd_path, jd_text, output, verbose, format):
    """Screen a resume against a job description"""
    
    # Validation
    if not jd_path and not jd_text:
        click.echo("Error: Either --jd-path or --jd-text must be provided", err=True)
        sys.exit(1)
    
    if jd_path and not validate_file_exists(jd_path):
        click.echo(f"Error: Job description file not found: {jd_path}", err=True)
        sys.exit(1)
    
    # Initialize system
    system = ResumeScreeningSystem()
    
    # Process resume
    if verbose:
        click.echo("Starting resume screening process...")
        click.echo(f"Resume: {resume_path}")
        if jd_path:
            click.echo(f"Job Description: {jd_path}")
        else:
            click.echo("Job Description: Provided as text")
        click.echo("-" * 50)
    
    # Run screening
    results = system.screen_resume(
        resume_path=resume_path,
        jd_path=jd_path,
        jd_text=jd_text,
        output_path=output,
        verbose=verbose
    )
    
    # Display results
    if 'error' in results:
        click.echo(f"Screening failed: {results['error']}", err=True)
        sys.exit(1)
    
    if format == 'json':
        click.echo(json.dumps(results, indent=2))
    else:
        # Summary format
        final_results = results.get('final_results', {})
        
        click.echo("\n" + "=" * 60)
        click.echo("RESUME SCREENING RESULTS")
        click.echo("=" * 60)
        
        click.echo(f"\n🎯 ATS SCORE: {final_results.get('ats_score', 0)}/100")
        
        if final_results.get('candidate_summary'):
            click.echo(f"\n📋 CANDIDATE SUMMARY:")
            click.echo(final_results['candidate_summary'])
        
        # Section scores
        section_scores = final_results.get('section_scores', {})
        if section_scores:
            click.echo(f"\n📊 SECTION BREAKDOWN:")
            for section, score in section_scores.items():
                if section != 'overall':
                    click.echo(f"  • {section.title()}: {score:.2f}")
        
        # Top recommendations
        recommendations = final_results.get('recommendations', [])
        if recommendations:
            click.echo(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                click.echo(f"  {i}. {rec.get('title', 'N/A')} (Impact: +{rec.get('expected_impact', 0)} points)")
        
        predicted_score = final_results.get('predicted_improved_score', 0)
        current_score = final_results.get('ats_score', 0)
        if predicted_score > current_score:
            click.echo(f"\n📈 PREDICTED SCORE AFTER IMPROVEMENTS: {predicted_score}/100")
        
        if output:
            click.echo(f"\n💾 Full results saved to: {output}")
        
        click.echo("\n" + "=" * 60)

@main.command()
def test():
    """Run system test with sample data"""
    click.echo("Running Resume Screening System test...")
    
    # This will be implemented with sample data
    click.echo("Creating sample resume and job description...")
    
    # Create sample files for testing
    sample_resume = create_sample_resume()
    sample_jd = create_sample_job_description()
    
    # Save to temporary files
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_resume)
        resume_path = f.name
    
    system = ResumeScreeningSystem()
    
    try:
        results = system.screen_resume(
            resume_path=resume_path,
            jd_text=sample_jd,
            verbose=True
        )
        
        if 'error' not in results:
            click.echo("\n✅ Test completed successfully!")
            click.echo(f"Sample ATS Score: {results['final_results']['ats_score']}/100")
        else:
            click.echo(f"\n❌ Test failed: {results['error']}")
            
    finally:
        # Cleanup
        Path(resume_path).unlink(missing_ok=True)

@main.command()
@click.option('--skills', type=float, default=DEFAULT_WEIGHTS.skills_match,
              help=f'Skills weight (default: {DEFAULT_WEIGHTS.skills_match})')
@click.option('--experience', type=float, default=DEFAULT_WEIGHTS.experience_relevance,
              help=f'Experience weight (default: {DEFAULT_WEIGHTS.experience_relevance})')
@click.option('--education', type=float, default=DEFAULT_WEIGHTS.education_match,
              help=f'Education weight (default: {DEFAULT_WEIGHTS.education_match})')
@click.option('--extras', type=float, default=DEFAULT_WEIGHTS.extras,
              help=f'Extras weight (default: {DEFAULT_WEIGHTS.extras})')
def configure(skills, experience, education, extras):
    """Configure scoring weights"""
    total = skills + experience + education + extras
    if abs(total - 1.0) > 0.001:
        click.echo(f"Error: Weights must sum to 1.0 (current sum: {total})", err=True)
        sys.exit(1)
    
    click.echo("Current scoring weights:")
    click.echo(f"  Skills: {skills:.2f} ({skills*100:.0f}%)")
    click.echo(f"  Experience: {experience:.2f} ({experience*100:.0f}%)")
    click.echo(f"  Education: {education:.2f} ({education*100:.0f}%)")
    click.echo(f"  Extras: {extras:.2f} ({extras*100:.0f}%)")
    click.echo("\nNote: Weights are applied during runtime. Modify config.py for permanent changes.")

def create_sample_resume():
    """Create sample resume for testing"""
    return """
John Smith
Email: john.smith@email.com
Phone: (555) 123-4567
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Experienced Full Stack Developer with 5+ years of expertise in React, Node.js, and Python. 
Proven track record of building scalable web applications and leading development teams.

TECHNICAL SKILLS
• Programming Languages: JavaScript, Python, Java, TypeScript
• Frontend: React, Vue.js, HTML5, CSS3, Bootstrap, Tailwind CSS
• Backend: Node.js, Express, Django, FastAPI
• Databases: PostgreSQL, MongoDB, MySQL
• Cloud: AWS, Azure, Docker, Kubernetes
• Tools: Git, Jenkins, Jest, Cypress

PROFESSIONAL EXPERIENCE

Senior Full Stack Developer | TechCorp Inc. | 2021 - Present
• Led development of e-commerce platform serving 100K+ users using React and Node.js
• Implemented microservices architecture reducing system latency by 40%
• Mentored junior developers and established coding standards
• Collaborated with cross-functional teams to deliver features on time

Full Stack Developer | StartupXYZ | 2019 - 2021
• Built responsive web applications using React, Redux, and Express
• Developed RESTful APIs serving mobile and web clients
• Implemented automated testing reducing bugs by 30%
• Participated in agile development process and code reviews

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2019
GPA: 3.7/4.0
Relevant Coursework: Data Structures, Algorithms, Database Systems

PROJECTS
E-Commerce Platform
• Built full-stack application with React frontend and Node.js backend
• Integrated payment processing and inventory management
• Technologies: React, Node.js, MongoDB, Stripe API

CERTIFICATIONS
• AWS Certified Developer Associate (2022)
• MongoDB Certified Developer (2021)
"""

def create_sample_job_description():
    """Create sample job description for testing"""
    return """
Senior Full Stack Developer - Remote

Company: InnovateTech Solutions
Location: Remote (US)
Employment Type: Full-time

JOB SUMMARY
We are seeking a highly skilled Senior Full Stack Developer to join our growing team. 
You will be responsible for developing and maintaining web applications using modern 
technologies and best practices.

KEY RESPONSIBILITIES
• Design and develop scalable web applications using React and Node.js
• Collaborate with product managers and designers to implement new features
• Write clean, maintainable code following best practices
• Participate in code reviews and mentor junior developers
• Optimize application performance and ensure security compliance
• Work with DevOps team to deploy applications on cloud platforms

REQUIRED QUALIFICATIONS
• Bachelor's degree in Computer Science or related field
• 5+ years of experience in full stack development
• Strong proficiency in JavaScript, React, and Node.js
• Experience with database systems (PostgreSQL, MongoDB)
• Knowledge of RESTful API design and development
• Familiarity with version control systems (Git)
• Experience with cloud platforms (AWS, Azure)
• Strong problem-solving skills and attention to detail

PREFERRED QUALIFICATIONS
• Experience with TypeScript and modern JavaScript frameworks
• Knowledge of containerization technologies (Docker, Kubernetes)
• Experience with CI/CD pipelines
• Familiarity with microservices architecture
• Previous experience in mentoring or team leadership
• Knowledge of testing frameworks (Jest, Cypress)

TECHNICAL REQUIREMENTS
• React.js, Redux, HTML5, CSS3
• Node.js, Express.js
• PostgreSQL, MongoDB
• AWS or Azure cloud services
• Git version control
• Agile/Scrum methodology

BENEFITS
• Competitive salary and equity package
• Health, dental, and vision insurance
• Unlimited PTO policy
• Professional development budget
• Remote work flexibility

COMPANY INFO
InnovateTech Solutions is a fast-growing SaaS company focused on providing 
innovative solutions for enterprise customers. We value collaboration, 
continuous learning, and work-life balance.
"""

if __name__ == '__main__':
    main()