#!/usr/bin/env python3
"""
Interactive Resume Screener
A user-friendly interface for the AI-powered resume matching system.
"""

from resume_screener import ResumeMatcherRAG
import sys

def get_user_input():
    """Get job description and resume from user input."""
    print("🤖 AI-Powered Resume Screener")
    print("=" * 50)
    
    print("\n📋 JOB DESCRIPTION:")
    print("Enter the job description (press Enter twice when done):")
    jd_lines = []
    while True:
        line = input()
        if line == "" and jd_lines and jd_lines[-1] == "":
            break
        jd_lines.append(line)
    
    job_description = "\n".join(jd_lines[:-1])  # Remove the last empty line
    
    print("\n📄 RESUME:")
    print("Enter the candidate's resume (press Enter twice when done):")
    resume_lines = []
    while True:
        line = input()
        if line == "" and resume_lines and resume_lines[-1] == "":
            break
        resume_lines.append(line)
    
    resume_text = "\n".join(resume_lines[:-1])  # Remove the last empty line
    
    return job_description.strip(), resume_text.strip()

def display_results(result):
    """Display the analysis results in a formatted way."""
    print("\n" + "=" * 50)
    print("📊 ANALYSIS RESULTS")
    print("=" * 50)
    
    # Relevance Score
    score = result['relevance_score']
    print(f"\n🎯 RELEVANCE SCORE: {score * 100:.1f}%")
    
    # Visual score bar
    filled = int(score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"[{bar}]")
    
    # Matched Skills
    if result['matched_skills']:
        print(f"\n✅ MATCHED SKILLS:")
        for skill in result['matched_skills']:
            print(f"   • {skill}")
    else:
        print(f"\n❌ NO SKILLS MATCHED")
    
    # Highlighted Sections
    if result['highlighted_sections']:
        print(f"\n🔍 RELEVANT RESUME SECTIONS:")
        for i, section in enumerate(result['highlighted_sections'], 1):
            print(f"   {i}. {section[:100]}{'...' if len(section) > 100 else ''}")
    
    # Detailed LLM Assessments
    print(f"\n🤖 DETAILED ASSESSMENTS:")
    for i, assessment in enumerate(result['llm_assessments'], 1):
        print(f"\n   Requirement {i}: {assessment['jd_chunk']}")
        print(f"   AI Response: {assessment['llm_response']}")

def main():
    """Main function to run the interactive screener."""
    try:
        # Get input from user
        job_description, resume_text = get_user_input()
        
        if not job_description or not resume_text:
            print("❌ Error: Both job description and resume are required.")
            return
        
        print("\n🔄 Processing... Please wait while the AI analyzes the documents...")
        
        # Initialize the screener
        screener = ResumeMatcherRAG()
        
        # Run the analysis
        result = screener.match_resume(job_description, resume_text)
        
        # Display results
        display_results(result)
        
        # Save results option
        save = input("\n💾 Would you like to save the results to a file? (y/n): ").lower()
        if save == 'y':
            filename = input("Enter filename (e.g., analysis_results.txt): ")
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("AI-Powered Resume Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Relevance Score: {result['relevance_score'] * 100:.1f}%\n\n")
                f.write(f"Matched Skills: {', '.join(result['matched_skills']) if result['matched_skills'] else 'None'}\n\n")
                f.write("Detailed Assessments:\n")
                for i, assessment in enumerate(result['llm_assessments'], 1):
                    f.write(f"\nRequirement {i}: {assessment['jd_chunk']}\n")
                    f.write(f"AI Response: {assessment['llm_response']}\n")
            
            print(f"✅ Results saved to {filename}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Analysis cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure all required packages are installed.")

if __name__ == "__main__":
    main() 