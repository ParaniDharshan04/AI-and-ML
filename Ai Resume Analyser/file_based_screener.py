#!/usr/bin/env python3
"""
File-Based Resume Screener
Reads job descriptions and resumes from text files for batch processing.
"""

from resume_screener import ResumeMatcherRAG
import os
import sys

def read_text_file(filename):
    """Read text content from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error reading file '{filename}': {e}")
        return None

def save_results_to_file(result, output_filename):
    """Save analysis results to a file."""
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("AI-Powered Resume Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Relevance Score: {result['relevance_score'] * 100:.1f}%\n\n")
            f.write(f"Matched Skills: {', '.join(result['matched_skills']) if result['matched_skills'] else 'None'}\n\n")
            f.write("Relevant Resume Sections:\n")
            for i, section in enumerate(result['highlighted_sections'], 1):
                f.write(f"{i}. {section}\n")
            f.write("\nDetailed Assessments:\n")
            for i, assessment in enumerate(result['llm_assessments'], 1):
                f.write(f"\nRequirement {i}: {assessment['jd_chunk']}\n")
                f.write(f"AI Response: {assessment['llm_response']}\n")
        print(f"✅ Results saved to {output_filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Main function for file-based screening."""
    print("📁 File-Based Resume Screener")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) == 3:
        jd_file = sys.argv[1]
        resume_file = sys.argv[2]
    else:
        # Interactive file input
        jd_file = input("Enter job description file path: ").strip()
        resume_file = input("Enter resume file path: ").strip()
    
    # Read files
    print(f"\n📖 Reading job description from: {jd_file}")
    job_description = read_text_file(jd_file)
    if not job_description:
        return
    
    print(f"📖 Reading resume from: {resume_file}")
    resume_text = read_text_file(resume_file)
    if not resume_text:
        return
    
    print("\n🔄 Processing... Please wait while the AI analyzes the documents...")
    
    try:
        # Initialize and run analysis
        screener = ResumeMatcherRAG()
        result = screener.match_resume(job_description, resume_text)
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        score = result['relevance_score']
        print(f"\n🎯 RELEVANCE SCORE: {score * 100:.1f}%")
        
        # Visual score bar
        filled = int(score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"[{bar}]")
        
        if result['matched_skills']:
            print(f"\n✅ MATCHED SKILLS:")
            for skill in result['matched_skills']:
                print(f"   • {skill}")
        else:
            print(f"\n❌ NO SKILLS MATCHED")
        
        # Save results
        output_file = f"analysis_{os.path.splitext(os.path.basename(resume_file))[0]}.txt"
        save_results_to_file(result, output_file)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 