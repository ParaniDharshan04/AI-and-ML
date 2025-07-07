import re
import os
from dotenv import load_dotenv
load_dotenv()
from google.generativeai.generative_models import GenerativeModel
from typing import Dict, List

# LLM extraction using Gemini
# Set your Gemini API key as an environment variable: GEMINI_API_KEY
def llm_extract(prompt: str, text: str) -> dict:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        # Don't print error message to avoid cluttering logs
        return {}
    try:
        # Use GenerativeModel directly, API key is set via environment variable
        model = GenerativeModel('gemini-pro')
        system_prompt = (
            "You are a resume parsing assistant. Given the following text, extract the following fields as JSON: "
            "name, years_experience, skills (list), job_titles (list), companies (list), education, certifications (list), soft_skills (list). "
            "If a field is missing, use an empty string or empty list."
        )
        user_prompt = f"{prompt}\n\n{text}"
        response = model.generate_content([system_prompt, user_prompt])
        import json
        content = response.text
        # Try to find JSON in the response
        try:
            start = content.index('{')
            end = content.rindex('}') + 1
            json_str = content[start:end]
            return json.loads(json_str)
        except Exception:
            return {}
    except Exception as e:
        # Don't print error message to avoid cluttering logs
        return {}

# 1. Extract job requirements from job description
def extract_job_requirements(job_description: str) -> dict:
    # Try LLM extraction first (if available)
    # result = llm_extract("Extract job requirements", job_description)
    # if result: return result
    # Fallback: regex/simple parsing
    requirements = {
        'job_title': '',
        'required_skills': [],
        'preferred_skills': [],
        'years_experience': '',
        'education': '',
        'soft_skills': []
    }
    # Job title (first line or line with 'title')
    lines = job_description.split('\n')
    for line in lines:
        if 'title' in line.lower():
            requirements['job_title'] = line.strip()
            break
    if not requirements['job_title'] and lines:
        requirements['job_title'] = lines[0].strip()
    # Required skills (look for 'required', 'must have', etc.)
    req_skills = re.findall(r"(must have|required|proficient in|experience with) ([\w, \-\/]+)", job_description, re.IGNORECASE)
    for _, skills in req_skills:
        requirements['required_skills'] += [s.strip() for s in skills.split(',') if s.strip()]
    
    # Also look for common technical skills mentioned in the text
    tech_skills = re.findall(r"(machine learning|deep learning|natural language processing|nlp|python|java|tensorflow|pytorch|aws|azure|google cloud|sql|mysql|data analysis|computer vision|ai|artificial intelligence)", job_description, re.IGNORECASE)
    for skill in tech_skills:
        if skill.lower() not in [s.lower() for s in requirements['required_skills']]:
            requirements['required_skills'].append(skill.lower())
    
    # Preferred skills
    pref_skills = re.findall(r"(preferred|nice to have|plus:|bonus:) ([\w, \-\/]+)", job_description, re.IGNORECASE)
    for _, skills in pref_skills:
        requirements['preferred_skills'] += [s.strip() for s in skills.split(',') if s.strip()]
    # Years of experience
    exp = re.search(r"(\d+\+?) years? of experience", job_description, re.IGNORECASE)
    if exp:
        requirements['years_experience'] = exp.group(0)
    # Education
    edu = re.search(r"(bachelor|master|phd|degree in [\w ]+)", job_description, re.IGNORECASE)
    if edu:
        requirements['education'] = edu.group(0)
    # Soft skills
    softs = re.findall(r"(communication|teamwork|leadership|problem[- ]?solving|adaptability|creativity|work ethic|time management|attention to detail)", job_description, re.IGNORECASE)
    requirements['soft_skills'] = list(set([s.lower() for s in softs]))
    return requirements

# 2. Extract fields from resume
# Now uses LLM by default
def extract_resume_fields(resume_text: str, use_llm: bool = True) -> dict:
    # Try LLM extraction first (if enabled)
    if use_llm:
        result = llm_extract("Extract resume fields", resume_text)
        if result:
            # Clean and title-case the candidate name if present
            if 'name' in result and result['name']:
                result['name'] = ' '.join(result['name'].strip().split()).title()
            return result
    # Fallback: regex/section-based parsing
    fields = {
        'name': '',
        'years_experience': '',
        'skills': [],
        'job_titles': [],
        'companies': [],
        'education': '',
        'soft_skills': []
    }
    lines = resume_text.split('\n')
    # Name (first non-empty line, cleaned and title-cased)
    for line in lines:
        clean_line = line.strip()
        if clean_line and 'name' not in clean_line.lower():
            fields['name'] = ' '.join(clean_line.split()).title()
            break
    if not fields['name'] and lines:
        fields['name'] = ' '.join(lines[0].strip().split()).title()
    # Years of experience (look for various patterns)
    exp = re.search(r"(\d+\+?) years? of (relevant )?experience", resume_text, re.IGNORECASE)
    if exp:
        fields['years_experience'] = exp.group(0)
    else:
        # Try to infer from work experience section
        work_exp_section = re.search(r'(work|professional) experience[\s\S]{0,1000}', resume_text, re.IGNORECASE)
        if work_exp_section:
            years = re.findall(r'(\d{4})', work_exp_section.group(0))
            if len(years) >= 2:
                try:
                    fields['years_experience'] = f"{int(years[-1]) - int(years[0])} years (inferred)"
                except:
                    pass
    # Skills (look for 'skills' section)
    skills_section = re.search(r"skills[:\-\s]+([\w, \-\/]+)", resume_text, re.IGNORECASE)
    if skills_section:
        fields['skills'] = [s.strip() for s in skills_section.group(1).split(',') if s.strip()]
    
    # Also look for common technical skills mentioned in the resume text
    tech_skills = re.findall(r"(machine learning|deep learning|natural language processing|nlp|python|java|tensorflow|pytorch|aws|azure|google cloud|sql|mysql|data analysis|computer vision|ai|artificial intelligence|sentiment analysis|gemini api|canva|figma|vs code)", resume_text, re.IGNORECASE)
    for skill in tech_skills:
        if skill.lower() not in [s.lower() for s in fields['skills']]:
            fields['skills'].append(skill.lower())
    # Job titles and companies
    jobs = re.findall(r"(\w+ [Mm]anager|[Ee]ngineer|[Dd]eveloper|[Aa]nalyst|[Ss]cientist|[Dd]esigner|[Aa]rchitect|[Ll]ead|[Ii]ntern) at ([\w &]+)", resume_text)
    for title, company in jobs:
        fields['job_titles'].append(title.strip())
        fields['companies'].append(company.strip())
    # Education (section-based and regex)
    edu_section = re.search(r'(education|academic background|qualifications)[\s\S]{0,300}', resume_text, re.IGNORECASE)
    if edu_section:
        edu_lines = edu_section.group(0).split('\n')[1:4]
        fields['education'] = ' | '.join([l.strip() for l in edu_lines if l.strip()])
    else:
        edu = re.search(r"(bachelor|b\.sc|master|m\.sc|phd|doctorate|degree in [\w ]+|university|college)", resume_text, re.IGNORECASE)
        if edu:
            fields['education'] = edu.group(0)
    # Soft skills
    softs = re.findall(r"(communication|teamwork|leadership|problem[- ]?solving|adaptability|creativity|work ethic|time management|attention to detail)", resume_text, re.IGNORECASE)
    fields['soft_skills'] = list(set([s.lower() for s in softs]))
    return fields

# 3. Compare and score
def compare_resume_to_job(job_req: dict, resume_fields: dict) -> dict:
    # Debug output to diagnose extraction and comparison
    print("Job Requirements Extracted:", job_req)
    print("Resume Fields Extracted:", resume_fields)
    matched_skills = [s for s in job_req['required_skills'] if s.lower() in [r.lower() for r in resume_fields['skills']]]
    missing_skills = [s for s in job_req['required_skills'] if s.lower() not in [r.lower() for r in resume_fields['skills']]]
    matched_pref_skills = [s for s in job_req['preferred_skills'] if s.lower() in [r.lower() for r in resume_fields['skills']]]
    # Experience
    exp_match = job_req['years_experience'] in resume_fields['years_experience'] if job_req['years_experience'] else True
    # Education
    edu_match = job_req['education'].lower() in resume_fields['education'].lower() if job_req['education'] else True
    # Soft skills
    matched_soft = [s for s in job_req['soft_skills'] if s in resume_fields['soft_skills']]
    # Scoring
    score = 0
    total = 0
    if job_req['required_skills']:
        total += len(job_req['required_skills'])
        score += len(matched_skills)
    if job_req['preferred_skills']:
        total += len(job_req['preferred_skills'])
        score += 0.5 * len(matched_pref_skills)
    if job_req['years_experience']:
        total += 1
        score += 1 if exp_match else 0
    if job_req['education']:
        total += 1
        score += 1 if edu_match else 0
    if job_req['soft_skills']:
        total += len(job_req['soft_skills'])
        score += len(matched_soft)
    match_score = int((score / total) * 100) if total > 0 else 0
    # Verdict
    if match_score >= 80:
        verdict = 'Strong Match'
    elif match_score >= 50:
        verdict = 'Moderate Match'
    else:
        verdict = 'Weak Match'
    return {
        'candidate_name': resume_fields['name'],
        'match_score': match_score,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'education': resume_fields['education'],
        'experience': resume_fields['years_experience'],
        'soft_skills': matched_soft,
        'verdict': verdict
    }

# 4. Format output
def format_structured_report(result: dict) -> str:
    candidate_name = result['candidate_name'] or 'Not specified'
    return f"""
<div style='border:1px solid #e0e0e0; border-radius:10px; padding:20px; background:#fafbfc;'>
  <h4 style='margin-top:0; color:#2d3a4a;'><b>Candidate Name:</b> {candidate_name}</h4>
  <div style='margin-bottom:10px;'><b>Match Score:</b> {result['match_score']}/100</div>
  <div style='margin-bottom:10px;'><b>Summary of Matches:</b></div>
  <pre style='background:#fff; border-radius:6px; padding:12px; font-size:1em;'>
Matched Skills: {', '.join(result['matched_skills']) or 'None'}\nMissing Skills: {', '.join(result['missing_skills']) or 'None'}\nEducation: {result['education'] or 'Not specified'}\nExperience: {result['experience'] or 'Not specified'}\nSoft Skills: {', '.join(result['soft_skills']) or 'None'}
  </pre>
  <div style='margin-top:10px;'><b>Final Verdict:</b> {result['verdict']}</div>
</div>
"""

def llm_screen_resume(resume_text: str, jd_text: str) -> dict:
    from google.generativeai.generative_models import GenerativeModel
    import os
    import json
    model = GenerativeModel('gemini-pro')
    prompt = f'''
You are an AI resume screening assistant.

Your task is to analyze a candidate's resume and compare it with a given job description (JD). You must parse both the resume and the JD, extract relevant details, and compute a match score based on the alignment of skills, experience, and qualifications.

### Instructions:

1. **Parse the Resume**:
   - Extract structured information including:
     - Name (if available), Education, Work Experience, Skills, Certifications, Projects
   - Focus especially on technical skills, tools, platforms, and programming languages.

2. **Parse the Job Description**:
   - Identify key elements:
     - Required Skills
     - Preferred Skills
     - Minimum Experience
     - Educational Qualifications
     - Job Responsibilities

3. **Match Criteria**:
   - Match the resume against the JD using both exact keyword matching and semantic similarity.
   - Consider:
     - Skill overlap
     - Relevance of job titles
     - Alignment of past responsibilities with current role expectations
     - Years of experience vs. required years
     - Certifications or tools required

4. **Scoring System**:
   - Assign match percentages for each category:
     - Skills Match: XX%
     - Experience Match: XX%
     - Education Match: XX%
     - Overall Match Score: XX%
   - Identify key **missing or weak areas** in the resume.

5. **Output Format**:
   - Return a JSON object with the following fields:
     - name
     - skills_match (int)
     - experience_match (int)
     - education_match (int)
     - overall_match_score (int)
     - gaps_identified (list of strings)
     - summary (string)

6. **Be concise but specific. Do not make assumptions if data is missing.**

Resume Text: {resume_text}
Job Description Text: {jd_text}
Now perform the analysis and return your structured output as JSON.
'''
    response = model.generate_content(prompt)
    content = response.text
    try:
        start = content.index('{')
        end = content.rindex('}') + 1
        json_str = content[start:end]
        return json.loads(json_str)
    except Exception:
        return {"summary": content}

# Replace the main analysis path to use the new LLM-based function
def structured_resume_analysis(resume_text: str, jd_text: str) -> dict:
    result = llm_screen_resume(resume_text, jd_text)
    # Fallback formatting if summary is present
    if 'summary' in result:
        return {"structured_report": result['summary']}
    # Otherwise, format a detailed report
    report = f"""
<div style='border:1px solid #e0e0e0; border-radius:10px; padding:20px; background:#fafbfc;'>
  <h4 style='margin-top:0; color:#2d3a4a;'><b>Candidate Name:</b> {result.get('name', 'Not specified')}</h4>
  <div style='margin-bottom:10px;'><b>Skills Match:</b> {result.get('skills_match', 'N/A')}%</div>
  <div style='margin-bottom:10px;'><b>Experience Match:</b> {result.get('experience_match', 'N/A')}%</div>
  <div style='margin-bottom:10px;'><b>Education Match:</b> {result.get('education_match', 'N/A')}%</div>
  <div style='margin-bottom:10px;'><b>Overall Match Score:</b> {result.get('overall_match_score', 'N/A')}%</div>
  <div style='margin-bottom:10px;'><b>Gaps Identified:</b> {', '.join(result.get('gaps_identified', [])) or 'None'}</div>
</div>
"""
    return {"structured_report": report} 