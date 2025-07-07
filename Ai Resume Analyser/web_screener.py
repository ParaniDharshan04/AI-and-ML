#!/usr/bin/env python3
"""
Web-Based Resume Screener
A Flask web application for uploading resumes and analyzing them against job descriptions.
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from resume_screener import ResumeMatcherRAG
import tempfile
import PyPDF2
import docx
from structured_resume_analysis import extract_job_requirements, extract_resume_fields, compare_resume_to_job, format_structured_report

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    """Extract text from uploaded file (.txt, .pdf, .docx)."""
    ext = file_path.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == 'pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text if text.strip() else "[No extractable text found in PDF]"
        elif ext == 'docx':
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text if text.strip() else "[No extractable text found in DOCX]"
        else:
            return f"Unsupported file type: .{ext}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    try:
        # Check if file was uploaded
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload .txt, .pdf, .doc, or .docx files'}), 400
        
        # Get job description from form
        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename or "uploaded_resume")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from file (now handles .txt, .pdf, .docx)
        resume_text = extract_text_from_file(file_path)

        # Structured analysis
        job_req = extract_job_requirements(job_description)
        resume_fields = extract_resume_fields(resume_text)
        comparison = compare_resume_to_job(job_req, resume_fields)
        structured_report = format_structured_report(comparison)

        # Prepare response data
        response_data = {
            'success': True,
            'filename': filename,
            'structured_report': structured_report,
            'structured_score': comparison['match_score'],
            'structured_verdict': comparison['verdict'],
            'job_description': job_description,
            'resume_preview': resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
        }
        
        # Save results to file
        results_filename = f"analysis_{timestamp}.json"
        results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        response_data['results_file'] = results_filename
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_results(filename):
    """Download analysis results as JSON file."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

@app.route('/results/<filename>')
def view_results(filename):
    """View analysis results in a formatted page."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return render_template('results.html', data=data)
    except Exception as e:
        return f"Error loading results: {str(e)}", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 