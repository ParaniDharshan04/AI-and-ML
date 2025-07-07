# 🤖 AI-Powered Resume Screener

An intelligent resume screening system that uses **Retrieval-Augmented Generation (RAG)** to match resumes against job descriptions with detailed AI analysis.

## 🚀 Features

- **AI-Powered Analysis**: Uses advanced language models for semantic understanding
- **Multiple Interfaces**: Web, command-line, and file-based options
- **Detailed Assessments**: Provides reasoning for each match/mismatch
- **Skill Extraction**: Automatically identifies matched skills and experiences
- **Modern UI**: Beautiful web interface with drag-and-drop file upload
- **Export Results**: Download analysis results in JSON format

## 📋 Requirements

- Python 3.8+
- Required packages (see installation below)

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install sentence-transformers transformers torch numpy scikit-learn flask werkzeug tf-keras
   ```

## 🎯 How to Use

### 🌐 **Web Interface (Recommended)**

1. **Start the web server**:
   ```bash
   python web_screener.py
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Upload your resume**:
   - Drag & drop or click to browse
   - Supported formats: `.txt`, `.pdf`, `.doc`, `.docx`
   - Paste the job description in the text area

4. **Get instant analysis**:
   - Relevance score with visual progress bar
   - Matched skills highlighted
   - Detailed AI assessments for each requirement
   - Download results option

### 📁 **File-Based Processing**

1. **Prepare your files**:
   - Job description: `job_description.txt`
   - Resume: `resume.txt`

2. **Run the analysis**:
   ```bash
   python file_based_screener.py job_description.txt resume.txt
   ```

3. **View results** in the terminal and saved file

### 💻 **Interactive Command Line**

1. **Run the interactive version**:
   ```bash
   python interactive_screener.py
   ```

2. **Follow the prompts** to enter job description and resume text

3. **Get formatted results** with option to save

### 🔧 **Direct Code Usage**

```python
from resume_screener import ResumeMatcherRAG

# Initialize the screener
screener = ResumeMatcherRAG()

# Analyze resume
result = screener.match_resume(job_description, resume_text)

# Access results
print(f"Relevance Score: {result['relevance_score']}")
print(f"Matched Skills: {result['matched_skills']}")
```

## 📊 Output Structure

The system provides comprehensive analysis results:

```json
{
  "relevance_score": 0.75,
  "relevance_percentage": 75.0,
  "matched_skills": ["Python", "scikit-learn", "AWS"],
  "highlighted_sections": ["Relevant resume sections..."],
  "llm_assessments": [
    {
      "jd_chunk": "Requirement text",
      "resume_chunks": ["Relevant resume parts"],
      "llm_response": "AI assessment with reasoning"
    }
  ]
}
```

## 🏗️ Architecture

### **RAG (Retrieval-Augmented Generation) Pipeline**

1. **Text Preprocessing**: Clean and normalize text
2. **Chunking**: Split documents into overlapping segments
3. **Embedding**: Convert text to semantic vectors
4. **Retrieval**: Find most relevant resume sections for each job requirement
5. **Generation**: AI assessment of matches with detailed reasoning
6. **Aggregation**: Combine results into comprehensive analysis

### **AI Models Used**

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model**: `google/flan-t5-small`
- **Vector Search**: Cosine similarity with NumPy

## 📁 Project Structure

```
resume analyser/
├── resume_screener.py          # Core AI screening engine
├── web_screener.py            # Flask web application
├── interactive_screener.py    # Command-line interface
├── file_based_screener.py     # File processing interface
├── templates/
│   └── index.html             # Web interface template
├── uploads/                   # Uploaded files and results
├── example_job_description.txt # Sample job description
├── example_resume.txt         # Sample resume
└── README.md                  # This file
```

## 🎨 Web Interface Features

- **Modern Design**: Beautiful gradient background with glass-morphism effects
- **Drag & Drop**: Intuitive file upload with visual feedback
- **Real-time Processing**: Live progress indicators during analysis
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Visual Results**: Progress bars, skill badges, and formatted assessments
- **Export Options**: Download results as JSON files

## 🔍 Example Usage

### Sample Job Description
```
Senior Python Developer

We are seeking a talented Senior Python Developer with:
- 5+ years of Python development experience
- Strong knowledge of machine learning frameworks
- Experience with cloud platforms (AWS, Azure)
- Proficiency in data analysis and visualization
```

### Sample Resume
```
JOHN DOE
Software Engineer

EXPERIENCE:
- 6 years of Python development
- Built ML models using scikit-learn and TensorFlow
- Deployed solutions on AWS and Azure
- Experience with REST APIs and Docker
```

### Expected Output
```
🎯 RELEVANCE SCORE: 85.0%
[██████████████████░░]

✅ MATCHED SKILLS:
   • Python
   • scikit-learn
   • AWS
   • Azure
   • REST APIs
   • Docker

🤖 DETAILED ASSESSMENTS:
   Requirement 1: 5+ years of Python development
   AI Response: Yes, the candidate has 6 years of experience...
```

## 🚀 Advanced Features

- **Batch Processing**: Analyze multiple resumes against one job description
- **Custom Models**: Easily swap embedding and language models
- **Configurable Parameters**: Adjust chunk sizes, similarity thresholds
- **Extensible Architecture**: Add new file format support or analysis types

## 🤝 Contributing

Feel free to contribute improvements:
- Add support for more file formats (PDF, DOCX)
- Enhance the AI models or prompts
- Improve the web interface
- Add new analysis features

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Resume Screening! 🎯** 