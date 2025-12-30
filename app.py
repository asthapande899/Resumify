import streamlit as st
import pickle
import re
import PyPDF2
import io
import numpy as np
import base64
import os
import sys
import subprocess
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
from utils import get_suggestions, SKILL_MAP
from sklearn.metrics.pairwise import cosine_similarity

# Check if model exists, train if not
if not os.path.exists("model/tfidf.pkl") or not os.path.exists("model/clf.pkl"):
    with st.spinner("Training model for the first time... This may take a minute."):
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Run training script
        try:
            import train
            from importlib import reload
            reload(train)
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Error training model: {e}")
            st.info("Please run 'python train.py' manually in your terminal.")

# Now try to load the model
try:
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    clf = pickle.load(open("model/clf.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure you have run 'python train.py' to create the model files.")
    st.stop()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.lower()

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def create_pdf_report(analysis_data, suggestions, tips):
    """Create a PDF report with analysis results"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_path = tmp_file.name
    
    # Create PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Resume Analysis Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated on: {analysis_data['timestamp']}")
    
    # Line separator
    c.line(50, height - 80, width - 50, height - 80)
    
    y_position = height - 100
    
    # 1. Overview Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "OVERVIEW")
    y_position -= 25
    
    c.setFont("Helvetica", 12)
    overview_text = [
        f"Effectiveness Score: {analysis_data['effectiveness']:.0f}%",
        f"Predicted Role: {analysis_data['predicted_role']}",
        f"Target Role: {analysis_data['target_role']}",
        f"Role Match: {'Yes' if analysis_data['role_match'] else 'No'}"
    ]
    
    for line in overview_text:
        c.drawString(60, y_position, line)
        y_position -= 20
    
    y_position -= 10
    
    # 2. Score Breakdown
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "SCORE BREAKDOWN")
    y_position -= 25
    
    c.setFont("Helvetica", 12)
    score_text = [
        f"Role Match: {'40/40' if analysis_data['role_match'] else '0/40'}",
        f"Skills Match: {len(analysis_data['present_skills'])}/{len(analysis_data['required_skills'])} skills found ({analysis_data['skill_match_percentage']:.1f}/40)",
        f"Job Description Match: {analysis_data['desc_score']:.1f}/20"
    ]
    
    for line in score_text:
        c.drawString(60, y_position, line)
        y_position -= 20
    
    y_position -= 10
    
    # 3. Skills Analysis
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "SKILLS ANALYSIS")
    y_position -= 25
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, y_position, f"Skills Present ({len(analysis_data['present_skills'])}):")
    y_position -= 20
    
    c.setFont("Helvetica", 11)
    present_text = ', '.join(analysis_data['present_skills'][:8])
    # Split long text into multiple lines
    while present_text:
        c.drawString(70, y_position, present_text[:80])
        y_position -= 15
        present_text = present_text[80:]
    
    y_position -= 10
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, y_position, f"Skills to Add ({len(analysis_data['skills_missing'])}):")
    y_position -= 20
    
    c.setFont("Helvetica", 11)
    missing_text = ', '.join(analysis_data['skills_missing'][:8])
    while missing_text:
        c.drawString(70, y_position, missing_text[:80])
        y_position -= 15
        missing_text = missing_text[80:]
    
    y_position -= 20
    
    # Check if we need a new page
    if y_position < 100:
        c.showPage()
        y_position = height - 50
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 30, f"Page 2 - Generated on: {analysis_data['timestamp']}")
        y_position = height - 80
    
    # 4. Project Suggestions
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "PROJECT SUGGESTIONS")
    y_position -= 25
    
    c.setFont("Helvetica", 11)
    for suggestion in suggestions[:4]:  # Show first 4 suggestions
        if y_position < 50:
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 30, f"Page 3 - Generated on: {analysis_data['timestamp']}")
            y_position = height - 80
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "PROJECT SUGGESTIONS (continued)")
            y_position -= 25
            c.setFont("Helvetica", 11)
        
        c.drawString(60, y_position, f"• {suggestion}")
        y_position -= 18
    
    y_position -= 10
    
    # 5. Improvement Tips
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "IMPROVEMENT TIPS")
    y_position -= 25
    
    c.setFont("Helvetica", 11)
    for tip in tips[:6]:  # Show first 6 tips
        if y_position < 50:
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 30, f"Page 4 - Generated on: {analysis_data['timestamp']}")
            y_position = height - 80
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "IMPROVEMENT TIPS (continued)")
            y_position -= 25
            c.setFont("Helvetica", 11)
        
        c.drawString(60, y_position, f"• {tip}")
        y_position -= 18
    
    y_position -= 20
    
    # 6. Recommendations
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "RECOMMENDATIONS")
    y_position -= 25
    
    c.setFont("Helvetica", 11)
    if analysis_data['effectiveness'] >= 70:
        recommendations = [
            "1. Your resume is well-targeted for this role. Keep it updated.",
            "2. Consider advanced certifications to stand out.",
            "3. Network with professionals in this field."
        ]
    else:
        recommendations = [
            "1. Focus on adding missing skills and tailoring content.",
            "2. Work on projects that demonstrate required skills.",
            "3. Practice interview questions specific to this role."
        ]
    
    for rec in recommendations:
        if y_position < 50:
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 30, f"Page 5 - Generated on: {analysis_data['timestamp']}")
            y_position = height - 80
            c.setFont("Helvetica", 11)
        
        c.drawString(60, y_position, rec)
        y_position -= 18
    
    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(width - 200, 30, "Generated by Resumify")
    
    # Save PDF
    c.save()
    
    # Read the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    return pdf_bytes

def create_download_link(content, filename, text, is_pdf=False):
    """Create a download link for content"""
    if is_pdf:
        b64 = base64.b64encode(content).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="background-color:#1DB954; color:black; padding:10px 20px; border-radius:25px; text-decoration:none; font-weight:600; display:inline-block; margin:5px;">{text}</a>'
    else:
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="background-color:#282828; color:white; padding:10px 20px; border-radius:25px; text-decoration:none; font-weight:600; display:inline-block; margin:5px; border:1px solid #333;">{text}</a>'
    return href

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Resumify",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #121212;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}
.stTextArea textarea {
    background-color: #181818;
    color: #ffffff;
    border-radius: 8px;
    border: 1px solid #333333;
    font-size: 14px;
}
.stSelectbox div[data-baseweb="select"] {
    background-color: #181818;
    color: #ffffff;
}
.stButton button {
    background-color: #1DB954;
    color: #000000;
    border-radius: 25px;
    font-weight: 600;
    padding: 12px 24px;
    font-size: 16px;
    width: 100%;
    border: none;
    transition: all 0.2s ease;
}
.stButton button:hover {
    background-color: #1ed760;
    transform: scale(1.02);
}
.analysis-card {
    background-color: #181818;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 4px solid #1DB954;
}
.role-card {
    background: linear-gradient(135deg, #1DB95420, #1DB95410);
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
}
.skill-pill {
    display: inline-block;
    background-color: #282828;
    color: #ffffff;
    padding: 6px 16px;
    border-radius: 20px;
    margin: 5px;
    font-size: 14px;
    border: 1px solid #333333;
}
.skill-pill.missing {
    background-color: #ff6b6b20;
    color: #ff6b6b;
    border: 1px solid #ff6b6b40;
}
.skill-pill.present {
    background-color: #1DB95420;
    color: #1DB954;
    border: 1px solid #1DB95440;
}
.project-item {
    background-color: #282828;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 3px solid #1DB954;
}
.tip-item {
    background-color: #282828;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 8px 0;
}
.match-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}
.match-good {
    background-color: #1DB954;
}
.match-warning {
    background-color: #ffa726;
}
.match-bad {
    background-color: #ff6b6b;
}
h1, h2, h3, h4 {
    color: #ffffff;
    font-weight: 600;
}
h1 {
    font-size: 32px;
}
h3 {
    font-size: 20px;
    margin-bottom: 15px;
}
.progress-container {
    background-color: #282828;
    border-radius: 10px;
    padding: 5px;
    margin: 15px 0;
}
.progress-fill {
    background-color: #1DB954;
    height: 10px;
    border-radius: 5px;
    transition: width 0.5s ease;
}
.scroll-message {
    background-color: #1DB954;
    color: black;
    padding: 15px;
    border-radius: 10px;
    margin: 20px 0;
    text-align: center;
    font-weight: 600;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='color:#1DB954;'>Resumify</h1>", unsafe_allow_html=True)
# Removed subtitle line

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Your Resume")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    
    # Text input
    resume_text = st.text_area("Additional Information (if not in resume)", height=250, 
                              placeholder="Add any skills, projects, or details not covered in your PDF resume...")

with col2:
    st.markdown("### Target Job")
    
    # Job role selection
    job_role_options = ["Select job role"] + list(SKILL_MAP.keys())
    selected_job_role = st.selectbox("Choose Target Role", job_role_options)
    
    # Job description
    job_desc = st.text_area("Job Description", height=150, placeholder="Paste the complete job description here...")
    
    # Additional job info
    additional_job_info = st.text_area("Additional Information (if any about job role)", height=100,
                                      placeholder="Add any additional requirements, preferences, or specific details about this job role...")

# ---------------- ANALYZE BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Analyze Resume Effectiveness"):
    if not resume_text.strip() and uploaded_file is None:
        st.warning("Please upload PDF or add additional information")
    elif selected_job_role == "Select job role":
        st.warning("Please select a job role")
    else:
        with st.spinner("Analyzing resume..."):
            # Extract text from PDF if uploaded
            if uploaded_file is not None:
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text.strip():
                    resume_text = extracted_text + "\n\n" + resume_text
            
            # Clean and predict
            cleaned = clean_text(resume_text)
            vector = tfidf.transform([cleaned])
            
            # Get predicted role
            predicted_role = clf.predict(vector)[0]
            
            # Calculate effectiveness score
            effectiveness = 0
            
            # 1. Check role match
            role_match = predicted_role.lower() == selected_job_role.lower()
            if role_match:
                effectiveness += 40
            
            # 2. Check skill overlap
            required_skills = []
            if selected_job_role in SKILL_MAP:
                for skill_list in SKILL_MAP[selected_job_role].values():
                    required_skills.extend(skill_list)
            
            resume_skills_found = 0
            present_skills = []
            missing_skills = []
            
            for skill in required_skills:
                if skill.lower() in resume_text.lower():
                    resume_skills_found += 1
                    present_skills.append(skill)
                else:
                    missing_skills.append(skill)
            
            if required_skills:
                skill_match = (resume_skills_found / len(required_skills)) * 40
                effectiveness += min(skill_match, 40)
            
            # 3. Check job description match
            desc_score_value = 0
            if job_desc.strip() or additional_job_info.strip():
                combined_job_text = job_desc + "\n" + additional_job_info
                job_vec = tfidf.transform([clean_text(combined_job_text)])
                desc_score_value = cosine_similarity(vector, job_vec)[0][0] * 20
                effectiveness += desc_score_value
            
            effectiveness = min(effectiveness, 100)
            
            # Store results for download
            analysis_data = {
                "effectiveness": effectiveness,
                "predicted_role": predicted_role,
                "target_role": selected_job_role,
                "required_skills": required_skills,
                "skills_missing": missing_skills[:12],
                "present_skills": present_skills,
                "role_match": role_match,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "desc_score": desc_score_value,
                "skill_match_percentage": (resume_skills_found / len(required_skills)) * 40 if required_skills else 0
            }
            
            # ---------------- SCROLL MESSAGE ----------------
            st.markdown("""
            <div class="scroll-message">
                Scroll down to see complete analysis
            </div>
            """, unsafe_allow_html=True)
            
            # ---------------- RESULTS ----------------
            st.markdown("---")
            
            # 1. EFFECTIVENESS SCORE
            st.markdown("""
            <div class="analysis-card">
                <h3>Resume Effectiveness Score</h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<h1 style='color:#1DB954; font-size: 48px; margin: 10px 0;'>{effectiveness:.0f}%</h1>", unsafe_allow_html=True)
            
            # Progress bar
            progress_value = max(0, min(1, effectiveness / 100))
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-fill" style="width: {progress_value*100}%"></div>
            </div>
            """, unsafe_allow_html=True)
            
            if effectiveness >= 70:
                st.markdown("""
                <div style="display: flex; align-items: center; margin-top: 10px;">
                    <span class="match-indicator match-good"></span>
                    <span style="color:#1DB954; font-weight: 500;">Strong match for this role</span>
                </div>
                """, unsafe_allow_html=True)
            elif effectiveness >= 40:
                st.markdown("""
                <div style="display: flex; align-items: center; margin-top: 10px;">
                    <span class="match-indicator match-warning"></span>
                    <span style="color:#ffa726; font-weight: 500;">Needs improvement for this role</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; align-items: center; margin-top: 10px;">
                    <span class="match-indicator match-bad"></span>
                    <span style="color:#ff6b6b; font-weight: 500;">Low match for this role</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 2. ROLE ANALYSIS
            st.markdown("""
            <div class="role-card">
                <h3>Role Analysis</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div>
                <p style="color:#b3b3b3; margin-bottom: 5px;">Predicted Role</p>
                <p style="color:#ffffff; font-size: 24px; font-weight: 600;">{predicted_role}</p>
            </div>
            <div>
                <p style="color:#b3b3b3; margin-bottom: 5px;">Target Role</p>
                <p style="color:#ffffff; font-size: 24px; font-weight: 600;">{selected_job_role}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if predicted_role.lower() != selected_job_role.lower():
                st.markdown(f"""
                <div style="grid-column: span 2; background-color: #ff6b6b15; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <p style="color:#ff6b6b; margin: 0;">
                        Resume better matches <strong>{predicted_role}</strong> roles. Consider tailoring for <strong>{selected_job_role}</strong> positions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="grid-column: span 2; background-color: #1DB95415; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <p style="color:#1DB954; margin: 0;">
                        Resume is well-targeted for <strong>{selected_job_role}</strong> roles.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # 3. SKILLS ANALYSIS
            if missing_skills:
                st.markdown("""
                <div class="analysis-card">
                    <h3>Skills to Add</h3>
                    <p style="color:#b3b3b3; margin-bottom: 15px;">Consider adding these skills to improve your resume:</p>
                """, unsafe_allow_html=True)
                
                # Show missing skills in pill format
                cols = st.columns(3)
                for idx, skill in enumerate(missing_skills[:12]):  # Show max 12 skills
                    col_idx = idx % 3
                    with cols[col_idx]:
                        st.markdown(f'<span class="skill-pill missing">{skill}</span>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # 4. PROJECT SUGGESTIONS
            st.markdown("""
            <div class="analysis-card">
                <h3>Project Suggestions</h3>
            """, unsafe_allow_html=True)
            
            if selected_job_role in ["Data Science", "Python Developer"]:
                suggestions = [
                    "Machine learning model with real-world dataset",
                    "Data visualization dashboard using Tableau or Power BI",
                    "Web application using Flask or Django framework",
                    "Automated data pipeline with Python scripts"
                ]
            elif selected_job_role in ["Web Designing"]:
                suggestions = [
                    "Responsive portfolio website with modern design",
                    "E-commerce website template with product catalog",
                    "Web application using React or Vue.js framework",
                    "Website redesign case study with before/after analysis"
                ]
            elif selected_job_role in ["HR"]:
                suggestions = [
                    "Employee onboarding process documentation",
                    "Performance management system design",
                    "HR policy compliance audit report",
                    "Employee engagement survey analysis"
                ]
            else:
                suggestions = [
                    "Portfolio showcasing your best work samples",
                    "Case study documenting a successful project",
                    "Technical documentation for a complex process",
                    "Certification in relevant tools or methodologies"
                ]
            
            for project in suggestions:
                st.markdown(f"""
                <div class="project-item">
                    <p style="margin: 0; color:#ffffff;">{project}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 5. QUICK IMPROVEMENT TIPS
            st.markdown("""
            <div class="analysis-card">
                <h3>Improvement Tips</h3>
            """, unsafe_allow_html=True)
            
            tips = [
                "Use action verbs to describe achievements",
                "Quantify results with specific numbers and metrics",
                "Keep resume length to 1-2 pages maximum",
                "Tailor content for each specific job application",
                "Highlight most relevant experience first",
                "Include relevant certifications and training"
            ]
            
            for tip in tips:
                st.markdown(f"""
                <div class="tip-item">
                    <p style="margin: 0; color:#b3b3b3;">• {tip}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- DOWNLOAD SECTION ----------------
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <h3>Download Analysis Report</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create download content for TXT version
            download_content = f"""
RESUME ANALYSIS REPORT
Generated on: {analysis_data['timestamp']}
================================================

OVERVIEW
--------
Effectiveness Score: {analysis_data['effectiveness']:.0f}%
Predicted Role: {analysis_data['predicted_role']}
Target Role: {analysis_data['target_role']}
Role Match: {'Yes' if analysis_data['role_match'] else 'No'}

SCORE BREAKDOWN
---------------
Role Match: {'40/40' if analysis_data['role_match'] else '0/40'}
Skills Match: {len(analysis_data['present_skills'])}/{len(analysis_data['required_skills'])} skills found ({analysis_data['skill_match_percentage']:.1f}/40)
Job Description Match: {analysis_data['desc_score']:.1f}/20

SKILLS ANALYSIS
---------------
Skills Present ({len(analysis_data['present_skills'])}):
{', '.join(analysis_data['present_skills'][:10])}

Skills to Add ({len(analysis_data['skills_missing'])}):
{', '.join(analysis_data['skills_missing'])}

PROJECT SUGGESTIONS
-------------------
{chr(10).join(suggestions)}

IMPROVEMENT TIPS
----------------
{chr(10).join(tips)}

RECOMMENDATIONS
---------------
1. {'Your resume is well-targeted for this role. Keep it updated.' if analysis_data['effectiveness'] >= 70 else 'Focus on adding missing skills and tailoring content.'}
2. {'Consider advanced certifications to stand out.' if analysis_data['effectiveness'] >= 70 else 'Work on projects that demonstrate required skills.'}
3. {'Network with professionals in this field.' if analysis_data['effectiveness'] >= 70 else 'Practice interview questions specific to this role.'}

================================================
Generated by Resumify
            """
            
            # Create PDF report
            pdf_bytes = create_pdf_report(analysis_data, suggestions, tips)
            
            # Display both download options side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    create_download_link(
                        pdf_bytes,
                        f"Resume_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "Download PDF Report",
                        is_pdf=True
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    create_download_link(
                        download_content,
                        f"Resume_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "Download TXT Report"
                    ),
                    unsafe_allow_html=True
                )
            
            st.markdown("<br><br>", unsafe_allow_html=True)