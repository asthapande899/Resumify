SKILL_MAP = {
    "Data Science": {
        "Core": ["Python", "Machine Learning", "Statistics", "SQL", "Data Visualization"],
        "Advanced": ["Deep Learning", "Natural Language Processing", "Big Data", "Cloud Platforms"],
        "Tools": ["Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "Tableau", "Power BI"]
    },
    "HR": {
        "Core": ["Recruitment", "Employee Relations", "Payroll Management", "HR Policies"],
        "Advanced": ["Talent Management", "Performance Management", "Compensation & Benefits", "HR Analytics"],
        "Tools": ["HRMS Software", "MS Office", "ATS", "Payroll Software"]
    },
    "Advocate": {
        "Core": ["Legal Research", "Drafting", "Litigation", "Client Counseling"],
        "Advanced": ["Contract Law", "Corporate Law", "IPR", "Arbitration"],
        "Tools": ["Legal Databases", "MS Office", "Case Management Software"]
    },
    "Arts": {
        "Core": ["Creative Design", "Visual Communication", "Art History", "Color Theory"],
        "Advanced": ["Digital Art", "Photography", "Art Education", "Exhibition Management"],
        "Tools": ["Adobe Creative Suite", "Procreate", "Canvas", "Traditional Media"]
    },
    "Web Designing": {
        "Core": ["HTML5", "CSS3", "JavaScript", "Responsive Design"],
        "Advanced": ["React", "Vue.js", "UI/UX Design", "Web Performance"],
        "Tools": ["Figma", "Adobe XD", "VS Code", "Git"]
    },
    "Java Developer": {
        "Core": ["Java", "Spring Framework", "Hibernate", "REST APIs"],
        "Advanced": ["Microservices", "Spring Boot", "Cloud Deployment", "Kafka"],
        "Tools": ["Maven", "Gradle", "IntelliJ IDEA", "Docker"]
    },
    "Python Developer": {
        "Core": ["Python", "Django", "Flask", "APIs"],
        "Advanced": ["FastAPI", "Celery", "Redis", "PostgreSQL"],
        "Tools": ["Git", "Docker", "AWS", "Postman"]
    },
    "DevOps Engineer": {
        "Core": ["Docker", "Kubernetes", "CI/CD", "Linux"],
        "Advanced": ["AWS/Azure/GCP", "Terraform", "Ansible", "Monitoring"],
        "Tools": ["Jenkins", "GitLab CI", "Prometheus", "Grafana"]
    }
}

def get_suggestions(role, resume_text):
    """
    Get skill suggestions for improvement based on role
    """
    suggestions = []
    resume_text_lower = resume_text.lower()
    
    if role in SKILL_MAP:
        role_skills = SKILL_MAP[role]
        
        # Check Core Skills
        for skill in role_skills["Core"]:
            if skill.lower() not in resume_text_lower:
                suggestions.append(f"Add core skill: {skill}")
        
        # Check Tools
        for tool in role_skills["Tools"]:
            if tool.lower() not in resume_text_lower:
                suggestions.append(f"Consider adding tool: {tool}")
    
    # General suggestions for any role
    general_checks = [
        ("quantifiable achievements", "Add quantifiable achievements (e.g., 'Improved efficiency by 30%')"),
        ("action verbs", "Use strong action verbs (e.g., 'Developed', 'Implemented', 'Managed')"),
        ("contact information", "Ensure contact info is present and clear"),
        ("education section", "Include education details with dates")
    ]
    
    for check, suggestion in general_checks:
        if check not in resume_text_lower:
            suggestions.append(suggestion)
    
    return suggestions[:10]  # Limit to top 10 suggestions

def calculate_match_score(resume_text, job_description):
    """
    Calculate match percentage between resume and job description
    """
    if not job_description or not resume_text:
        return 0
    
    # Simple keyword matching (can be enhanced)
    job_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())
    
    if not job_words:
        return 0
    
    common_words = job_words.intersection(resume_words)
    match_percentage = (len(common_words) / len(job_words)) * 100
    
    return min(match_percentage, 100)  # Cap at 100%

def extract_keywords(text, top_n=10):
    """
    Extract top keywords from text (simplified version)
    """
    from collections import Counter
    import re
    
    # Remove common words and short words
    stop_words = set(['the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'were'])
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]
    
    # Count and return top keywords
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(top_n)]