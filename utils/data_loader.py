import pandas as pd
import numpy as np
import os
import random

# ── Synthetic fallback generators ──────────────────────────────────────────

INDUSTRIES = ["Technology", "Healthcare", "Finance", "Education", "Manufacturing",
              "Retail", "Legal", "Marketing", "Engineering", "Consulting"]

ROLES = ["Software Engineer", "Data Analyst", "Product Manager", "HR Manager",
         "Financial Analyst", "Marketing Specialist", "Operations Manager",
         "Data Scientist", "DevOps Engineer", "Business Analyst", "UX Designer",
         "Recruiter", "Sales Manager", "Research Scientist", "Consultant"]

SKILLS = ["Python", "SQL", "Machine Learning", "Java", "JavaScript", "React",
          "Data Analysis", "Project Management", "Communication", "Leadership",
          "Excel", "Tableau", "AWS", "Docker", "Kubernetes", "NLP",
          "Deep Learning", "R", "Scala", "Spark", "Power BI", "Git",
          "Agile", "Scrum", "TensorFlow", "PyTorch", "C++", "Go"]

EDUCATION = ["Bachelor's", "Master's", "PhD", "Associate's", "High School"]
GENDERS = ["Male", "Female", "Non-binary"]
RACES = ["White", "Black or African American", "Hispanic or Latino",
         "Asian", "Native American", "Two or More Races"]
COMPANIES = ["TechCorp", "HealthPlus", "FinanceHub", "EduLearn", "ManuFactory",
             "RetailGiant", "LegalEagle", "MarketPro", "EngiSolve", "ConsultCo",
             "DataDriven", "CloudBase", "InnovateTech", "GlobalFinance", "MedSolutions"]


def generate_synthetic_jobs(n=500):
    random.seed(42)
    np.random.seed(42)
    rows = []
    for i in range(n):
        industry = random.choice(INDUSTRIES)
        role = random.choice(ROLES)
        skills = random.sample(SKILLS, k=random.randint(3, 7))
        year = random.randint(2018, 2024)
        rows.append({
            "job_id": f"JOB_{i+1:04d}",
            "title": role,
            "company": random.choice(COMPANIES),
            "industry": industry,
            "skills_required": ", ".join(skills),
            "min_experience_years": random.randint(0, 10),
            "education_required": random.choice(EDUCATION),
            "salary_min": random.randint(40000, 80000),
            "salary_max": random.randint(80000, 180000),
            "year": year,
            "location": random.choice(["New York", "San Francisco", "Chicago",
                                        "Austin", "Seattle", "Boston", "Remote"])
        })
    return pd.DataFrame(rows)


def generate_synthetic_candidates(n=400):
    random.seed(7)
    np.random.seed(7)
    rows = []
    for i in range(n):
        skills = random.sample(SKILLS, k=random.randint(3, 8))
        hired = random.random()
        # Introduce realistic bias: lower hire rate for certain groups
        gender = random.choice(GENDERS)
        race = random.choice(RACES)
        bias_factor = 1.0
        if gender == "Female":
            bias_factor = 0.75
        if race in ["Black or African American", "Hispanic or Latino"]:
            bias_factor *= 0.70
        hired = 1 if random.random() < (0.65 * bias_factor) else 0

        rows.append({
            "candidate_id": f"CAND_{i+1:04d}",
            "name": f"Candidate_{i+1}",
            "gender": gender,
            "race": race,
            "education": random.choice(EDUCATION),
            "years_experience": random.randint(0, 20),
            "skills": ", ".join(skills),
            "applied_role": random.choice(ROLES),
            "industry": random.choice(INDUSTRIES),
            "hired": hired,
            "salary_offered": random.randint(50000, 160000) if hired else None,
            "year": random.randint(2018, 2024)
        })
    return pd.DataFrame(rows)


def generate_synthetic_onet():
    rows = []
    for skill in SKILLS:
        for role in random.sample(ROLES, k=random.randint(3, 8)):
            rows.append({
                "occupation": role,
                "skill": skill,
                "importance": round(random.uniform(1.0, 5.0), 2),
                "level": round(random.uniform(1.0, 7.0), 2)
            })
    return pd.DataFrame(rows)


# ── Real dataset loaders ────────────────────────────────────────────────────

def extract_skills_from_text(text: str) -> str:
    """Extract skill keywords from free-text job descriptions or resumes."""
    SKILL_KEYWORDS = [
        "python", "sql", "java", "javascript", "react", "node", "aws", "azure", "gcp",
        "docker", "kubernetes", "machine learning", "deep learning", "nlp", "tensorflow",
        "pytorch", "spark", "hadoop", "scala", "r", "tableau", "power bi", "excel",
        "git", "agile", "scrum", "project management", "data analysis", "communication",
        "leadership", "marketing", "sales", "finance", "accounting", "recruiting",
        "photoshop", "illustrator", "indesign", "html", "css", "c++", "c#", "go",
        "typescript", "mongodb", "postgresql", "mysql", "redis", "kafka", "airflow",
        "data visualization", "statistics", "research", "writing", "design", "testing",
        "devops", "linux", "networking", "security", "compliance", "training",
        "customer service", "operations", "strategy", "consulting", "problem solving",
        "sharepoint", "microsoft office", "adobe", "figma", "sketch", "jira", "confluence"
    ]
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    found = [skill.title() for skill in SKILL_KEYWORDS if skill in text_lower]
    return ", ".join(found) if found else "General Skills"


def load_jobs_dataset(path) -> pd.DataFrame:
    """Load LinkedIn job postings CSV — actual columns:
       job_id, company_name, title, description, skills_desc, listed_time, location, etc.
    """
    df = pd.read_csv(path, low_memory=False)

    # Map actual column names to standard names
    rename_map = {}
    col_lower = {c.lower().strip(): c for c in df.columns}

    for standard, variants in {
        "title":    ["title", "job_title", "jobtitle", "position"],
        "company":  ["company_name", "company", "employer"],
        "location": ["location", "job_location", "city"],
        "year":     ["listed_time", "original_listed_time", "posting_date", "date_posted"],
        "skills_raw": ["skills_desc", "skills", "required_skills", "description"],
    }.items():
        for v in variants:
            if v in col_lower and standard not in rename_map.values():
                rename_map[col_lower[v]] = standard
                break

    df.rename(columns=rename_map, inplace=True)

    # Parse year from epoch milliseconds (LinkedIn uses ms timestamps)
    if "year" in df.columns:
        col = pd.to_numeric(df["year"], errors="coerce")
        # LinkedIn timestamps are in milliseconds
        if col.dropna().median() > 1e10:
            df["year"] = pd.to_datetime(col, unit="ms", errors="coerce").dt.year
        else:
            df["year"] = pd.to_datetime(col, unit="s", errors="coerce").dt.year
        df["year"] = df["year"].fillna(2023).astype(int)
    else:
        df["year"] = 2023

    # Extract skills from skills_desc free text
    if "skills_raw" in df.columns:
        df["skills_required"] = df["skills_raw"].apply(extract_skills_from_text)
    else:
        df["skills_required"] = "General Skills"

    # Industry — not in this dataset, derive from title
    if "industry" not in df.columns:
        df["industry"] = df.get("title", pd.Series(["Unknown"] * len(df))).apply(infer_industry_from_title)

    for col in ["title", "company", "location"]:
        if col not in df.columns:
            df[col] = "Unknown"

    return df[["title", "company", "industry", "skills_required", "location", "year"]].dropna(subset=["title"])


def infer_industry_from_title(title: str) -> str:
    """Infer industry from job title keywords."""
    if not isinstance(title, str):
        return "Other"
    t = title.lower()
    if any(k in t for k in ["software", "engineer", "developer", "data", "ml", "ai", "cloud", "devops"]):
        return "Technology"
    if any(k in t for k in ["nurse", "doctor", "medical", "health", "clinical", "therapist", "physician"]):
        return "Healthcare"
    if any(k in t for k in ["finance", "accountant", "analyst", "banker", "investment", "audit"]):
        return "Finance"
    if any(k in t for k in ["teacher", "professor", "education", "tutor", "academic", "instructor"]):
        return "Education"
    if any(k in t for k in ["marketing", "brand", "social media", "seo", "content", "advertising"]):
        return "Marketing"
    if any(k in t for k in ["hr", "human resource", "recruiter", "talent", "people ops"]):
        return "Human Resources"
    if any(k in t for k in ["sales", "account executive", "business development"]):
        return "Sales"
    if any(k in t for k in ["legal", "attorney", "lawyer", "counsel", "paralegal"]):
        return "Legal"
    if any(k in t for k in ["operations", "supply chain", "logistics", "manufacturing"]):
        return "Operations"
    return "Other"


def load_resume_dataset(path) -> pd.DataFrame:
    """Load Resume.csv — actual columns: ID, Resume_str, Resume_html, Category"""
    df = pd.read_csv(path)

    # Category → applied_role, Resume_str → extract skills
    rename_map = {}
    col_lower = {c.lower().strip(): c for c in df.columns}

    if "category" in col_lower:
        rename_map[col_lower["category"]] = "applied_role"
    if "resume_str" in col_lower:
        rename_map[col_lower["resume_str"]] = "resume_text"
    elif "resume" in col_lower:
        rename_map[col_lower["resume"]] = "resume_text"

    df.rename(columns=rename_map, inplace=True)

    if "applied_role" not in df.columns:
        df["applied_role"] = "Unknown"

    # Extract skills from resume text
    if "resume_text" in df.columns:
        df["skills"] = df["resume_text"].apply(extract_skills_from_text)
    else:
        df["skills"] = "General Skills"

    df["candidate_id"] = [f"CAND_{i+1:04d}" for i in range(len(df))]
    df["industry"] = df["applied_role"].apply(infer_industry_from_title)

    # Simulate hired flag and year since dataset doesn't have them
    np.random.seed(42)
    df["hired"] = np.random.randint(0, 2, len(df))
    df["year"] = np.random.randint(2018, 2025, len(df))

    return df[["candidate_id", "applied_role", "skills", "industry", "hired", "year"]]


def load_onet_dataset(path) -> pd.DataFrame:
    """Load O*NET Skills.csv — tab-separated with columns:
       O*NET-SOC Code, Title, Element ID, Element Name, Scale ID, Scale Name, Data Value, ...
    """
    # Try tab-separated first (O*NET default), fall back to comma
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if len(df.columns) < 3:
            df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)

    col_lower = {c.lower().strip(): c for c in df.columns}
    rename_map = {}

    for standard, variants in {
        "occupation": ["title", "o*net-soc title", "occupation_title", "occupation"],
        "skill":      ["element name", "element_name", "skill_name", "skill"],
        "importance": ["data value", "data_value", "importance", "value"],
        "scale":      ["scale id", "scale_id"],
    }.items():
        for v in variants:
            if v in col_lower and standard not in rename_map.values():
                rename_map[col_lower[v]] = standard
                break

    df.rename(columns=rename_map, inplace=True)

    # O*NET has both Importance (IM) and Level (LV) rows — keep only Importance
    if "scale" in df.columns:
        df = df[df["scale"] == "IM"]

    for col in ["occupation", "skill", "importance"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0)
    return df[["occupation", "skill", "importance"]].dropna()
