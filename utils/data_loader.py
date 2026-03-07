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

def load_jobs_dataset(path: str) -> pd.DataFrame:
    """Load LinkedIn job postings CSV."""
    df = pd.read_csv(path)
    # Normalize common column name variants
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}
    for standard, variants in {
        "title": ["job_title", "jobtitle", "position", "title"],
        "company": ["company_name", "employer", "organization"],
        "industry": ["industry_name", "sector", "field"],
        "skills_required": ["skills", "required_skills", "skill_abr", "normalized_skills"],
        "location": ["job_location", "city", "state"],
        "year": ["posting_year", "date_posted", "listed_time"],
    }.items():
        for v in variants:
            if v in col_lower and standard not in df.columns:
                rename_map[col_lower[v]] = standard
    df.rename(columns=rename_map, inplace=True)

    if "year" not in df.columns:
        df["year"] = 2023
    else:
        df["year"] = pd.to_datetime(df["year"], errors="coerce").dt.year.fillna(2023).astype(int)

    for col in ["title", "company", "industry", "skills_required", "location"]:
        if col not in df.columns:
            df[col] = "Unknown"

    return df[["title", "company", "industry", "skills_required", "location", "year"]].dropna(subset=["title"])


def load_resume_dataset(path: str) -> pd.DataFrame:
    """Load resume dataset CSV."""
    df = pd.read_csv(path)
    col_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    for standard, variants in {
        "applied_role": ["category", "job_category", "role", "position"],
        "skills": ["resume", "resume_str", "skills", "text"],
    }.items():
        for v in variants:
            if v in col_lower and standard not in df.columns:
                rename_map[col_lower[v]] = standard
    df.rename(columns=rename_map, inplace=True)

    for col in ["applied_role", "skills"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["candidate_id"] = [f"CAND_{i+1:04d}" for i in range(len(df))]
    df["hired"] = np.random.randint(0, 2, len(df))
    df["industry"] = df.get("industry", "Unknown")
    df["year"] = np.random.randint(2018, 2025, len(df))
    return df[["candidate_id", "applied_role", "skills", "industry", "hired", "year"]]


def load_onet_dataset(path: str) -> pd.DataFrame:
    """Load O*NET Skills or Occupation Data CSV."""
    df = pd.read_csv(path)
    col_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    for standard, variants in {
        "occupation": ["title", "occupation_title", "o*net-soc title", "o*net title"],
        "skill": ["element name", "skill_name", "skill", "element_name"],
        "importance": ["data value", "importance_score", "value", "data_value"],
    }.items():
        for v in variants:
            if v in col_lower and standard not in df.columns:
                rename_map[col_lower[v]] = standard
    df.rename(columns=rename_map, inplace=True)

    for col in ["occupation", "skill", "importance"]:
        if col not in df.columns:
            df[col] = "Unknown"

    return df[["occupation", "skill", "importance"]].dropna()
