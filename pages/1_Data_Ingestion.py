import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import (generate_synthetic_jobs, generate_synthetic_candidates,
                                generate_synthetic_onet, load_jobs_dataset,
                                load_resume_dataset, load_onet_dataset)
from utils.spark_utils import get_spark, spark_process_jobs, spark_process_candidates

st.set_page_config(page_title="Data Ingestion", page_icon="📥", layout="wide")

st.title("📥 Data Ingestion & Processing")
st.markdown("Load real-world datasets or use synthetic fallback data. Data is processed using **Apache Spark**.")

# ── Spark Status ────────────────────────────────────────────────────────────
with st.expander("⚡ Spark Session Status", expanded=False):
    if "spark" not in st.session_state:
        with st.spinner("Initializing Spark..."):
            st.session_state.spark = get_spark()
    if st.session_state.spark:
        st.success("✅ Spark session active")
        st.code(f"App: {st.session_state.spark.sparkContext.appName}\n"
                f"Version: {st.session_state.spark.version}")
    else:
        st.warning("⚠️ Spark not available — using Pandas fallback (fully functional)")

st.divider()

# ── Dataset Sources Info ────────────────────────────────────────────────────
with st.expander("📖 Where to download real datasets", expanded=False):
    st.markdown("""
    | Dataset | Source | Notes |
    |---|---|---|
    | **LinkedIn Job Postings** | [Kaggle - arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Download `job_postings.csv` |
    | **Resume Dataset** | [Kaggle - snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | Download `Resume.csv` |
    | **O\\*NET Skills** | [onetcenter.org/database.html](https://www.onetcenter.org/database.html) | Download `Skills.txt` (tab-separated, rename to .csv) |
    
    Upload any of these below. **Synthetic data is used automatically as fallback.**
    """)

st.divider()

# ── Upload Section ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💼 Job Postings")
    jobs_file = st.file_uploader("Upload LinkedIn Jobs CSV", type=["csv"], key="jobs_upload")
    if jobs_file:
        try:
            df_jobs = load_jobs_dataset(jobs_file)
            st.session_state["jobs_df"] = df_jobs
            st.success(f"✅ Loaded {len(df_jobs):,} job postings")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        if "jobs_df" not in st.session_state:
            st.session_state["jobs_df"] = generate_synthetic_jobs(500)
        st.info("🔄 Using synthetic job postings (500 records)")

with col2:
    st.subheader("👤 Candidates / Resumes")
    resume_file = st.file_uploader("Upload Resume CSV", type=["csv"], key="resume_upload")
    if resume_file:
        try:
            df_cands = load_resume_dataset(resume_file)
            st.session_state["candidates_df"] = df_cands
            st.success(f"✅ Loaded {len(df_cands):,} candidate records")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        if "candidates_df" not in st.session_state:
            st.session_state["candidates_df"] = generate_synthetic_candidates(400)
        st.info("🔄 Using synthetic candidate data (400 records)")

with col3:
    st.subheader("🎯 O*NET Skills")
    onet_file = st.file_uploader("Upload O*NET Skills CSV", type=["csv"], key="onet_upload")
    if onet_file:
        try:
            df_onet = load_onet_dataset(onet_file)
            st.session_state["onet_df"] = df_onet
            st.success(f"✅ Loaded {len(df_onet):,} skill-occupation records")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        if "onet_df" not in st.session_state:
            st.session_state["onet_df"] = generate_synthetic_onet()
        st.info("🔄 Using synthetic O*NET data")

st.divider()

# ── Spark Processing ────────────────────────────────────────────────────────
st.subheader("⚡ Run Spark Processing")
st.markdown("Process the loaded datasets to extract skill demand and hiring patterns.")

if st.button("🚀 Process Datasets with Spark", type="primary"):
    spark = st.session_state.get("spark")
    jobs_df = st.session_state.get("jobs_df", pd.DataFrame())
    candidates_df = st.session_state.get("candidates_df", pd.DataFrame())

    with st.spinner("Processing job postings..."):
        skill_demand = spark_process_jobs(jobs_df, spark)
        st.session_state["skill_demand_df"] = skill_demand

    with st.spinner("Processing candidate data..."):
        hiring_patterns = spark_process_candidates(candidates_df, spark)
        st.session_state["hiring_patterns_df"] = hiring_patterns

    st.success("✅ Processing complete!")

st.divider()

# ── Data Previews ───────────────────────────────────────────────────────────
st.subheader("🔍 Data Preview")

tab1, tab2, tab3, tab4 = st.tabs(["Job Postings", "Candidates", "O*NET Skills", "Processed Output"])

with tab1:
    df = st.session_state.get("jobs_df", pd.DataFrame())
    st.metric("Total Records", f"{len(df):,}")
    if not df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Industries", df["industry"].nunique() if "industry" in df.columns else "—")
        with col_b:
            st.metric("Companies", df["company"].nunique() if "company" in df.columns else "—")
        st.dataframe(df.head(20), use_container_width=True)

with tab2:
    df = st.session_state.get("candidates_df", pd.DataFrame())
    st.metric("Total Records", f"{len(df):,}")
    if not df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            hired = df["hired"].sum() if "hired" in df.columns else 0
            st.metric("Hired", f"{int(hired):,}")
        with col_b:
            rate = df["hired"].mean() * 100 if "hired" in df.columns else 0
            st.metric("Hire Rate", f"{rate:.1f}%")
        st.dataframe(df.head(20), use_container_width=True)

with tab3:
    df = st.session_state.get("onet_df", pd.DataFrame())
    st.metric("Total Records", f"{len(df):,}")
    if not df.empty:
        st.dataframe(df.head(20), use_container_width=True)

with tab4:
    sd = st.session_state.get("skill_demand_df")
    hp = st.session_state.get("hiring_patterns_df")
    if sd is not None:
        st.markdown("**Skill Demand (from Spark)**")
        st.dataframe(sd.head(20), use_container_width=True)
    if hp is not None:
        st.markdown("**Hiring Patterns (from Spark)**")
        st.dataframe(hp.head(20), use_container_width=True)
    if sd is None and hp is None:
        st.info("Run Spark processing above to see output here.")
