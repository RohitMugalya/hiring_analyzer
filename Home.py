import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Hiring Bias & Skill Flow Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Large-Scale Hiring Bias & Skill Flow Analyzer")
st.markdown("### Powered by Apache Spark · Neo4j · Streamlit")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This platform analyzes large-scale employment data to:
    - **Detect hiring bias** across gender, race, education, and industry using the EEOC 4/5ths rule
    - **Track skill demand trends** across industries over time
    - **Visualize career pathways** and skill flows using interactive Sankey diagrams
    - **Explore knowledge graphs** of candidates, skills, companies, and job roles

    ---
    
    ### 🗺️ Navigation Guide
    
    | Page | Description |
    |---|---|
    | 📥 **Data Ingestion** | Upload real datasets or use synthetic fallback. Run Spark processing. |
    | ⚖️ **Hiring Bias Analyzer** | Detect bias patterns across demographic groups using AIR methodology. |
    | 🔄 **Skill Flow Analyzer** | Track skill demand trends, emerging skills, and industry flows. |
    | 🕸️ **Knowledge Graph** | Explore the Neo4j graph of candidates, jobs, skills, and companies. |
    
    ---
    
    ### 📦 Supported Real-World Datasets
    - **LinkedIn Job Postings** — [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
    - **Resume Dataset** — [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
    - **O*NET Skills Database** — [onetcenter.org](https://www.onetcenter.org/database.html)
    
    Upload datasets in the **Data Ingestion** page. All pages work with **synthetic data** automatically if no datasets are uploaded.
    """)

with col2:
    st.markdown("### ⚙️ System Status")
    
    # Python packages
    try:
        import pyspark
        st.success(f"✅ PySpark {pyspark.__version__}")
    except:
        st.error("❌ PySpark not found")

    try:
        import neo4j
        st.success(f"✅ Neo4j Driver {neo4j.__version__}")
    except:
        st.error("❌ Neo4j driver not found")

    # Neo4j connectivity
    try:
        from utils.graph_utils import get_driver
        driver = get_driver()
        if driver:
            st.success("✅ Neo4j Connected")
        else:
            st.warning("⚠️ Neo4j Offline")
    except:
        st.warning("⚠️ Neo4j check failed")

    try:
        import plotly
        st.success(f"✅ Plotly {plotly.__version__}")
    except:
        st.error("❌ Plotly not found")

    try:
        import sklearn
        st.success(f"✅ Scikit-learn {sklearn.__version__}")
    except:
        st.error("❌ Scikit-learn not found")

    st.divider()
    st.markdown("### 📊 Session Data")
    jobs = st.session_state.get("jobs_df")
    cands = st.session_state.get("candidates_df")
    onet = st.session_state.get("onet_df")
    st.write(f"Jobs loaded: **{len(jobs):,}**" if jobs is not None else "Jobs: *not loaded*")
    st.write(f"Candidates loaded: **{len(cands):,}**" if cands is not None else "Candidates: *not loaded*")
    st.write(f"O*NET records: **{len(onet):,}**" if onet is not None else "O*NET: *not loaded*")

st.divider()
st.caption("Built with Streamlit · Apache Spark · Neo4j · Plotly | Data sources: LinkedIn, O*NET, Kaggle Resume Dataset")
