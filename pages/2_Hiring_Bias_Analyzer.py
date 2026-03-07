import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import generate_synthetic_candidates

st.set_page_config(page_title="Hiring Bias Analyzer", page_icon="⚖️", layout="wide")

st.title("⚖️ Hiring Bias Analyzer")
st.markdown("Detect statistically significant hiring disparities across **gender, race, education, and industry**.")


def get_candidates():
    df = st.session_state.get("candidates_df")
    if df is None or df.empty:
        df = generate_synthetic_candidates(400)
        st.session_state["candidates_df"] = df
    return df


def compute_bias_score(group_rate, overall_rate):
    """Adverse Impact Ratio — below 0.8 indicates potential bias (4/5ths rule)."""
    if overall_rate == 0:
        return 1.0
    return round(group_rate / overall_rate, 3)


df = get_candidates()

# Ensure required columns
for col in ["gender", "race", "education", "industry", "hired", "applied_role"]:
    if col not in df.columns:
        df[col] = "Unknown"
df["hired"] = pd.to_numeric(df["hired"], errors="coerce").fillna(0).astype(int)

# ── Filters ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")
industries = ["All"] + sorted(df["industry"].dropna().unique().tolist())
sel_industry = st.sidebar.selectbox("Industry", industries)
years = df["year"].dropna().unique() if "year" in df.columns else []
if len(years) > 0:
    year_range = st.sidebar.slider("Year Range",
                                    int(min(years)), int(max(years)),
                                    (int(min(years)), int(max(years))))
    df = df[df["year"].between(year_range[0], year_range[1])]
if sel_industry != "All":
    df = df[df["industry"] == sel_industry]

total = len(df)
hired_total = df["hired"].sum()
overall_rate = df["hired"].mean() if total > 0 else 0

# ── KPI Row ──────────────────────────────────────────────────────────────────
st.subheader("📊 Overview Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Candidates", f"{total:,}")
c2.metric("Total Hired", f"{int(hired_total):,}")
c3.metric("Overall Hire Rate", f"{overall_rate*100:.1f}%")
bias_groups = 0

st.divider()

# ── Bias by Gender ────────────────────────────────────────────────────────────
st.subheader("👥 Bias Analysis by Gender")
gender_stats = df.groupby("gender").agg(
    total=("hired", "count"),
    hired=("hired", "sum")
).reset_index()
gender_stats["hire_rate"] = gender_stats["hired"] / gender_stats["total"]
gender_stats["adverse_impact_ratio"] = gender_stats["hire_rate"].apply(
    lambda r: compute_bias_score(r, overall_rate))
gender_stats["bias_flag"] = gender_stats["adverse_impact_ratio"].apply(
    lambda x: "🚨 Potential Bias" if x < 0.8 else "✅ Within Threshold")
bias_groups += (gender_stats["adverse_impact_ratio"] < 0.8).sum()

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(gender_stats, x="gender", y="hire_rate", color="bias_flag",
                 color_discrete_map={"🚨 Potential Bias": "#ef4444", "✅ Within Threshold": "#22c55e"},
                 title="Hire Rate by Gender", labels={"hire_rate": "Hire Rate"},
                 text=gender_stats["hire_rate"].apply(lambda x: f"{x:.1%}"))
    fig.add_hline(y=overall_rate * 0.8, line_dash="dash", line_color="orange",
                  annotation_text="4/5ths Rule Threshold")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.dataframe(gender_stats[["gender", "total", "hired", "hire_rate",
                                "adverse_impact_ratio", "bias_flag"]]
                 .rename(columns={"hire_rate": "Hire Rate", "adverse_impact_ratio": "AIR"}),
                 use_container_width=True)

st.divider()

# ── Bias by Race ──────────────────────────────────────────────────────────────
st.subheader("🏷️ Bias Analysis by Race/Ethnicity")
race_stats = df.groupby("race").agg(
    total=("hired", "count"),
    hired=("hired", "sum")
).reset_index()
race_stats["hire_rate"] = race_stats["hired"] / race_stats["total"]
race_stats["adverse_impact_ratio"] = race_stats["hire_rate"].apply(
    lambda r: compute_bias_score(r, overall_rate))
race_stats["bias_flag"] = race_stats["adverse_impact_ratio"].apply(
    lambda x: "🚨 Potential Bias" if x < 0.8 else "✅ Within Threshold")
bias_groups += (race_stats["adverse_impact_ratio"] < 0.8).sum()

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(race_stats, x="hire_rate", y="race", orientation="h",
                 color="bias_flag",
                 color_discrete_map={"🚨 Potential Bias": "#ef4444", "✅ Within Threshold": "#22c55e"},
                 title="Hire Rate by Race/Ethnicity",
                 text=race_stats["hire_rate"].apply(lambda x: f"{x:.1%}"))
    fig.add_vline(x=overall_rate * 0.8, line_dash="dash", line_color="orange",
                  annotation_text="4/5ths Threshold")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.dataframe(race_stats[["race", "total", "hired", "hire_rate",
                              "adverse_impact_ratio", "bias_flag"]]
                 .rename(columns={"hire_rate": "Hire Rate", "adverse_impact_ratio": "AIR"}),
                 use_container_width=True)

st.divider()

# ── Bias by Education ─────────────────────────────────────────────────────────
st.subheader("🎓 Bias Analysis by Education Level")
edu_stats = df.groupby("education").agg(
    total=("hired", "count"),
    hired=("hired", "sum")
).reset_index()
edu_stats["hire_rate"] = edu_stats["hired"] / edu_stats["total"]
edu_order = ["High School", "Associate's", "Bachelor's", "Master's", "PhD"]
edu_stats["edu_order"] = edu_stats["education"].apply(
    lambda x: edu_order.index(x) if x in edu_order else 99)
edu_stats = edu_stats.sort_values("edu_order")

fig = px.line(edu_stats, x="education", y="hire_rate", markers=True,
              title="Hire Rate by Education Level",
              labels={"hire_rate": "Hire Rate", "education": "Education"},
              text=edu_stats["hire_rate"].apply(lambda x: f"{x:.1%}"))
fig.update_traces(textposition="top center", line_color="#6366f1", marker_size=10)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Heatmap: Industry x Gender ────────────────────────────────────────────────
st.subheader("🔥 Hire Rate Heatmap: Industry × Gender")
heat_df = df.groupby(["industry", "gender"])["hired"].mean().reset_index()
heat_pivot = heat_df.pivot(index="industry", columns="gender", values="hired").fillna(0)
fig = px.imshow(heat_pivot, color_continuous_scale="RdYlGn",
                title="Hire Rate by Industry and Gender",
                labels={"color": "Hire Rate"},
                aspect="auto")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Summary ───────────────────────────────────────────────────────────────────
c4.metric("Biased Group Count", f"{int(bias_groups)}", delta="groups below 4/5ths rule",
          delta_color="inverse")

st.subheader("📋 Bias Summary")
st.info("""
**Methodology:** Adverse Impact Ratio (AIR) — the 4/5ths (80%) rule from EEOC guidelines.
- **AIR < 0.8** → Potential discriminatory impact (flagged 🚨)
- **AIR ≥ 0.8** → Within acceptable threshold (✅)

This is a statistical indicator — not legal proof of discrimination. Use alongside qualitative analysis.
""")
