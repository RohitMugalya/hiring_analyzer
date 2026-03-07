import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import generate_synthetic_jobs, generate_synthetic_onet
from utils.spark_utils import get_spark, spark_process_jobs

st.set_page_config(page_title="Skill Flow Analyzer", page_icon="🔄", layout="wide")

st.title("🔄 Skill Flow Analyzer")
st.markdown("Track how **skill demand evolves** across industries and years. Identify emerging, declining, and cross-industry skills.")


def get_jobs():
    df = st.session_state.get("jobs_df")
    if df is None or df.empty:
        df = generate_synthetic_jobs(500)
        st.session_state["jobs_df"] = df
    return df


def get_onet():
    df = st.session_state.get("onet_df")
    if df is None or df.empty:
        df = generate_synthetic_onet()
        st.session_state["onet_df"] = df
    return df


def get_skill_demand():
    df = st.session_state.get("skill_demand_df")
    if df is None:
        jobs_df = get_jobs()
        spark = st.session_state.get("spark")
        df = spark_process_jobs(jobs_df, spark)
        st.session_state["skill_demand_df"] = df
    return df


jobs_df = get_jobs()
onet_df = get_onet()
skill_demand = get_skill_demand()

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")
all_industries = sorted(jobs_df["industry"].dropna().unique().tolist()) if "industry" in jobs_df.columns else []
sel_industries = st.sidebar.multiselect("Industries", all_industries, default=all_industries[:4])

all_skills = sorted(skill_demand["skill"].dropna().unique().tolist()) if "skill" in skill_demand.columns else []
top_skills = skill_demand.groupby("skill")["demand_count"].sum().nlargest(10).index.tolist()
sel_skills = st.sidebar.multiselect("Skills to Track", all_skills, default=top_skills[:6])

years = sorted(jobs_df["year"].dropna().unique().tolist()) if "year" in jobs_df.columns else [2023]
if len(years) >= 2:
    year_range = st.sidebar.slider("Year Range", int(min(years)), int(max(years)),
                                    (int(min(years)), int(max(years))))
else:
    year_range = (int(years[0]), int(years[0]))

st.divider()

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.subheader("📊 Skill Demand Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Job Postings", f"{len(jobs_df):,}")
c2.metric("Unique Skills", f"{skill_demand['skill'].nunique() if 'skill' in skill_demand.columns else 0:,}")
c3.metric("Industries Covered", f"{jobs_df['industry'].nunique() if 'industry' in jobs_df.columns else 0:,}")
c4.metric("Years of Data", f"{len(years)}")

st.divider()

# ── Top Skills Bar Chart ──────────────────────────────────────────────────────
st.subheader("🏆 Top 15 Most Demanded Skills")
top15 = skill_demand.groupby("skill")["demand_count"].sum().nlargest(15).reset_index()
fig = px.bar(top15, x="demand_count", y="skill", orientation="h",
             color="demand_count", color_continuous_scale="Blues",
             title="Overall Skill Demand",
             labels={"demand_count": "Job Postings", "skill": "Skill"})
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Skill Demand Over Time ────────────────────────────────────────────────────
st.subheader("📈 Skill Demand Trends Over Time")
if sel_skills and "year" in skill_demand.columns:
    trend_df = skill_demand[
        (skill_demand["skill"].isin(sel_skills)) &
        (skill_demand["year"].between(year_range[0], year_range[1]))
    ].groupby(["year", "skill"])["demand_count"].sum().reset_index()

    if not trend_df.empty:
        fig = px.line(trend_df, x="year", y="demand_count", color="skill",
                      markers=True, title="Skill Demand Over Time",
                      labels={"demand_count": "Demand Count", "year": "Year"})
        fig.update_traces(line_width=2, marker_size=7)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected filters.")
else:
    st.info("Select skills in the sidebar to see trends.")

st.divider()

# ── Skill Flow Across Industries (Sankey) ─────────────────────────────────────
st.subheader("🌊 Skill Flow Across Industries")
if sel_industries and sel_skills:
    flow_df = skill_demand[
        (skill_demand["industry"].isin(sel_industries)) &
        (skill_demand["skill"].isin(sel_skills[:8]))
    ].groupby(["industry", "skill"])["demand_count"].sum().reset_index()

    if not flow_df.empty:
        industries_list = flow_df["industry"].unique().tolist()
        skills_list = flow_df["skill"].unique().tolist()
        all_nodes = industries_list + skills_list
        node_idx = {n: i for i, n in enumerate(all_nodes)}

        sources = [node_idx[r["industry"]] for _, r in flow_df.iterrows()]
        targets = [node_idx[r["skill"]] for _, r in flow_df.iterrows()]
        values = flow_df["demand_count"].tolist()

        colors_ind = ["#6366f1"] * len(industries_list)
        colors_sk = ["#22c55e"] * len(skills_list)

        fig = go.Figure(go.Sankey(
            node=dict(label=all_nodes, color=colors_ind + colors_sk,
                      pad=15, thickness=20),
            link=dict(source=sources, target=targets, value=values,
                      color="rgba(99,102,241,0.3)")
        ))
        fig.update_layout(title="Skill Flow: Industries → Skills", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No flow data for selected filters.")

st.divider()

# ── Industry × Skill Heatmap ──────────────────────────────────────────────────
st.subheader("🔥 Industry × Skill Demand Heatmap")
heat_df = skill_demand.copy()
if sel_industries:
    heat_df = heat_df[heat_df["industry"].isin(sel_industries)]
top_sk = heat_df.groupby("skill")["demand_count"].sum().nlargest(12).index.tolist()
heat_df = heat_df[heat_df["skill"].isin(top_sk)]

if not heat_df.empty:
    pivot = heat_df.groupby(["industry", "skill"])["demand_count"].sum().reset_index()
    pivot = pivot.pivot(index="industry", columns="skill", values="demand_count").fillna(0)
    fig = px.imshow(pivot, color_continuous_scale="Viridis",
                    title="Skill Demand Intensity by Industry",
                    labels={"color": "Demand Count"}, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── O*NET Skill Importance ────────────────────────────────────────────────────
st.subheader("🎯 O*NET: Skill Importance by Occupation")
if not onet_df.empty and "occupation" in onet_df.columns:
    occupations = sorted(onet_df["occupation"].dropna().unique().tolist())
    sel_occ = st.selectbox("Select Occupation", occupations[:20])
    occ_skills = onet_df[onet_df["occupation"] == sel_occ].sort_values("importance", ascending=False).head(15)
    if not occ_skills.empty:
        fig = px.bar(occ_skills, x="importance", y="skill", orientation="h",
                     color="importance", color_continuous_scale="Oranges",
                     title=f"Top Skills for: {sel_occ}",
                     labels={"importance": "Importance Score", "skill": "Skill"})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Emerging vs Declining Skills ─────────────────────────────────────────────
st.subheader("📉 Emerging vs Declining Skills")
if "year" in skill_demand.columns and skill_demand["year"].nunique() >= 2:
    all_years = sorted(skill_demand["year"].dropna().unique())
    first_yr, last_yr = all_years[0], all_years[-1]

    early = skill_demand[skill_demand["year"] == first_yr].groupby("skill")["demand_count"].sum()
    late = skill_demand[skill_demand["year"] == last_yr].groupby("skill")["demand_count"].sum()

    trend = pd.DataFrame({"early": early, "late": late}).fillna(0)
    trend["change"] = trend["late"] - trend["early"]
    trend["pct_change"] = ((trend["late"] - trend["early"]) / (trend["early"] + 1)) * 100
    trend = trend.reset_index()

    col1, col2 = st.columns(2)
    with col1:
        emerging = trend.nlargest(8, "pct_change")
        fig = px.bar(emerging, x="pct_change", y="skill", orientation="h",
                     color_discrete_sequence=["#22c55e"],
                     title=f"🚀 Fastest Growing Skills ({first_yr}→{last_yr})",
                     labels={"pct_change": "% Growth", "skill": "Skill"})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        declining = trend.nsmallest(8, "pct_change")
        fig = px.bar(declining, x="pct_change", y="skill", orientation="h",
                     color_discrete_sequence=["#ef4444"],
                     title=f"📉 Declining Skills ({first_yr}→{last_yr})",
                     labels={"pct_change": "% Change", "skill": "Skill"})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Multi-year data needed to show emerging/declining trends.")
