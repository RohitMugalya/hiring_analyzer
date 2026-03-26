import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import (generate_synthetic_candidates, generate_synthetic_jobs,
                                generate_synthetic_onet, SKILLS, INDUSTRIES, ROLES, EDUCATION)

st.set_page_config(page_title="Candidate Insight Engine", page_icon="🎯", layout="wide")

st.title("🎯 Candidate Insight Engine")
st.markdown("Enter your profile below to get personalized hiring insights, career transition analysis, and skill gap recommendations — powered by real market data.")

# ── Load session data ─────────────────────────────────────────────────────────
def get_data():
    candidates_df = st.session_state.get("candidates_df")
    if candidates_df is None or candidates_df.empty:
        candidates_df = generate_synthetic_candidates(400)
    jobs_df = st.session_state.get("jobs_df")
    if jobs_df is None or jobs_df.empty:
        jobs_df = generate_synthetic_jobs(500)
    onet_df = st.session_state.get("onet_df")
    if onet_df is None or onet_df.empty:
        onet_df = generate_synthetic_onet()
    skill_demand  = st.session_state.get("skill_demand_df")

    if skill_demand is None:
        from utils.spark_utils import spark_process_jobs
        skill_demand = spark_process_jobs(jobs_df, st.session_state.get("spark"))
        st.session_state["skill_demand_df"] = skill_demand

    return candidates_df, jobs_df, onet_df, skill_demand

candidates_df, jobs_df, onet_df, skill_demand = get_data()

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_industries():
    if "industry" in candidates_df.columns:
        return sorted(candidates_df["industry"].dropna().unique().tolist())
    return INDUSTRIES

def get_roles():
    if "applied_role" in candidates_df.columns:
        return sorted(candidates_df["applied_role"].dropna().unique().tolist())
    return ROLES

def get_skills_list():
    all_skills = set(SKILLS)
    if "skill" in skill_demand.columns:
        all_skills.update(skill_demand["skill"].dropna().unique().tolist())
    if "skill" in onet_df.columns:
        all_skills.update(onet_df["skill"].dropna().unique().tolist())
    return sorted(all_skills)

def compute_hire_probability(target_industry, target_role, user_skills, years_exp, education):
    """
    Hybrid: pattern-match base rate, optionally layer calibrated LR if class balance is ok.
    Returns (probability float 0-1, method string, confidence string)
    """
    df = candidates_df.copy()
    df["hired"] = pd.to_numeric(df["hired"], errors="coerce").fillna(0).astype(int)

    # ── Base rate from matching filters ──────────────────────────────────────
    subset = df.copy()
    if "industry" in df.columns:
        ind_match = df[df["industry"] == target_industry]
        if len(ind_match) >= 10:
            subset = ind_match
    if "applied_role" in subset.columns:
        role_match = subset[subset["applied_role"] == target_role]
        if len(role_match) >= 5:
            subset = role_match

    base_rate = subset["hired"].mean() if len(subset) > 0 else 0.5

    # ── Skill overlap bonus ───────────────────────────────────────────────────
    demanded = get_demanded_skills(target_industry, top_n=20)
    demanded_set = set([s.lower() for s in demanded])
    user_set     = set([s.lower() for s in user_skills])
    overlap = len(user_set & demanded_set)
    skill_bonus = min(overlap / max(len(demanded_set), 1), 1.0) * 0.20

    # ── Experience adjustment ─────────────────────────────────────────────────
    exp_adj = 0.0
    if years_exp >= 5:
        exp_adj = 0.05
    elif years_exp >= 2:
        exp_adj = 0.02
    elif years_exp == 0:
        exp_adj = -0.05

    # ── Education adjustment ──────────────────────────────────────────────────
    edu_map = {"High School": -0.05, "Associate's": -0.02,
               "Bachelor's": 0.0, "Master's": 0.04, "PhD": 0.06}
    edu_adj = edu_map.get(education, 0.0)

    raw_prob = base_rate + skill_bonus + exp_adj + edu_adj

    # ── Try logistic regression if class balance is OK ────────────────────────
    method = "Pattern Matching"
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        hire_counts = df["hired"].value_counts()
        minority_ratio = hire_counts.min() / hire_counts.sum() if len(hire_counts) == 2 else 0
        # Only use LR if minority class >= 25% of data (reasonably balanced)
        if minority_ratio >= 0.25 and len(df) >= 50:
            features = []
            labels   = df["hired"].values

            # Encode industry
            le_ind = LabelEncoder()
            if "industry" in df.columns:
                ind_enc = le_ind.fit_transform(df["industry"].fillna("Unknown"))
                features.append(ind_enc.reshape(-1, 1))

            # Encode role
            le_role = LabelEncoder()
            if "applied_role" in df.columns:
                role_enc = le_role.fit_transform(df["applied_role"].fillna("Unknown"))
                features.append(role_enc.reshape(-1, 1))

            # Experience
            if "years_experience" in df.columns:
                exp_col = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0).values
                features.append(exp_col.reshape(-1, 1))

            if len(features) >= 2:
                X = np.hstack(features)
                lr = LogisticRegression(max_iter=300, class_weight="balanced")
                lr.fit(X, labels)

                # Build input row
                x_new = []
                if "industry" in df.columns:
                    try:
                        ind_val = le_ind.transform([target_industry])[0]
                    except ValueError:
                        ind_val = 0
                    x_new.append(ind_val)
                if "applied_role" in df.columns:
                    try:
                        role_val = le_role.transform([target_role])[0]
                    except ValueError:
                        role_val = 0
                    x_new.append(role_val)
                if "years_experience" in df.columns:
                    x_new.append(years_exp)

                lr_prob = lr.predict_proba([x_new])[0][1]
                # Blend: 60% LR + 40% pattern
                raw_prob = 0.6 * lr_prob + 0.4 * raw_prob
                method = "ML (Logistic Regression, balanced)"
    except Exception:
        pass

    final_prob = float(np.clip(raw_prob, 0.05, 0.95))

    if final_prob >= 0.65:
        confidence = "High"
    elif final_prob >= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"

    return final_prob, method, confidence


def get_demanded_skills(industry, top_n=15):
    if skill_demand is None or skill_demand.empty:
        return []
    if "industry" in skill_demand.columns:
        df = skill_demand[skill_demand["industry"] == industry]
    else:
        df = skill_demand
    if df.empty:
        df = skill_demand
    top = df.groupby("skill")["demand_count"].sum().nlargest(top_n)
    return top.index.tolist()


def get_transition_feasibility(current_industry, target_industry, user_skills):
    """Returns hire rates for both domains and transferable skill count."""
    df = candidates_df.copy()
    df["hired"] = pd.to_numeric(df["hired"], errors="coerce").fillna(0).astype(int)

    if "industry" not in df.columns:
        return None, None, [], []

    curr_rate   = df[df["industry"] == current_industry]["hired"].mean() if current_industry != target_industry else None
    target_rate = df[df["industry"] == target_industry]["hired"].mean()

    curr_skills   = set(s.lower() for s in get_demanded_skills(current_industry, 20))
    target_skills = set(s.lower() for s in get_demanded_skills(target_industry, 20))
    user_set      = set(s.lower() for s in user_skills)

    transferable = [s for s in user_skills if s.lower() in target_skills]
    missing      = [s.title() for s in target_skills - user_set][:8]

    return curr_rate, target_rate, transferable, missing


def get_onet_skills_for_role(role, top_n=10):
    if onet_df is None or onet_df.empty:
        return pd.DataFrame()
    matches = onet_df[onet_df["occupation"].str.contains(role, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()
    return matches.groupby("skill")["importance"].mean().nlargest(top_n).reset_index()


def find_similar_candidates(target_industry, target_role, user_skills, top_n=5):
    df = candidates_df.copy()
    df["hired"] = pd.to_numeric(df["hired"], errors="coerce").fillna(0).astype(int)
    user_set = set(s.lower() for s in user_skills)

    def overlap(row):
        row_skills = set(s.strip().lower() for s in str(row.get("skills", "")).split(","))
        return len(user_set & row_skills)

    df["skill_overlap"] = df.apply(overlap, axis=1)
    filtered = df.copy()
    if "industry" in df.columns:
        ind_match = df[df["industry"] == target_industry]
        if len(ind_match) > 0:
            filtered = ind_match
    if "applied_role" in filtered.columns:
        role_match = filtered[filtered["applied_role"] == target_role]
        if len(role_match) > 0:
            filtered = role_match

    top = filtered.sort_values("skill_overlap", ascending=False).head(top_n)
    return top


# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.subheader("📋 Your Profile")

with st.form("candidate_form"):
    col1, col2 = st.columns(2)

    with col1:
        current_industry = st.selectbox("Current Industry / Domain", get_industries(), index=0)
        target_industry  = st.selectbox("Target Industry / Domain", get_industries(), index=1)
        target_role      = st.selectbox("Target Job Role", get_roles(), index=0)

    with col2:
        education   = st.selectbox("Highest Education Level",
                                    ["High School", "Associate's", "Bachelor's", "Master's", "PhD"],
                                    index=2)
        years_exp   = st.slider("Years of Experience", 0, 25, 2)
        user_skills = st.multiselect("Your Skills", get_skills_list(),
                                      default=get_skills_list()[:4],
                                      help="Select all skills you currently have")

    submitted = st.form_submit_button("🔍 Analyze My Profile", type="primary")

if not submitted:
    st.info("Fill in your profile above and click **Analyze My Profile** to get insights.")
    st.stop()

if not user_skills:
    st.warning("Please select at least one skill.")
    st.stop()

# ── RESULTS ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Your Personalized Insights")

hire_prob, method, confidence = compute_hire_probability(
    target_industry, target_role, user_skills, years_exp, education)

curr_rate, target_rate, transferable, missing_skills = get_transition_feasibility(
    current_industry, target_industry, user_skills)

demanded_skills = get_demanded_skills(target_industry, top_n=20)
demanded_set    = set(s.lower() for s in demanded_skills)
user_set        = set(s.lower() for s in user_skills)
skill_match_pct = len(user_set & demanded_set) / max(len(demanded_set), 1) * 100
missing_top     = [s.title() for s in demanded_set - user_set][:8]
matched_skills  = [s for s in user_skills if s.lower() in demanded_set]

is_transition   = current_industry.lower() != target_industry.lower()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

k1.metric("🎯 Hire Probability", f"{hire_prob*100:.1f}%",
          delta=f"{confidence} confidence")

k2.metric("🔗 Skill Match", f"{skill_match_pct:.1f}%",
          delta=f"{len(matched_skills)} of {len(demanded_set)} skills matched")

k3.metric("🔄 Career Transition",
          "Same Domain" if not is_transition else "Cross-Domain",
          delta=f"{current_industry} → {target_industry}")

k4.metric("📚 Missing Key Skills", f"{len(missing_top)}",
          delta="skills to acquire", delta_color="inverse")

st.caption(f"_Hire probability computed using: {method}_")
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Hire Probability",
    "🔄 Career Transition",
    "🔧 Skill Gap Analysis",
    "👥 Similar Candidates"
])

# ── Tab 1: Hire Probability ───────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=hire_prob * 100,
            title={"text": f"Hire Probability<br><span style='font-size:0.8em;color:gray'>Target: {target_role} in {target_industry}</span>"},
            delta={"reference": (target_rate or 0.5) * 100, "suffix": "% (industry avg)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2E75B6"},
                "steps": [
                    {"range": [0, 40],  "color": "#fee2e2"},
                    {"range": [40, 65], "color": "#fef9c3"},
                    {"range": [65, 100],"color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "orange", "width": 3},
                    "thickness": 0.75,
                    "value": (target_rate or 0.5) * 100
                }
            }
        ))
        fig.update_layout(height=320, margin=dict(t=60, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### What This Means")
        if hire_prob >= 0.65:
            st.success(f"**Strong profile** for {target_role} in {target_industry}. Your skills and experience align well with market demand.")
        elif hire_prob >= 0.40:
            st.warning(f"**Moderate profile.** You have a reasonable chance but closing skill gaps will improve your odds significantly.")
        else:
            st.error(f"**Profile needs strengthening.** The target role has high competition or your skills don't closely match current demand in {target_industry}.")

        st.markdown("#### Factors Considered")
        edu_adj_map = {"High School": "-5", "Associate's": "-2", "Bachelor's": "0", "Master's": "+4", "PhD": "+6"}
        edu_adj_str = edu_adj_map.get(education, "0") + "%"
        exp_sign = "+" if years_exp >= 2 else ""
        exp_val  = 2 if years_exp >= 2 else (-5 if years_exp == 0 else 0)
        factors = {
            "Industry hire rate": f"{(target_rate or 0.5)*100:.1f}% base",
            "Skill match bonus": f"+{min(len(user_set & demanded_set)/max(len(demanded_set),1), 1.0)*20:.1f}%",
            "Experience adjustment": f"{exp_sign}{exp_val}%",
            "Education adjustment": edu_adj_str,
        }
        for k, v in factors.items():
            st.markdown(f"- **{k}:** {v}")

        # Industry comparison bar
        df_c = candidates_df.copy()
        df_c["hired"] = pd.to_numeric(df_c["hired"], errors="coerce").fillna(0)
        if "industry" in df_c.columns:
            ind_rates = df_c.groupby("industry")["hired"].mean().reset_index()
            ind_rates.columns = ["industry", "hire_rate"]
            ind_rates["highlight"] = ind_rates["industry"].apply(
                lambda x: "Your Target" if x == target_industry else "Other")
            fig2 = px.bar(ind_rates.sort_values("hire_rate", ascending=False),
                          x="hire_rate", y="industry", orientation="h",
                          color="highlight",
                          color_discrete_map={"Your Target": "#2E75B6", "Other": "#CBD5E1"},
                          title="Hire Rate Across Industries",
                          labels={"hire_rate": "Hire Rate", "industry": ""})
            fig2.update_layout(height=300, showlegend=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Career Transition ──────────────────────────────────────────────────
with tab2:
    if not is_transition:
        st.info(f"Your current and target domain are both **{current_industry}**. Select different domains to see transition analysis.")
    else:
        st.markdown(f"#### Transition: {current_industry} → {target_industry}")

        col_a, col_b = st.columns(2)

        with col_a:
            # Feasibility score
            curr_skills_set   = set(s.lower() for s in get_demanded_skills(current_industry, 20))
            target_skills_set = set(s.lower() for s in get_demanded_skills(target_industry, 20))
            overlap_domains   = len(curr_skills_set & target_skills_set)
            total_target      = len(target_skills_set)
            domain_overlap_pct = overlap_domains / max(total_target, 1) * 100

            feasibility = "High" if domain_overlap_pct >= 50 else "Medium" if domain_overlap_pct >= 30 else "Low"
            color_map   = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

            st.metric("Domain Skill Overlap", f"{domain_overlap_pct:.1f}%",
                      delta=f"{color_map[feasibility]} {feasibility} Feasibility")

            transferable_count = len(transferable)
            st.metric("Your Transferable Skills", f"{transferable_count}",
                      delta=f"out of {len(user_skills)} skills you have")

            if curr_rate is not None:
                st.metric(f"Hire Rate in {current_industry}", f"{curr_rate*100:.1f}%")
            if target_rate is not None:
                st.metric(f"Hire Rate in {target_industry}", f"{target_rate*100:.1f}%")

        with col_b:
            # Venn-style breakdown
            only_current = curr_skills_set - target_skills_set
            shared       = curr_skills_set & target_skills_set
            only_target  = target_skills_set - curr_skills_set

            fig = go.Figure()
            categories = ["Only in Current Domain", "Transferable (Shared)", "Only in Target Domain"]
            values     = [len(only_current), len(shared), len(only_target)]
            colors     = ["#94a3b8", "#22c55e", "#2E75B6"]
            fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors,
                                  text=values, textposition="outside"))
            fig.update_layout(title="Skill Distribution Between Domains",
                               height=300, margin=dict(t=40, b=10),
                               yaxis_title="Skill Count")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### ✅ Your Transferable Skills")
        if transferable:
            cols = st.columns(4)
            for i, s in enumerate(transferable):
                cols[i % 4].success(s)
        else:
            st.warning("None of your current skills directly match the target domain's top demands.")

        st.markdown("#### 🚨 Skills to Acquire for Transition")
        if missing_skills:
            cols = st.columns(4)
            for i, s in enumerate(missing_skills):
                cols[i % 4].error(s)
        else:
            st.success("Your skillset already covers the key demands of the target domain!")

# ── Tab 3: Skill Gap Analysis ─────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### Skills You Have vs. Demanded in {target_industry}")
        all_demanded = get_demanded_skills(target_industry, top_n=15)
        demand_df = skill_demand[skill_demand["industry"] == target_industry].groupby("skill")["demand_count"].sum().nlargest(15).reset_index() if "industry" in skill_demand.columns else pd.DataFrame()

        if not demand_df.empty:
            demand_df["status"] = demand_df["skill"].apply(
                lambda s: "✅ You Have It" if s.lower() in user_set else "❌ You Don't Have It")
            fig = px.bar(demand_df, x="demand_count", y="skill", orientation="h",
                         color="status",
                         color_discrete_map={"✅ You Have It": "#22c55e", "❌ You Don't Have It": "#ef4444"},
                         title=f"Top 15 Demanded Skills in {target_industry}",
                         labels={"demand_count": "Demand Count", "skill": ""})
            fig.update_layout(height=420, yaxis=dict(autorange="reversed"), margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Priority Skills to Learn")
        if missing_top:
            priority_df = pd.DataFrame({"skill": missing_top})
            if not skill_demand.empty and "skill" in skill_demand.columns:
                demand_map = skill_demand.groupby("skill")["demand_count"].sum().to_dict()
                priority_df["demand"] = priority_df["skill"].map(
                    lambda s: demand_map.get(s, demand_map.get(s.lower(), 50)))
            else:
                priority_df["demand"] = range(len(missing_top), 0, -1)

            priority_df = priority_df.sort_values("demand", ascending=False)
            fig = px.bar(priority_df, x="demand", y="skill", orientation="h",
                         color="demand", color_continuous_scale="Reds",
                         title="Missing Skills — Ranked by Market Demand",
                         labels={"demand": "Market Demand", "skill": ""})
            fig.update_layout(height=420, yaxis=dict(autorange="reversed"),
                               margin=dict(t=40), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("You already have all top demanded skills for this domain!")

    # O*NET Role-Specific Skills
    st.markdown(f"#### O\u2217NET: Skill Importance for *{target_role}*")
    onet_skills = get_onet_skills_for_role(target_role)
    if not onet_skills.empty:
        onet_skills["status"] = onet_skills["skill"].apply(
            lambda s: "✅ You Have It" if s.lower() in user_set else "❌ Missing")
        fig = px.bar(onet_skills, x="importance", y="skill", orientation="h",
                     color="status",
                     color_discrete_map={"✅ You Have It": "#22c55e", "❌ Missing": "#f59e0b"},
                     title=f"O*NET Skill Importance for {target_role}",
                     labels={"importance": "Importance Score (1-5)", "skill": ""})
        fig.update_layout(height=350, yaxis=dict(autorange="reversed"), margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No O*NET data found for this role. Try a broader role name.")

# ── Tab 4: Similar Candidates ─────────────────────────────────────────────────
with tab4:
    st.markdown(f"#### Candidates with Similar Profiles — {target_role} in {target_industry}")
    similar = find_similar_candidates(target_industry, target_role, user_skills)

    if not similar.empty:
        hired_count    = int(similar["hired"].sum())
        not_hired      = len(similar) - hired_count
        hire_rate_sim  = hired_count / len(similar) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Similar Candidates Found", len(similar))
        c2.metric("Got Hired", hired_count)
        c3.metric("Hire Rate (Similar Profiles)", f"{hire_rate_sim:.1f}%")

        cols_to_show = [c for c in ["candidate_id", "applied_role", "industry",
                                     "skills", "hired", "skill_overlap"]
                        if c in similar.columns]
        st.dataframe(similar[cols_to_show].rename(columns={
            "candidate_id": "Candidate ID",
            "applied_role": "Role Applied",
            "industry": "Industry",
            "skills": "Skills",
            "hired": "Hired (1=Yes)",
            "skill_overlap": "Skill Overlap Count"
        }), use_container_width=True)

        # Outcome pie
        fig = go.Figure(go.Pie(
            labels=["Hired", "Not Hired"],
            values=[hired_count, not_hired],
            marker_colors=["#22c55e", "#ef4444"],
            hole=0.4
        ))
        fig.update_layout(title="Outcome of Similar Candidates", height=300,
                           margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No similar candidates found in the current dataset.")

st.divider()

# ── Summary Card ──────────────────────────────────────────────────────────────
st.subheader("📝 Your Profile Summary")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    | Field | Value |
    |---|---|
    | **Current Domain** | {current_industry} |
    | **Target Role** | {target_role} in {target_industry} |
    | **Experience** | {years_exp} years |
    | **Education** | {education} |
    | **Skills Provided** | {len(user_skills)} |
    | **Hire Probability** | {hire_prob*100:.1f}% ({confidence} confidence) |
    | **Skill Match** | {skill_match_pct:.1f}% |
    | **Transition Type** | {"Cross-Domain" if is_transition else "Same Domain"} |
    """)

with col2:
    st.markdown("**Top 3 Recommendations:**")
    recs = []
    if missing_top:
        recs.append(f"🔧 Learn **{missing_top[0]}** — highest demanded skill you're missing in {target_industry}")
    if len(missing_top) > 1:
        recs.append(f"📈 Adding **{missing_top[1]}** could increase your hire probability by ~5-10%")
    if years_exp < 2:
        recs.append("⏳ Gaining 2+ years of experience significantly improves hiring outcomes in most domains")
    if education in ["High School", "Associate's"] and target_industry in ["Technology", "Finance"]:
        recs.append(f"🎓 A Bachelor's degree improves outcomes in {target_industry} by ~4-6%")
    if not recs:
        recs.append("✅ Strong profile — focus on staying current with emerging skills in your domain")
    for r in recs[:3]:
        st.markdown(f"- {r}")
