import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import generate_synthetic_jobs, generate_synthetic_candidates
from utils.graph_utils import get_driver, build_graph, get_graph_for_viz, clear_graph

st.set_page_config(page_title="Knowledge Graph", page_icon="🕸️", layout="wide")

st.title("🕸️ Knowledge Graph Explorer")
st.markdown("Visualize relationships between **Candidates → Skills → Jobs → Companies** using Neo4j + Plotly.")


def get_jobs():
    df = st.session_state.get("jobs_df")
    if df is None or df.empty:
        df = generate_synthetic_jobs(500)
        st.session_state["jobs_df"] = df
    return df


def get_candidates():
    df = st.session_state.get("candidates_df")
    if df is None or df.empty:
        df = generate_synthetic_candidates(400)
        st.session_state["candidates_df"] = df
    return df


# ── Neo4j Connection Status ──────────────────────────────────────────────────
st.subheader("🔌 Neo4j Connection")
col1, col2 = st.columns([2, 1])
with col1:
    driver = get_driver()
    if driver:
        st.success("✅ Connected to Neo4j at bolt://localhost:7687")
        neo4j_available = True
    else:
        st.warning("⚠️ Neo4j unavailable — showing in-memory graph via NetworkX")
        neo4j_available = False

with col2:
    node_limit = st.slider("Max nodes to display", 30, 150, 80)

st.divider()

# ── Build Graph Button ────────────────────────────────────────────────────────
if neo4j_available:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔨 Build / Rebuild Graph in Neo4j", type="primary"):
            jobs_df = get_jobs()
            candidates_df = get_candidates()
            with st.spinner("Clearing old graph..."):
                clear_graph(driver)
            with st.spinner("Building knowledge graph..."):
                build_graph(driver, jobs_df.head(100), candidates_df.head(100))
            st.session_state["graph_built"] = True
            st.success("✅ Graph built successfully!")
    with col_b:
        st.info("Graph persists in Neo4j. You can also open **http://localhost:7474** to explore via Neo4j Browser.")


# ── Build in-memory graph (always available) ──────────────────────────────────
def build_networkx_graph(jobs_df, candidates_df, limit=80):
    G = nx.Graph()
    node_colors = {}
    node_labels = {}

    count = 0
    for _, row in jobs_df.iterrows():
        if count >= limit:
            break
        company = str(row.get("company", "Unknown"))[:20]
        role = str(row.get("title", "Unknown"))[:20]
        industry = str(row.get("industry", "Unknown"))

        if company not in G:
            G.add_node(company, node_type="company")
            node_colors[company] = "#6366f1"
        job_node = f"{role}_{industry}"[:30]
        if job_node not in G:
            G.add_node(job_node, node_type="job")
            node_colors[job_node] = "#f59e0b"
        G.add_edge(company, job_node, rel="OFFERS")

        skills_raw = str(row.get("skills_required", ""))
        for skill in [s.strip() for s in skills_raw.split(",") if s.strip()][:3]:
            if skill not in G:
                G.add_node(skill, node_type="skill")
                node_colors[skill] = "#22c55e"
            G.add_edge(job_node, skill, rel="REQUIRES")
        count += 1

    for _, row in candidates_df.head(40).iterrows():
        cid = str(row.get("candidate_id", f"C_{_}"))
        role = str(row.get("applied_role", "Unknown"))[:20]
        industry = str(row.get("industry", "Unknown"))
        job_node = f"{role}_{industry}"[:30]

        if cid not in G:
            G.add_node(cid, node_type="candidate")
            node_colors[cid] = "#ef4444" if row.get("hired", 0) == 0 else "#84cc16"
        if job_node not in G:
            G.add_node(job_node, node_type="job")
            node_colors[job_node] = "#f59e0b"
        G.add_edge(cid, job_node, rel="APPLIED_FOR")

    return G, node_colors


def plotly_graph(G, node_colors):
    pos = nx.spring_layout(G, seed=42, k=1.2)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.7, color="#94a3b8"),
                            hoverinfo="none")

    # Group nodes by type for legend
    type_groups = {}
    for node in G.nodes():
        ntype = G.nodes[node].get("node_type", "unknown")
        if ntype not in type_groups:
            type_groups[ntype] = {"x": [], "y": [], "text": [], "color": "gray"}
        x, y = pos[node]
        type_groups[ntype]["x"].append(x)
        type_groups[ntype]["y"].append(y)
        type_groups[ntype]["text"].append(str(node))
        type_groups[ntype]["color"] = node_colors.get(node, "gray")

    type_color_map = {
        "company": "#6366f1",
        "job": "#f59e0b",
        "skill": "#22c55e",
        "candidate": "#ef4444",
    }
    type_labels = {
        "company": "🏢 Company",
        "job": "💼 Job",
        "skill": "🎯 Skill",
        "candidate": "👤 Candidate",
    }

    node_traces = []
    for ntype, data in type_groups.items():
        trace = go.Scatter(
            x=data["x"], y=data["y"],
            mode="markers+text",
            marker=dict(size=12, color=type_color_map.get(ntype, "gray"),
                        line=dict(width=1, color="white")),
            text=data["text"],
            textposition="top center",
            textfont=dict(size=8),
            hoverinfo="text",
            name=type_labels.get(ntype, ntype)
        )
        node_traces.append(trace)

    fig = go.Figure(data=[edge_trace] + node_traces,
                    layout=go.Layout(
                        title="Knowledge Graph: Candidates → Jobs → Skills → Companies",
                        showlegend=True,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                        height=650,
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        font=dict(color="white"),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig


# ── Render Graph ──────────────────────────────────────────────────────────────
st.subheader("🗺️ Graph Visualization")

graph_source = "neo4j" if (neo4j_available and st.session_state.get("graph_built")) else "networkx"

if graph_source == "neo4j":
    with st.spinner("Fetching graph from Neo4j..."):
        try:
            nodes, edges = get_graph_for_viz(driver, limit=node_limit)
            if nodes:
                # Build NetworkX from Neo4j data
                G = nx.Graph()
                node_colors = {}
                type_color = {"Company": "#6366f1", "Job": "#f59e0b",
                              "Skill": "#22c55e", "Candidate": "#ef4444"}
                for nid, nlabel in nodes.items():
                    G.add_node(nid, node_type=nlabel.lower())
                    node_colors[nid] = type_color.get(nlabel, "#94a3b8")
                for src, tgt, rel in edges:
                    G.add_edge(src, tgt, rel=rel)
                fig = plotly_graph(G, node_colors)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Graph is empty. Click 'Build Graph' above.")
        except Exception as e:
            st.error(f"Neo4j query error: {e}")
else:
    jobs_df = get_jobs()
    candidates_df = get_candidates()
    with st.spinner("Building in-memory graph..."):
        G, node_colors = build_networkx_graph(jobs_df, candidates_df, limit=node_limit)
    fig = plotly_graph(G, node_colors)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Graph Stats ───────────────────────────────────────────────────────────────
st.subheader("📊 Graph Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Nodes", len(G.nodes()) if 'G' in dir() else 0)
c2.metric("Total Edges", len(G.edges()) if 'G' in dir() else 0)
if 'G' in dir() and len(G.nodes()) > 0:
    node_types = {}
    for n in G.nodes():
        t = G.nodes[n].get("node_type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    c3.metric("Companies", node_types.get("company", 0))
    c4.metric("Skills", node_types.get("skill", 0))

st.divider()

# ── Top Connected Nodes ────────────────────────────────────────────────────────
st.subheader("🔗 Most Connected Nodes")
if 'G' in dir() and len(G.nodes()) > 0:
    degree_df = pd.DataFrame(
        [(n, G.degree(n), G.nodes[n].get("node_type", "unknown")) for n in G.nodes()],
        columns=["Node", "Connections", "Type"]
    ).sort_values("Connections", ascending=False).head(15)

    import plotly.express as px
    fig2 = px.bar(degree_df, x="Connections", y="Node", color="Type",
                  orientation="h",
                  color_discrete_map={"company": "#6366f1", "job": "#f59e0b",
                                      "skill": "#22c55e", "candidate": "#ef4444"},
                  title="Top 15 Most Connected Nodes",
                  labels={"Connections": "Degree Centrality"})
    fig2.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

# ── Cypher Query Box ──────────────────────────────────────────────────────────
if neo4j_available:
    st.divider()
    st.subheader("🔍 Custom Cypher Query")
    default_query = "MATCH (s:Skill)<-[:REQUIRES]-(j:Job)<-[:OFFERS]-(c:Company) RETURN s.name, j.title, c.name LIMIT 20"
    query = st.text_area("Enter Cypher Query", value=default_query, height=80)
    if st.button("▶️ Run Query"):
        try:
            from utils.graph_utils import query_graph
            results = query_graph(driver, query)
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.info("No results returned.")
        except Exception as e:
            st.error(f"Query error: {e}")
