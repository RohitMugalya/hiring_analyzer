from neo4j import GraphDatabase
import pandas as pd
import streamlit as st

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "admin123"


def get_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        return None


def clear_graph(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def build_graph(driver, jobs_df: pd.DataFrame, candidates_df: pd.DataFrame):
    with driver.session() as session:
        # Create job nodes + skill nodes + relationships
        for _, row in jobs_df.iterrows():
            session.run(
                "MERGE (c:Company {name: $company}) "
                "MERGE (j:Job {title: $title, industry: $industry}) "
                "MERGE (c)-[:OFFERS]->(j)",
                company=str(row.get("company", "Unknown")),
                title=str(row.get("title", "Unknown")),
                industry=str(row.get("industry", "Unknown"))
            )
            skills_raw = str(row.get("skills_required", ""))
            for skill in [s.strip() for s in skills_raw.split(",") if s.strip()]:
                session.run(
                    "MERGE (s:Skill {name: $skill}) "
                    "MERGE (j:Job {title: $title, industry: $industry}) "
                    "MERGE (j)-[:REQUIRES]->(s)",
                    skill=skill,
                    title=str(row.get("title", "Unknown")),
                    industry=str(row.get("industry", "Unknown"))
                )

        # Create candidate nodes
        for _, row in candidates_df.head(100).iterrows():
            session.run(
                "MERGE (cand:Candidate {id: $id}) "
                "SET cand.hired = $hired "
                "MERGE (j:Job {title: $role, industry: $industry}) "
                "MERGE (cand)-[:APPLIED_FOR]->(j)",
                id=str(row.get("candidate_id", "unknown")),
                hired=int(row.get("hired", 0)),
                role=str(row.get("applied_role", "Unknown")),
                industry=str(row.get("industry", "Unknown"))
            )
            skills_raw = str(row.get("skills", ""))
            for skill in [s.strip() for s in skills_raw.split(",") if s.strip()][:5]:
                session.run(
                    "MERGE (s:Skill {name: $skill}) "
                    "MERGE (cand:Candidate {id: $id}) "
                    "MERGE (cand)-[:HAS_SKILL]->(s)",
                    skill=skill,
                    id=str(row.get("candidate_id", "unknown"))
                )


def query_graph(driver, query: str):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]


def get_graph_for_viz(driver, limit=80):
    """Return nodes and edges for Plotly visualization."""
    records = query_graph(driver, f"""
        MATCH (a)-[r]->(b)
        RETURN a, type(r) as rel, b
        LIMIT {limit}
    """)
    nodes = {}
    edges = []
    for rec in records:
        a = rec["a"]
        b = rec["b"]
        a_id = list(a.items())[0][1] if a else "?"
        b_id = list(b.items())[0][1] if b else "?"
        a_label = list(a.labels)[0] if hasattr(a, "labels") else "Node"
        b_label = list(b.labels)[0] if hasattr(b, "labels") else "Node"
        nodes[a_id] = a_label
        nodes[b_id] = b_label
        edges.append((a_id, b_id, rec["rel"]))
    return nodes, edges
