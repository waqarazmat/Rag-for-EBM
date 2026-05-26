"""
Evaluation Dashboard — Alex EBM RAG System
Results from LangSmith experiment: Alex-Full-Metrics-Final-8923859e (n=10 questions)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Alex — Evaluation Results",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Evaluation Results")
st.caption("Benchmark: 10 management & finance questions · Model: GPT-4o-mini · Retrieval: FAISS + BM25 + CrossEncoder")
st.divider()

# ── Data ──────────────────────────────────────────────────────────────────────
QUESTIONS = [
    "Systematic Inquiry",
    "Unfinished Prototype",
    "Attitude of Wisdom",
    "Failure & Risk",
    "Standard Formulas",
    "Assets vs Liabilities",
    "Benchmarking Trap",
    "Financial Literacy",
    "Lifestyle Inflation",
    "Cause-and-Effect Logic",
]

FULL_QUESTIONS = [
    "What role does 'Systematic Inquiry' play when rigorous evidence is currently unavailable?",
    "What does it mean to treat an organization as an 'Unfinished Prototype' in decision-making?",
    "Why is the 'Attitude of Wisdom' essential for implementing Evidence-Based Management?",
    "How should successful leaders manage the mindset toward 'Failure' and 'Risk'?",
    "What are the dangers of blindly following 'Standard Formulas' or 'Conventional Wisdom'?",
    "How does the distinction between 'Assets' and 'Liabilities' determine long-term financial freedom?",
    "How can a manager avoid the 'Benchmarking Trap' when looking at successful competitors?",
    "Why is 'Financial Literacy' considered the most important foundation for building wealth?",
    "How does 'Lifestyle Inflation' prevent high-income earners from accumulating real wealth?",
    "Why is parsing 'Cause-and-Effect Logic' a fundamental requirement for Evidence-Based Management?",
]

df = pd.DataFrame({
    "Question":         QUESTIONS,
    "Full Question":    FULL_QUESTIONS,
    "Answer Relevance": [1.00, 1.00, 1.00, 1.00, 0.85, 1.00, 1.00, 1.00, 0.90, 1.00],
    "Groundedness":     [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.60, 0.90],
    "Faithfulness":     [0.933, 0.867, 0.947, 0.619, 0.714, 0.850, 1.000, 0.850, 0.778, 0.800],
    "Context Precision":[0.756, 0.000, 1.000, 0.589, 1.000, 0.867, 1.000, 0.450, 0.250, 1.000],
    "Recall@k":         [1.00, 1.00, 1.00, 1.00, 0.667, 1.00, 1.00, 1.00, 1.00, 1.00],
    "Precision@k":      [0.80, 1.00, 1.00, 1.00, 0.80, 1.00, 1.00, 1.00, 0.40, 1.00],
    "MRR":              [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 1.00],
    "Latency (s)":      [10.34, 9.47, 8.02, 12.68, 11.76, 8.50, 9.39, 8.47, 9.84, 11.98],
})

METRIC_COLS = ["Answer Relevance", "Groundedness", "Faithfulness", "Context Precision",
               "Recall@k", "Precision@k", "MRR"]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
means = df[METRIC_COLS].mean()
avg_latency = df["Latency (s)"].mean()

cols = st.columns(8)
kpis = [
    ("Answer Relevance", means["Answer Relevance"], 0.95),
    ("Groundedness",     means["Groundedness"],     0.85),
    ("Faithfulness",     means["Faithfulness"],      0.80),
    ("Context Precision",means["Context Precision"], 0.65),
    ("Recall@k",         means["Recall@k"],          0.90),
    ("Precision@k",      means["Precision@k"],        0.85),
    ("MRR",              means["MRR"],               0.90),
    ("Avg Latency",      avg_latency,                None),
]

for col, (label, value, threshold) in zip(cols, kpis):
    with col:
        if label == "Avg Latency":
            st.metric(label, f"{value:.1f}s")
        else:
            delta = f"{value - threshold:+.3f} vs target" if threshold else None
            st.metric(label, f"{value:.3f}", delta=delta,
                      delta_color="normal" if (threshold and value >= threshold) else "inverse")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Per-Question Metrics")
    metric_choice = st.selectbox(
        "Metric",
        METRIC_COLS,
        label_visibility="collapsed",
    )
    colors = [
        "#d62728" if v < 0.5 else "#ff7f0e" if v < 0.75 else "#2ca02c"
        for v in df[metric_choice]
    ]
    fig_bar = go.Figure(go.Bar(
        x=df["Question"],
        y=df[metric_choice],
        marker_color=colors,
        text=[f"{v:.3f}" for v in df[metric_choice]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>" + metric_choice + ": %{y:.3f}<extra></extra>",
    ))
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.15], title="Score"),
        xaxis=dict(tickangle=-30),
        margin=dict(t=20, b=10),
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_bar.add_hline(y=means[metric_choice], line_dash="dash", line_color="grey",
                      annotation_text=f"mean {means[metric_choice]:.3f}",
                      annotation_position="top right")
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Radar — Mean Metrics")
    radar_metrics = ["Answer Relevance", "Groundedness", "Faithfulness",
                     "Context Precision", "Recall@k", "Precision@k", "MRR"]
    radar_vals = [means[m] for m in radar_metrics] + [means[radar_metrics[0]]]
    radar_labels = radar_metrics + [radar_metrics[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_vals,
        theta=radar_labels,
        fill="toself",
        fillcolor="rgba(26,115,232,0.15)",
        line=dict(color="#1a73e8", width=2),
        name="Advanced RAG",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Advanced vs Baseline ───────────────────────────────────────────────────────
st.subheader("Advanced RAG vs Baseline (Ablation)")
st.caption("Baseline: single FAISS query, no BM25 / CrossEncoder reranking · n=9")

comp_df = pd.DataFrame({
    "System":      ["Advanced RAG", "Baseline RAG"],
    "Precision@k": [0.844, 0.944],
    "Recall@k":    [0.963, 1.000],
    "MRR":         [0.917, 0.944],
})

fig_comp = px.bar(
    comp_df.melt(id_vars="System", var_name="Metric", value_name="Score"),
    x="Metric", y="Score", color="System",
    barmode="group",
    color_discrete_map={"Advanced RAG": "#1a73e8", "Baseline RAG": "#e8a21a"},
    text_auto=".3f",
)
fig_comp.update_traces(textposition="outside")
fig_comp.update_layout(
    yaxis=dict(range=[0, 1.15]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=40, b=10),
    height=340,
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig_comp, use_container_width=True)

with st.expander("ℹ️ Why does Baseline beat Advanced on raw retrieval?"):
    st.markdown("""
    The Baseline retrieves **fewer chunks (k=4)** and uses a **single query**, so precision is artificially high.
    Advanced RAG uses **multi-query expansion + BM25 + CrossEncoder** — this introduces noise on Q8 (Financial Literacy)
    where the HyDE-generated queries drifted from the original intent, dropping P@k from 0.750 → 0.200.
    The metrics that matter for EBM quality — **Faithfulness** and **Groundedness** — are only available for the Advanced system.
    """)

st.divider()

# ── Full data table ────────────────────────────────────────────────────────────
st.subheader("Full Results Table")

display_df = df.drop(columns=["Full Question"]).set_index("Question")
display_df = display_df.round(3)

def color_score(val):
    if not isinstance(val, float):
        return ""
    if val >= 0.9:
        return "background-color: #c6efce; color: #276221"
    elif val >= 0.7:
        return "background-color: #ffeb9c; color: #9c5700"
    else:
        return "background-color: #ffc7ce; color: #9c0006"

styled = display_df.style.applymap(
    color_score,
    subset=["Answer Relevance", "Groundedness", "Faithfulness",
            "Context Precision", "Recall@k", "Precision@k", "MRR"]
).format(precision=3)

st.dataframe(styled, use_container_width=True, height=420)

st.divider()
st.caption("Experiment: `Alex-Full-Metrics-Final-8923859e` · Evaluator LLM: GPT-4o · Embeddings: BAAI/bge-base-en-v1.5")
