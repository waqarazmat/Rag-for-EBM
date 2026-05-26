"""
Evaluation Dashboard — Alex EBM RAG System
Results from LangSmith experiment: Alex-Advanced-v2-27056a35
Benchmark: Alex_Management_Benchmark_v8 (38 questions)
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
st.caption(
    "Benchmark: **Alex_Management_Benchmark_v8** · 38 questions · "
    "Model: GPT-4o-mini · Retrieval: FAISS + BM25 + CrossEncoder · "
    "Evaluator LLM: GPT-4o · Embeddings: BAAI/bge-base-en-v1.5"
)
st.divider()

# ── Raw data ───────────────────────────────────────────────────────────────────
RAW = [
    {"q": "Amor Fati (Obstacle Is the Way)",         "faith": 0.3333, "rel": 0.8684, "ctx": 1.0000, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 0.4,  "mrr": 1.0,    "lat": 10.42, "oot": False},
    {"q": "OKR Misses (Obstacle Is the Way)",         "faith": 0.3077, "rel": 0.7485, "ctx": 0.0000, "ans_rel": 0.9,  "grnd": 0.9,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 7.09,  "oot": False},
    {"q": "Three Disciplines (Obstacle Is the Way)",  "faith": 0.6875, "rel": 0.7479, "ctx": 0.0000, "ans_rel": 0.9,  "grnd": 0.7,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 6.51,  "oot": False},
    {"q": "Marcus Aurelius Quote (Obstacle)",         "faith": 0.6154, "rel": 0.9012, "ctx": 0.0000, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 6.43,  "oot": False},
    {"q": "Optimum Selling Strategy (Ready Fire Aim)","faith": 0.9412, "rel": 0.8540, "ctx": 0.0000, "ans_rel": 0.8,  "grnd": 0.6,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 8.04,  "oot": False},
    {"q": "Four-Stage Framework (Ready Fire Aim)",    "faith": 0.6000, "rel": 0.7591, "ctx": 0.7000, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 6.29,  "oot": False},
    {"q": "Rule #1 of Entrepreneurship (RFA)",        "faith": 0.6250, "rel": 0.8448, "ctx": 1.0000, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 0.6,  "mrr": 1.0,    "lat": 6.81,  "oot": False},
    {"q": "Social Proof (Influence)",                 "faith": 0.9000, "rel": 0.8775, "ctx": 0.7500, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 5.02,  "oot": False},
    {"q": "Reciprocity (Influence)",                  "faith": 0.2857, "rel": 0.8904, "ctx": 0.0000, "ans_rel": 0.9,  "grnd": 0.6,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 11.28, "oot": False},
    {"q": "Commitment Bias (Influence)",              "faith": 0.4706, "rel": 0.8437, "ctx": 0.0000, "ans_rel": 0.9,  "grnd": 0.9,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 6.10,  "oot": False},
    {"q": "Active Fund Managers (Random Walk)",       "faith": 0.7778, "rel": 0.7852, "ctx": 1.0000, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 0.8,  "mrr": 1.0,    "lat": 5.50,  "oot": False},
    {"q": "Technical Analysis (Random Walk)",         "faith": 1.0000, "rel": 0.7832, "ctx": 1.0000, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 6.45,  "oot": False},
    {"q": "Fundamental Analysis / Graham (RW)",       "faith": 0.9000, "rel": 0.7707, "ctx": 0.8333, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 5.21,  "oot": False},
    {"q": "Dalio's Core Principle (Principles)",      "faith": 0.0000, "rel": 0.4190, "ctx": 0.0000, "ans_rel": 0.0,  "grnd": 0.0,  "rec": 1.0,  "prec": 0.8,  "mrr": 1.0,    "lat": 3.88,  "oot": False},
    {"q": "Failure-to-Progress Formula (Principles)", "faith": 0.6842, "rel": 0.7629, "ctx": 0.0000, "ans_rel": 0.6,  "grnd": 0.5,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 8.33,  "oot": False},
    {"q": "Two Internal Barriers (Principles)",       "faith": 0.3333, "rel": 0.0000, "ctx": 0.0000, "ans_rel": 0.8,  "grnd": 1.0,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 5.65,  "oot": False},
    {"q": "Asset vs Liability — Car (Rich Dad)",      "faith": 0.5833, "rel": 0.8446, "ctx": 1.0000, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 9.41,  "oot": False},
    {"q": "Money Flow — Poor/Middle/Rich (Rich Dad)", "faith": 0.8333, "rel": 0.8999, "ctx": 0.3333, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 5.74,  "oot": False},
    {"q": "Lifestyle Inflation (Rich Dad)",           "faith": 0.5455, "rel": 0.6800, "ctx": 0.0000, "ans_rel": 0.8,  "grnd": 0.6,  "rec": 1.0,  "prec": 0.4,  "mrr": 0.25,   "lat": 7.86,  "oot": False},
    {"q": "Four Laws of Behavior Change (Atomic H.)", "faith": 0.0000, "rel": 0.9094, "ctx": 0.0000, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 0.8,  "mrr": 1.0,    "lat": 6.22,  "oot": False},
    {"q": "Habit Stacking (Atomic Habits)",           "faith": 0.4286, "rel": 0.8967, "ctx": 0.3250, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 0.4,  "mrr": 0.25,   "lat": 4.77,  "oot": False},
    {"q": "Stories vs Statistics (Made to Stick)",    "faith": 0.2500, "rel": 0.7154, "ctx": 0.3333, "ans_rel": 0.9,  "grnd": 0.9,  "rec": 1.0,  "prec": 0.4,  "mrr": 0.5,    "lat": 6.49,  "oot": False},
    {"q": "Mean-Variance Analysis (Markowitz)",       "faith": 0.7500, "rel": 0.8493, "ctx": 1.0000, "ans_rel": 0.9,  "grnd": 0.9,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 5.93,  "oot": False},
    {"q": "Curse of Knowledge (Made to Stick)",       "faith": 0.0714, "rel": 0.6422, "ctx": 0.0000, "ans_rel": 0.7,  "grnd": 0.6,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 7.42,  "oot": False},
    {"q": "⚠️ Blockchain / ESG [OOT]",               "faith": 0.8182, "rel": 0.0000, "ctx": 0.0000, "ans_rel": 0.2,  "grnd": 0.3,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 5.85,  "oot": True},
    {"q": "Anchoring Bias (Thinking F&S)",            "faith": 0.7692, "rel": 0.7890, "ctx": 0.8333, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 0.6,  "mrr": 1.0,    "lat": 7.78,  "oot": False},
    {"q": "System 1 vs System 2 (Thinking F&S)",     "faith": 0.5714, "rel": 0.8422, "ctx": 0.3333, "ans_rel": 1.0,  "grnd": 0.8,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 5.58,  "oot": False},
    {"q": "Six Leadership Styles (Goleman 2000)",     "faith": 0.3333, "rel": 0.8376, "ctx": 0.3333, "ans_rel": 0.9,  "grnd": 0.7,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 8.35,  "oot": False},
    {"q": "Libertarian Paternalism (Sunstein/Thaler)","faith": 0.9333, "rel": 0.9018, "ctx": 0.7000, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 0.8,  "mrr": 1.0,    "lat": 7.03,  "oot": False},
    {"q": "Behavioural Finance Shift (Shiller 2006)", "faith": 0.8750, "rel": 0.7613, "ctx": 0.2500, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 0.2,  "mrr": 0.25,   "lat": 7.03,  "oot": False},
    {"q": "Five EI Components (Goleman HBR)",         "faith": 0.6000, "rel": 0.8824, "ctx": 0.8333, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 0.6,  "mrr": 0.3333, "lat": 8.53,  "oot": False},
    {"q": "Three Forms of EMH (Fama 1970)",           "faith": 0.8182, "rel": 0.9143, "ctx": 0.2500, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 0.8,  "mrr": 1.0,    "lat": 10.36, "oot": False},
    {"q": "Habit Formation Duration (Lally 2010)",    "faith": 0.9286, "rel": 0.8198, "ctx": 0.0000, "ans_rel": 0.95, "grnd": 0.9,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 7.22,  "oot": False},
    {"q": "⚠️ mRNA Vaccine Trials [OOT]",            "faith": 0.9000, "rel": 0.0000, "ctx": 0.0000, "ans_rel": 0.4,  "grnd": 0.5,  "rec": 0.0,  "prec": 0.0,  "mrr": 0.0,    "lat": 7.26,  "oot": True},
    {"q": "Psychological Safety (Edmondson 1999)",    "faith": 0.7857, "rel": 0.8933, "ctx": 0.9167, "ans_rel": 1.0,  "grnd": 0.9,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 8.24,  "oot": False},
    {"q": "SUCCESs Framework (Made to Stick)",        "faith": 0.0000, "rel": 0.8122, "ctx": 0.0000, "ans_rel": 0.9,  "grnd": 0.7,  "rec": 1.0,  "prec": 0.8,  "mrr": 1.0,    "lat": 8.46,  "oot": False},
    {"q": "Planning Fallacy (Thinking F&S)",          "faith": 0.6250, "rel": 0.7752, "ctx": 0.9167, "ans_rel": 1.0,  "grnd": 1.0,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 5.72,  "oot": False},
    {"q": "Descriptive vs Injunctive Norms (Cialdini)","faith": 0.7333, "rel": 0.9486, "ctx": 1.0000, "ans_rel": 0.9, "grnd": 0.9,  "rec": 1.0,  "prec": 1.0,  "mrr": 1.0,    "lat": 7.20,  "oot": False},
]

df = pd.DataFrame(RAW)
df.index = [f"Q{i+1}" for i in range(len(df))]

IN_SCOPE = df[~df["oot"]]
METRIC_COLS = {
    "Faithfulness":      "faith",
    "Answer Relevancy":  "rel",
    "Context Precision": "ctx",
    "Answer Relevance":  "ans_rel",
    "Groundedness":      "grnd",
    "Recall@k":          "rec",
    "Precision@k":       "prec",
    "MRR":               "mrr",
}

# ── KPI Cards ─────────────────────────────────────────────────────────────────
means_all  = {k: df[v].mean()       for k, v in METRIC_COLS.items()}
means_in   = {k: IN_SCOPE[v].mean() for k, v in METRIC_COLS.items()}
avg_lat    = IN_SCOPE["lat"].mean()

TARGETS = {
    "Faithfulness": 0.70, "Answer Relevancy": 0.80, "Context Precision": 0.50,
    "Answer Relevance": 0.90, "Groundedness": 0.80, "Recall@k": 0.85,
    "Precision@k": 0.70, "MRR": 0.80,
}

st.markdown("#### Mean scores — 36 in-scope questions  *(2 intentional out-of-scope excluded)*")
cols = st.columns(9)
for col, (label, col_key) in zip(cols, METRIC_COLS.items()):
    v = means_in[label]
    t = TARGETS[label]
    col.metric(label, f"{v:.3f}", delta=f"{v-t:+.3f}", delta_color="normal" if v >= t else "inverse")
cols[-1].metric("Avg Latency", f"{avg_lat:.1f}s")

st.divider()

# ── Main charts ────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Per-Question Score")
    metric_choice = st.selectbox("Metric", list(METRIC_COLS.keys()), label_visibility="collapsed")
    col_key = METRIC_COLS[metric_choice]
    vals    = df[col_key].tolist()
    labels  = df.index.tolist()
    topics  = df["q"].tolist()
    colors  = ["#e8c11a" if df["oot"].iloc[i] else
               ("#d62728" if v < 0.5 else "#ff7f0e" if v < 0.75 else "#2ca02c")
               for i, v in enumerate(vals)]

    fig_bar = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=colors,
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
        customdata=topics,
        hovertemplate="<b>%{x}</b><br>%{customdata}<br>" + metric_choice + ": %{y:.3f}<extra></extra>",
    ))
    fig_bar.add_hline(
        y=means_in[metric_choice], line_dash="dash", line_color="grey",
        annotation_text=f"mean {means_in[metric_choice]:.3f} (in-scope)",
        annotation_position="top right",
    )
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.18], title="Score"),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        margin=dict(t=20, b=10), height=400,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("🟡 = intentional out-of-scope query · 🔴 <0.5 · 🟠 0.5–0.75 · 🟢 ≥0.75 · hover for full question")

with right:
    st.subheader("Radar — Mean Scores (in-scope)")
    r_labels = list(METRIC_COLS.keys())
    r_vals   = [means_in[m] for m in r_labels] + [means_in[r_labels[0]]]
    r_labels_closed = r_labels + [r_labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=r_vals, theta=r_labels_closed,
        fill="toself",
        fillcolor="rgba(26,115,232,0.15)",
        line=dict(color="#1a73e8", width=2),
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40), height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Out-of-scope handling ──────────────────────────────────────────────────────
st.subheader("Out-of-Scope Query Handling")
oot_df = df[df["oot"]][["q", "ans_rel", "grnd", "faith", "rel"]].copy()
oot_df.columns = ["Question", "Answer Relevance", "Groundedness", "Faithfulness", "RAGAS Relevancy"]
st.dataframe(oot_df.style.format(precision=3), use_container_width=True)
with st.expander("What should happen for out-of-scope queries?"):
    st.markdown("""
    Alex is designed to **decline gracefully** when a query falls outside the knowledge base.
    For blockchain/mRNA questions: Answer Relevance ≈ 0.2–0.4 confirms Alex hedged correctly rather than hallucinating.
    High Faithfulness scores on these rows mean the *words* Alex used were grounded — but Answer Relevance correctly
    shows the response wasn't useful for the question, as intended.
    """)

st.divider()

# ── Score distribution ─────────────────────────────────────────────────────────
st.subheader("Score Distribution")
dist_metric = st.selectbox("Metric for distribution", list(METRIC_COLS.keys()), key="dist", label_visibility="collapsed")
dist_col = METRIC_COLS[dist_metric]
fig_hist = px.histogram(
    IN_SCOPE, x=dist_col, nbins=10,
    labels={dist_col: dist_metric},
    color_discrete_sequence=["#1a73e8"],
    title=f"{dist_metric} — distribution across 36 in-scope questions",
)
fig_hist.update_layout(
    xaxis=dict(range=[0, 1.05]),
    margin=dict(t=40, b=10), height=300,
    plot_bgcolor="white", paper_bgcolor="white",
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Full table ─────────────────────────────────────────────────────────────────
st.subheader("Full Results Table")
display_df = df[["q", "faith", "rel", "ctx", "ans_rel", "grnd", "rec", "prec", "mrr", "lat"]].copy()
display_df.columns = ["Question", "Faithfulness", "Ans Relevancy", "Ctx Precision",
                      "Ans Relevance", "Groundedness", "Recall@k", "Precision@k", "MRR", "Latency (s)"]

score_cols = ["Faithfulness", "Ans Relevancy", "Ctx Precision", "Ans Relevance",
              "Groundedness", "Recall@k", "Precision@k", "MRR"]

def color_score(val):
    if not isinstance(val, float): return ""
    if val >= 0.75: return "background-color: #c6efce; color: #276221"
    if val >= 0.50: return "background-color: #ffeb9c; color: #9c5700"
    return "background-color: #ffc7ce; color: #9c0006"

styled = display_df.style.applymap(color_score, subset=score_cols).format(precision=3)
st.dataframe(styled, use_container_width=True, height=480)

st.divider()
st.caption(
    "Experiment: `Alex-Advanced-v2-27056a35` · Dataset: `Alex_Management_Benchmark_v8` · "
    "38 questions (36 in-scope + 2 intentional OOT)"
)
