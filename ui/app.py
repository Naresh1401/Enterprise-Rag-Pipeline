"""
ui/app.py
==========
Streamlit frontend for the Enterprise RAG system.

Features:
  - Document upload + indexing
  - Chat interface with source citations
  - Hallucination risk indicator
  - Retrieval metrics visualisation
  - RAGAS evaluation runner

Run:
    streamlit run ui/app.py
"""

import streamlit as st
import requests
import json
import time
import plotly.graph_objects as go
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "Enterprise RAG",
    page_icon  = "🔍",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-low    { color: #28a745; font-weight: bold; }
.risk-medium { color: #fd7e14; font-weight: bold; }
.risk-high   { color: #dc3545; font-weight: bold; }
.source-card {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 4px;
    font-size: 0.9em;
}
.source-cited { border-left-color: #28a745; }
.metric-card {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(endpoint: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{endpoint}", timeout=120, **kwargs)
        return r.json(), r.ok
    except Exception as e:
        return {"error": str(e)}, False


def risk_badge(level: str) -> str:
    cls = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(level, "risk-low")
    return f'<span class="{cls}">● {level} RISK</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Enterprise RAG")
    st.caption("Production-grade RAG with RAGAS evaluation")
    st.divider()

    # Health check
    health = api_get("/health")
    if health:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline — run: uvicorn src.api.main:app --reload")
        st.stop()

    # Stats
    stats = api_get("/stats")
    if stats:
        st.metric("Documents Indexed", stats.get("documents_indexed", 0))
        st.metric("Chunks in Index",   stats.get("chunks_in_index", 0))
        st.metric("Queries Handled",   stats.get("queries_handled", 0))
        avg_lat = stats.get("avg_latency_ms", 0)
        st.metric("Avg Latency",       f"{avg_lat:.0f}ms")

    st.divider()
    st.caption(f"LLM: `{stats.get('llm_model', '—') if stats else '—'}`")
    st.caption(f"Embedder: `{stats.get('embedding_model', '—') if stats else '—'}`")


# ── Main Tabs ─────────────────────────────────────────────────────────────

tab_chat, tab_ingest, tab_eval = st.tabs(["💬 Chat", "📁 Ingest Documents", "📊 Evaluation"])


# ════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("Ask your knowledge base")

    # Settings row
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        question = st.text_input(
            "Question",
            placeholder="What is the refund policy? / Explain the authentication flow...",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.number_input("Top K chunks", min_value=1, max_value=20, value=5)
    with col3:
        detect_halluc = st.toggle("Hallucination check", value=True)

    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating..."):
                t0 = time.time()
                result, ok = api_post(
                    "/query",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "detect_hallucination": detect_halluc,
                    }
                )

            if not ok:
                st.error(f"Error: {result.get('detail', result)}")
            else:
                # ── Answer ───────────────────────────────────────────
                st.markdown("### Answer")

                if not result.get("has_answer"):
                    st.warning(result["answer"])
                else:
                    st.markdown(result["answer"])

                # ── Hallucination badge ──────────────────────────────
                if result.get("hallucination"):
                    h = result["hallucination"]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(
                            f"**Hallucination Risk** {risk_badge(h['risk_level'])}",
                            unsafe_allow_html=True,
                        )
                    with col_b:
                        st.metric("Faithfulness", f"{h['faithfulness_score']:.3f}")
                    with col_c:
                        if h.get("flagged_claims"):
                            with st.expander(f"⚠️ {len(h['flagged_claims'])} flagged claims"):
                                for claim in h["flagged_claims"]:
                                    st.caption(f"• {claim}")

                # ── Latency ──────────────────────────────────────────
                st.caption(f"⏱ {result['latency_ms']:.0f}ms | Model: {result['model']}")

                # ── Sources ──────────────────────────────────────────
                st.markdown("### Sources")
                for src in result.get("sources", []):
                    cited_cls = "source-cited" if src.get("cited") else "source-card"
                    cited_icon = "✅" if src.get("cited") else "📄"
                    page_info = f" · p.{src['page']}" if src.get("page") else ""
                    section_info = f" · {src['section']}" if src.get("section") else ""
                    st.markdown(
                        f'<div class="{cited_cls} source-card">'
                        f'<b>{cited_icon} [{src["rank"]}] {src["filename"]}</b>'
                        f'{page_info}{section_info} '
                        f'<span style="color:#6c757d">score: {src["score"]:.4f}</span><br>'
                        f'<i>{src["excerpt"]}</i>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Chat History ─────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    if question and st.button("Add to history"):
        st.session_state.history.append(question)

    if st.session_state.history:
        with st.expander("📜 Query History"):
            for q in reversed(st.session_state.history[-10:]):
                st.caption(f"• {q}")


# ════════════════════════════════════════════════════════════════════════
# TAB 2: INGEST
# ════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.header("Index Documents")
    st.info("Supported formats: **PDF, DOCX, HTML, TXT, Markdown**")

    uploaded = st.file_uploader(
        "Drop your documents here",
        type=["pdf", "docx", "doc", "html", "htm", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded:
        st.write(f"**{len(uploaded)} file(s) selected:**")
        for f in uploaded:
            st.caption(f"  · {f.name}  ({f.size / 1024:.1f} KB)")

        if st.button("⚡ Index Documents", type="primary"):
            with st.spinner(f"Indexing {len(uploaded)} file(s)..."):
                files_payload = [
                    ("files", (f.name, f.read(), f.type or "application/octet-stream"))
                    for f in uploaded
                ]
                result, ok = api_post("/ingest", files=files_payload)

            if ok:
                st.success(
                    f"✅ Indexed **{result['files_indexed']} files** → "
                    f"**{result['chunks_added']} chunks** in {result['latency_ms']:.0f}ms"
                )
                st.rerun()
            else:
                st.error(f"Indexing failed: {result.get('detail', result)}")

    st.divider()
    if st.button("🗑️ Clear Index", type="secondary"):
        result, ok = api_post("/index")
        if ok:
            st.success("Index cleared.")
            st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TAB 3: EVALUATION
# ════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.header("RAGAS Evaluation")
    st.markdown("Measure RAG quality: faithfulness, answer relevancy, context precision.")

    # Input
    st.subheader("Test Questions")
    default_qs = (
        "What is the main topic of the indexed documents?\n"
        "What are the key concepts discussed?\n"
        "Can you summarise the main findings?"
    )
    questions_text = st.text_area(
        "Enter one question per line",
        value=default_qs,
        height=150,
    )
    use_ground_truth = st.checkbox("Include ground truth answers (for context recall)")

    ground_truths_text = ""
    if use_ground_truth:
        ground_truths_text = st.text_area(
            "Ground truth answers (one per line, same order as questions)",
            height=150,
        )

    if st.button("▶️ Run Evaluation", type="primary"):
        questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
        ground_truths = None
        if use_ground_truth and ground_truths_text:
            ground_truths = [g.strip() for g in ground_truths_text.strip().split("\n") if g.strip()]

        if not questions:
            st.warning("Enter at least one question.")
        else:
            with st.spinner(f"Evaluating {len(questions)} questions..."):
                result, ok = api_post(
                    "/evaluate",
                    json={"questions": questions, "ground_truths": ground_truths},
                )

            if not ok:
                st.error(f"Evaluation failed: {result.get('detail', result)}")
            else:
                st.success("Evaluation complete!")

                # ── Score cards ──────────────────────────────────────
                cols = st.columns(4)
                metrics = [
                    ("Faithfulness",      result.get("faithfulness",      0)),
                    ("Answer Relevancy",  result.get("answer_relevancy",  0)),
                    ("Context Precision", result.get("context_precision", 0)),
                    ("RAGAS Score",       result.get("ragas_score",       0)),
                ]
                for col, (name, val) in zip(cols, metrics):
                    with col:
                        color = "#28a745" if val >= 0.7 else "#fd7e14" if val >= 0.5 else "#dc3545"
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div style="font-size:2em;font-weight:bold;color:{color}">{val:.3f}</div>'
                            f'<div style="color:#6c757d">{name}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Latency ──────────────────────────────────────────
                st.markdown("---")
                col1, col2 = st.columns(2)
                col1.metric("Latency P50", f"{result.get('latency_p50_ms', 0):.0f}ms")
                col2.metric("Latency P95", f"{result.get('latency_p95_ms', 0):.0f}ms")

                # ── Radar chart ──────────────────────────────────────
                fig = go.Figure(go.Scatterpolar(
                    r=[
                        result.get("faithfulness", 0),
                        result.get("answer_relevancy", 0),
                        result.get("context_precision", 0),
                        result.get("faithfulness", 0),
                    ],
                    theta=["Faithfulness", "Answer Relevancy", "Context Precision", "Faithfulness"],
                    fill="toself",
                    line_color="#007bff",
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 1])),
                    showlegend=False,
                    title="RAGAS Metric Radar",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Raw results ──────────────────────────────────────
                with st.expander("📋 Raw JSON Results"):
                    st.json(result)
