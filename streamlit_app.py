# streamlit_app.py
import os
import streamlit as st
from build_graph import build_graph
from langgraph.types import Command
from llm_helpers import paraphrase_question, clarify
import json
import io
import csv

st.set_page_config(page_title="Excel AI Interviewer", page_icon="🧠", layout="centered")

# Block the interviewer if the OpenAI API key is missing
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API key is not configured. Set OPENAI_API_KEY to use the interviewer.")
    st.info("Locally: export OPENAI_API_KEY=sk-...  |  Streamlit Cloud: add it under Secrets.")
    st.stop()

if "graph" not in st.session_state:
    graph, initial_state = build_graph()
    st.session_state.graph = graph
    st.session_state.config = {"configurable": {"thread_id": "excel_interview"}}
    st.session_state.result = graph.invoke({"state": initial_state}, config=st.session_state.config)
    st.session_state.last_meta = None
    st.session_state.chat = []  # per-question chat
    st.session_state.pending = False
    st.session_state.drafts = {}
    st.session_state.last_submitted_id = None

def render_interrupt(meta: dict):
    kind = meta.get("kind", "")
    if kind == "introduction":
        st.chat_message("assistant").write(meta.get("message", ""))
        if st.button("Continue", disabled=st.session_state.pending):
            st.session_state.result = st.session_state.graph.invoke(Command(resume="continue"), config=st.session_state.config)
            st.rerun()
    elif kind == "information_gathering":
        if "message" in meta: st.chat_message("assistant").write(meta["message"])
        prompt = meta.get("prompt", "> ")

        # Special-case the final proceed step
        if "Ready to begin the Excel questions?" in prompt:
            if st.button("Proceed", disabled=st.session_state.pending):
                st.session_state.result = st.session_state.graph.invoke(Command(resume="continue"), config=st.session_state.config)
                st.rerun()
            return

        form_key = f"ig_form_{hash(prompt)%10**8}"
        input_key = f"ig_input_{hash(prompt+'input')%10**8}"

        with st.form(key=form_key):
            ans = st.text_input(prompt, key=input_key)
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.result = st.session_state.graph.invoke(Command(resume=ans), config=st.session_state.config)
                st.rerun()
    elif kind in ("normal", "scenario"):
        st.session_state.last_meta = meta
        st.chat_message("assistant").write(meta.get("question", ""))
    elif kind == "conclusion":
        st.chat_message("assistant").write(meta.get("message", ""))
        recs = meta.get("recommendations", [])
        if recs:
            st.markdown("**Recommendations**")
            for i, r in enumerate(recs, 1): st.write(f"{i}. {r}")
        st.success("Interview complete.")

        # Exports
        export = meta.get("export", {}) or {}
        if export:
            st.markdown("---")
            st.markdown("**Download your results**")

            # JSON export
            json_bytes = json.dumps(export, indent=2).encode("utf-8")
            st.download_button(
                "Download full evaluation (JSON)",
                data=json_bytes,
                file_name="excel_interview_evaluation.json",
                mime="application/json",
            )

            # Topic competency CSV
            topic_comp = export.get("topic_competency", {}) or {}
            if topic_comp:
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(["topic","count","avg_tech","avg_soft","avg_tier","best_tier","correct","partial","incorrect","last_tech"])
                for topic, d in topic_comp.items():
                    vc = d.get("verdict_counts", {}) or {}
                    writer.writerow([
                        topic,
                        int(d.get("count", 0)),
                        float(d.get("avg_tech", 0.0)),
                        float(d.get("avg_soft", 0.0)),
                        float(d.get("avg_tier", 0.0)),
                        int(d.get("best_tier", 0)),
                        int(vc.get("correct", 0)),
                        int(vc.get("partial", 0)),
                        int(vc.get("incorrect", 0)),
                        float(d.get("last_tech", 0.0)),
                    ])
                st.download_button(
                    "Download topic competency (CSV)",
                    data=buf.getvalue(),
                    file_name="topic_competency.csv",
                    mime="text/csv",
                )

            # Events CSV (audit)
            events = export.get("events", []) or []
            if events:
                buf2 = io.StringIO()
                w2 = csv.writer(buf2)
                w2.writerow([
                    "turn","id","topic","tier","format","level_tag_at_ask",
                    "verdict","tech_score","soft_score","question_text","answer_text","why_next_question"
                ])
                for ev in events:
                    w2.writerow([
                        int(ev.get("turn", 0)),
                        ev.get("id",""),
                        ev.get("topic",""),
                        ev.get("tier",""),
                        ev.get("format",""),
                        ev.get("level_tag_at_ask",""),
                        ev.get("verdict",""),
                        float(ev.get("tech_score",0.0)),
                        float(ev.get("soft_score",0.0)),
                        (ev.get("question_text","") or "").replace("\n"," ").strip(),
                        (ev.get("answer_text","") or "").replace("\n"," ").strip(),
                        (ev.get("why_next_question","") or "").replace("\n"," ").strip(),
                    ])
                st.download_button(
                    "Download Q&A events (CSV)",
                    data=buf2.getvalue(),
                    file_name="events_audit.csv",
                    mime="text/csv",
                )

def asked_from_meta(meta: dict) -> dict:
    return {
        "id": meta.get("id"),
        "text": meta.get("question"),
        "topic": meta.get("topic"),
        "tier": meta.get("tier"),
        "level_tag": meta.get("level"),
        "question_format": meta.get("format"),
    }

# Handle current interrupt
intrs = st.session_state.result.get("__interrupt__")
if intrs:
    intr = intrs[0] if isinstance(intrs, list) else intrs
    meta = intr.value if hasattr(intr, "value") else intr
    render_interrupt(meta)

# Inline controls for current question
meta = st.session_state.last_meta or {}
if meta.get("kind") in ("normal", "scenario"):
    asked = asked_from_meta(meta)
    qid = asked.get("id") or "unknown"
    draft_key = f"draft_{qid}"
    if qid not in st.session_state.drafts:
        st.session_state.drafts[qid] = ""
    if draft_key not in st.session_state:
        st.session_state[draft_key] = st.session_state.drafts.get(qid, "")

    # Clarification buttons row
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Rephrase", use_container_width=True, disabled=st.session_state.pending):
            with st.spinner("Rephrasing..."):
                alt = paraphrase_question(asked, seed=f"rephrase|{len(st.session_state.chat)}")
            if alt:
                st.session_state.last_meta["question"] = alt
                st.chat_message("assistant").write(alt)
    with c2:
        if st.button("Hint", use_container_width=True, disabled=st.session_state.pending):
            with st.spinner("Getting hint..."):
                tip = clarify(asked, "Provide a brief hint (no solution).", chat_window=st.session_state.chat) \
                      or "Hint: List the steps clearly; mention relevant Excel functions or menu paths."
            st.session_state.chat.append({"role": "assistant", "content": tip})
            st.chat_message("assistant").write(tip)

    # Free-form clarification chat
    user_msg = st.chat_input("Ask a clarification (optional)")
    if user_msg and not st.session_state.pending:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        st.chat_message("user").write(user_msg)
        with st.spinner("Thinking..."):
            reply = clarify(asked, user_msg, chat_window=st.session_state.chat) \
                    or "Let me restate the goal: focus on the key idea; avoid full formulas."
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

    # Draft answer + single submit
    st.markdown("---")
    st.text_area("Your final answer", height=150, key=draft_key)
    st.session_state.drafts[qid] = st.session_state[draft_key]

    submit_col = st.columns([1, 3, 1])[1]
    with submit_col:
        disabled = st.session_state.pending
        if st.button("Submit answer", use_container_width=True, disabled=disabled):
            final = st.session_state[draft_key] or ""
            st.session_state.pending = True
            with st.spinner("Submitting..."):
                st.session_state.result = st.session_state.graph.invoke(Command(resume=final), config=st.session_state.config)
                st.session_state.chat = []
                st.session_state.last_meta = None
                st.session_state.last_submitted_id = qid
            st.session_state.pending = False
            st.rerun()