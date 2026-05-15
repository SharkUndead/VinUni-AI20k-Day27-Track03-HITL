"""Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


async def recent_sessions() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   MAX(risk_level) AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 10
            """
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


with st.sidebar:
    st.header("Recent sessions")
    try:
        sessions = asyncio.run(recent_sessions())
    except Exception as e:
        sessions = []
        st.caption(f"No audit data yet: {e}")

    if not sessions:
        st.caption("No sessions yet.")
    for row in sessions:
        label = f"{row['thread_id'][:8]} - {row['worst_risk']} - {row['events']} events"
        if st.button(label, key=f"session_{row['thread_id']}", use_container_width=True):
            st.session_state.thread_id = row["thread_id"]
            st.session_state.pr_url = row["pr_url"]
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.rerun()
        st.caption(row["pr_url"])
        st.caption(f"last: {row['last_event']}")


with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


def render_approval_card(payload: dict) -> dict | None:
    """58-72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` - {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for idx, question in enumerate(payload.get("questions", []), start=1):
            answers[question] = st.text_input(question, key=f"escalation_q_{idx}")
        submitted_answers = st.form_submit_button("Submit answers", type="primary")
    if submitted_answers:
        return answers
    return None


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await graph.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await graph.ainvoke(Command(resume=resume_value), cfg)


if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR and asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result


payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url,
                st.session_state.thread_id,
                resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()


if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action in {"committed", "auto_committed"}:
        st.success(f"{action} - comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected - no comment posted")
    elif action.endswith("commit_failed"):
        st.error(f"{action} - GitHub comment was not posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(
        f"thread_id = {st.session_state.thread_id} | replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
