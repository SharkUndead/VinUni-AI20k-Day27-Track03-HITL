"""Shared types: graph state, structured PR analysis, and audit entries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


Decision = Literal["auto_approve", "human_approval", "escalate"]
HumanChoice = Literal["approve", "reject", "edit"]


# Lab-calibrated thresholds. AUTO_APPROVE is high so the demo can exercise HITL
# even when hosted models over-report confidence on simple PRs. ESCALATE stays
# near the original handout value; high-risk security PRs are also caught by the
# route heuristics in the exercise graph nodes.
AUTO_APPROVE_THRESHOLD = 0.90
ESCALATE_THRESHOLD = 0.58


def risk_level_for(confidence: float) -> str:
    """Map confidence to risk_level for AuditEntry."""
    if confidence >= AUTO_APPROVE_THRESHOLD:
        return "low"
    if confidence < ESCALATE_THRESHOLD:
        return "high"
    return "med"


class ReviewComment(BaseModel):
    """A single review comment the agent proposes."""

    file: str = Field(description="Path of the file the comment is about")
    line: int | None = Field(None, description="Line number, when known")
    severity: Literal["nit", "suggestion", "issue", "blocker"]
    body: str


class PRAnalysis(BaseModel):
    """LLM-structured output of the analyzer node."""

    summary: str = Field(description="One-paragraph description of what the PR does")
    risk_factors: list[str] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence that the review is complete and correct",
    )
    confidence_reasoning: str = Field(description="Why the model picked that confidence value")
    escalation_questions: list[str] = Field(
        default_factory=list,
        description="Specific questions to ask a human reviewer if escalating",
    )


class AuditEntry(BaseModel):
    """One row of the structured audit trail."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event was recorded (UTC).",
    )
    agent_id: str = Field(description="Identifier of the agent that produced the event.")
    action: str = Field(
        description="What the agent did at this step: fetch_pr, analyze, route, "
        "human_approval, escalate, synthesize, commit."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Current confidence at this step.",
    )
    risk_level: str = Field(
        description="Derived from confidence: low, med, or high. Use risk_level_for()."
    )
    reviewer_id: str | None = Field(
        default=None,
        description="GitHub username of the human reviewer for HITL events.",
    )
    decision: str = Field(
        description="Outcome at this step: auto, approve, reject, edit, escalate, or pending."
    )
    reason: str | None = Field(
        default=None,
        description="Free-text explanation: confidence reasoning, human feedback, etc.",
    )
    execution_time_ms: int = Field(
        ge=0,
        description="Wall-clock duration of the action in milliseconds.",
    )


class ReviewState(TypedDict, total=False):
    """LangGraph state: every node reads and updates this dict."""

    # Inputs
    pr_url: str
    thread_id: str

    # Populated by fetch_pr
    pr_title: str
    pr_author: str
    pr_diff: str
    pr_files: list[str]
    pr_head_sha: str

    # Populated by analyze
    analysis: PRAnalysis

    # Populated by route_by_confidence
    decision: Decision

    # Populated by HITL nodes
    human_choice: HumanChoice | None
    human_feedback: str | None
    escalation_answers: dict[str, str] | None

    # Populated by commit/final nodes
    posted_comment_body: str | None
    final_action: str | None
