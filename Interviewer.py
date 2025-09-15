from typing import Optional, Dict, Any, List, Tuple, cast
from datetime import datetime, timezone
import time

from state import (
    StateFactory, State, AskedQuestion, AnsweredQuestion, CandidateLevel,
    QuestionTier, QuestionFormat
)
from planner import Planner, BankIndex, BankItem
from evaluator import Evaluator
from utils import load_config_profile, load_questions, load_scenarios
from langgraph.types import interrupt
from llm_helpers import paraphrase_question


class Interviewer:
    """
    Stateless interview orchestrator. Holds config, bank index, planner, and evaluator.
    All methods accept a State and return updated State (and any outputs).
    """

    def __init__(self, *, rng_seed: Optional[int] = None):
        self.config: Dict[str, Any] = load_config_profile()

        # Build question bank index for planner + direct lookup by id
        items = load_questions()
        self._bank_index: BankIndex = self._build_bank_index(items)
        self._bank_by_id: Dict[str, BankItem] = {item["id"]: item for item in items}

        # Planner & Evaluator
        self.planner = Planner(config=self.config, rng_seed=rng_seed, bank_index=self._bank_index)
        self.evaluator = Evaluator(planner=self.planner, config=self.config)

        # Helpers
        self._factory = StateFactory()
        self._scenario_catalog = load_scenarios()

    # ---------- CX Methods (pure/state-in, state-out) ----------

    def get_introduction_message(self) -> str:
        """Generate a welcoming introduction message"""
        return """
🤖 Welcome to the AI Interview! 

I’ll be your interviewer today. This will be a fully automated session designed to get a clear picture of your Excel skills and give you some personalized feedback at the end.

📋 Here's how we will spend our time:
   1. We’ll start with a quick introduction and setup. (2 mins) 
   2. Then we will have a few questions related to your background and experience (3 mins)
   3. Following that, we will have some hands-on Excel questions (10-15 mins)
   4. Finally, we will wrap-up with personalized feedback (2 mins)

💡 Tips for success:
   • Answer naturally and try to explain your thinking process
   • It's better to say "I don't know" rather than guess
   • Take this as a chance to learn, not judgment!

Ready to discover your Excel strengths and growth areas?
        """.strip()

    def gather_candidate_information_interactive(self) -> Dict[str, Any]:
        """Interactively gather candidate information through prompts"""
        name = interrupt({
            "message": "Let's start with the basics:",
            "prompt": "What's your name?",
            "kind": "information_gathering"
        })

        years_str = interrupt({
            "prompt": f"Hi {name}! How many years of Excel experience do you have? (Enter a number, 0 if beginner)",
            "kind": "information_gathering"
        })
        try:
            years = max(0, int(float(str(years_str).strip())))
        except (ValueError, TypeError):
            years = 0

        domains_str = interrupt({
            "prompt": "What areas do you primarily use Excel for? (e.g., data analysis, reporting, budgeting, etc.)\nSeparate multiple areas with commas:",
            "kind": "information_gathering"
        })
        domains = [d.strip().lower() for d in str(domains_str).split(",") if d.strip()]

        level_str = interrupt({
            "prompt": "How would you rate your current Excel skill level?\n  1. Beginner (basic formulas, formatting)\n  2. Intermediate (IF statements, charts, basic analysis)\n  3. Advanced (VLOOKUP, PivotTables, complex formulas)\n  4. Expert (Power Query, advanced analytics, automation)\n\nEnter 1-4:",
            "kind": "information_gathering"
        })
        level_map = {"1": "beginner", "2": "intermediate", "3": "advanced", "4": "expert"}
        self_report_level = level_map.get(str(level_str).strip())

        experiences_text = interrupt({
            "prompt": "Briefly describe Excel-related tasks/projects you've done before.",
            "kind": "information_gathering"
        })
        experiences_text = str(experiences_text).strip()

        interrupt({
            "message": f"""
Perfect! Here's what I understand:
• Name: {name}
• Experience: {years} years  
• Focus areas: {', '.join(domains) if domains else 'General Excel use'}
• Self-assessment: {(self_report_level or 'intermediate').title()}
• Past experiences: {experiences_text or 'None provided'}

I'll start with {(self_report_level or 'intermediate')}-level questions and adapt based on your responses.
Remember: I'm here to help you learn, so don't worry about getting everything perfect!
            """.strip(),
            "prompt": "Ready to begin the Excel questions? (Press Enter to continue)",
            "kind": "information_gathering"
        })

        return {
            "name": str(name).strip(),
            "years_experience": years,
            "domains": domains,
            "experiences_text": experiences_text,
            "self_report_level": self_report_level
        }

    def start_with_info(self, state: State, candidate_info: Dict[str, Any]) -> State:
        """Initialize interview state with gathered candidate information"""
        new_state = self._factory.initialize(candidate_info)
        new_state["phase"] = "question_answering"
        return new_state

    def get_progress_info(self, state: State) -> Dict[str, Any]:
        """Get current progress information for display"""
        max_q = self.config.get("policy", {}).get("max_questions", 10)
        current_q = state.get("questions_asked", 0)
        level = state.get("level_estimation", "intermediate")
        coverage = state.get("topic_coverage", 0.0)
        return {
            "current_question": current_q,
            "estimated_total": max_q,
            "current_level": level,
            "coverage": coverage,
            "progress_percent": min(100, int((current_q / max_q) * 100)),
        }

    def render_question_text(self, asked: AskedQuestion, state: State) -> str:
        """Return paraphrased question text when available; fallback to original text."""
        seed = f"{asked['id']}|{state.get('questions_asked', 0)}"
        try:
            alt = paraphrase_question(asked, seed=seed)
        except Exception:
            alt = None
        return (alt or asked["text"]).strip()

    def generate_conclusion(self, state: State) -> Dict[str, Any]:
        """Generate comprehensive conclusion with insights and recommendations"""
        s = state
        candidate_name = s.get("candidate", {}).get("name", "")
        final_level = s.get("level_estimation", "intermediate")
        confidence = s.get("level_estimation_confidence", 0.0)
        coverage = s.get("topic_coverage", 0.0)
        questions_asked = s.get("questions_asked", 0)

        # Analyze performance by topic
        topic_scores = s.get("topic_scores", {})
        soft_scores = s.get("soft_skills_scores", {})

        strengths: List[str] = []
        improvements: List[str] = []

        for topic, score in topic_scores.items():
            if score >= 0.75:
                strengths.append(f"Strong {topic.replace('_', ' ').title()} skills")
            elif score < 0.50:
                improvements.append(f"{topic.replace('_', ' ').title()}")

        soft_avg = sum(soft_scores.values()) / max(len(soft_scores), 1) if soft_scores else 0.6
        if soft_avg >= 0.75:
            strengths.append("Clear communication and explanation skills")
        elif soft_avg < 0.60:
            improvements.append("Communication clarity (explaining your reasoning)")

        recommendations = self._generate_recommendations(final_level, improvements, topic_scores)

        summary = self._create_conclusion_summary(
            candidate_name, final_level, confidence, coverage, questions_asked, strengths, improvements
        )
        return {
            "summary": summary,
            "recommendations": recommendations,
            "final_level": final_level,
            "confidence": confidence,
            "coverage": coverage,
            "strengths": strengths,
            "improvements": improvements
        }

    def build_final_export(self, state: State) -> Dict[str, Any]:
        """
        Build a comprehensive final report including per-topic competency,
        soft skills breakdown, level trajectory, and event history.
        """
        s = state
        events = s.get("events", [])
        topic_scores = s.get("topic_scores", {})
        soft_scores = s.get("soft_skills_scores", {})

        # Aggregate per-topic competency
        topic_competency: Dict[str, Dict[str, Any]] = {}
        for i, ev in enumerate(events):
            asked = ev.get("asked", {}) or {}
            topic = asked.get("topic")
            if not topic:
                continue
            entry = topic_competency.setdefault(topic, {
                "count": 0,
                "avg_tech": 0.0,
                "avg_soft": 0.0,
                "avg_tier": 0.0,
                "best_tier": 0,
                "verdict_counts": {"correct": 0, "partial": 0, "incorrect": 0},
                "last_tech": None,
            })
            entry["count"] += 1
            entry["avg_tech"] += float(ev.get("tech_score", 0.0))
            entry["avg_soft"] += float(ev.get("soft_score", 0.0))
            entry["avg_tier"] += int(asked.get("tier", 1))
            entry["best_tier"] = max(int(entry["best_tier"]), int(asked.get("tier", 1)))
            verdict = str(ev.get("verdict", "partial"))
            if verdict in entry["verdict_counts"]:
                entry["verdict_counts"][verdict] += 1
            entry["last_tech"] = float(ev.get("tech_score", 0.0))

        for topic, agg in topic_competency.items():
            c = max(1, int(agg["count"]))
            agg["avg_tech"] = agg["avg_tech"] / c
            agg["avg_soft"] = agg["avg_soft"] / c
            agg["avg_tier"] = agg["avg_tier"] / c

        # Level trajectory
        level_trajectory = [
            {
                "turn": int(ev.get("turn", i + 1)),
                "level_before": ev.get("level_before"),
                "level_after": ev.get("level_after"),
            }
            for i, ev in enumerate(events)
        ]

        # Exportable event history (audit)
        exported_events = [
            {
                "turn": int(ev.get("turn", i + 1)),
                "id": (ev.get("asked", {}) or {}).get("id"),
                "question_text": (ev.get("asked", {}) or {}).get("text"),
                "topic": (ev.get("asked", {}) or {}).get("topic"),
                "tier": (ev.get("asked", {}) or {}).get("tier"),
                "format": (ev.get("asked", {}) or {}).get("question_format"),
                "level_tag_at_ask": (ev.get("asked", {}) or {}).get("level_tag"),
                "answer_text": (ev.get("answer", {}) or {}).get("text"),
                "verdict": ev.get("verdict"),
                "tech_score": float(ev.get("tech_score", 0.0)),
                "soft_score": float(ev.get("soft_score", 0.0)),
                "why_next_question": ev.get("why_next_question"),
            }
            for i, ev in enumerate(events)
        ]

        export: Dict[str, Any] = {
            "candidate": s.get("candidate", {}),
            "final": {
                "level": s.get("level_estimation", "intermediate"),
                "confidence": float(s.get("level_estimation_confidence", 0.0)),
                "topic_coverage": float(s.get("topic_coverage", 0.0)),
                "questions_asked": int(s.get("questions_asked", 0)),
            },
            "topic_scores": topic_scores,
            "topic_competency": topic_competency,
            "soft_skills": {
                "clarity": float(soft_scores.get("clarity", 0.0)),
                "framing": float(soft_scores.get("framing", 0.0)),
                "integrity": float(soft_scores.get("integrity", 0.0)),
                "overall": (
                    (float(soft_scores.get("clarity", 0.0))
                     + float(soft_scores.get("framing", 0.0))
                     + float(soft_scores.get("integrity", 0.0))) / 3.0
                    if soft_scores else 0.0
                ),
            },
            "level_trajectory": level_trajectory,
            "events": exported_events,
        }
        return export

    # ---------- Interview Flow (pure/state-in, state-out) ----------

    def ask_next(self, state: State) -> Tuple[AskedQuestion, State]:
        """
        Returns (AskedQuestion, updated_state). May select scenario or normal based on flags.
        Ensures next_question_spec is available for normal questions by invoking Evaluator if needed.
        """
        if state.get("should_end"):
            state["phase"] = "conclusion"
            raise RuntimeError("Interview already ended.")

        # Scenario routing (when pending and not done)
        if state.get("should_run_scenario") and not state.get("scenario_done"):
            asked = self._prepare_scenario_question(state)
            state = self._record_asked(state, asked)
            return asked, state

        # Ensure a next_question_spec exists (initial call or after resets)
        if not state.get("next_question_spec"):
            self.evaluator.evaluate_turn(state)
            if state.get("should_end"):
                state["phase"] = "conclusion"
                raise RuntimeError("No viable questions remain; interview ended.")
            if not state.get("next_question_spec"):
                raise RuntimeError("Planner did not yield a next question.")

        spec = cast(Dict[str, Any], state["next_question_spec"])
        bank_item = self._bank_by_id.get(spec["id"])
        if not bank_item:
            state["should_end"] = True
            state["phase"] = "conclusion"
            raise RuntimeError(f"Question id '{spec['id']}' not found in bank.")

        asked = self._mk_asked_from_bank(bank_item)
        state = self._record_asked(state, asked)
        return asked, state

    def record_answer(self, state: State, text: str) -> State:
        """Attach the candidate's answer to state."""
        now = datetime.now(timezone.utc)
        ans: AnsweredQuestion = {
            "text": text,
            "timestamp": now.isoformat(),
            "latency_seconds": 0.0,  # stateless default
        }
        state["last_answer"] = ans
        return state

    def evaluate(self, state: State, *, in_scenario_turn: bool = False) -> State:
        """
        Scores last Q/A and updates planning. Also marks scenario completion if applicable,
        transitions to conclusion if needed, and clears per-turn transients.
        """
        self.evaluator.evaluate_turn(state)

        if in_scenario_turn:
            state["scenario_done"] = True

        if state.get("should_end"):
            state["phase"] = "conclusion"

        # Clear per-turn fields
        state = self._factory.reset_transients(state)
        return state

    # ---------- Internals ----------

    def _record_asked(self, state: State, asked: AskedQuestion) -> State:
        state["last_question"] = asked
        state["phase"] = "question_answering"
        state["questions_asked"] = int(state.get("questions_asked", 0)) + 1
        return state

    def _mk_asked_from_bank(self, item: BankItem) -> AskedQuestion:
        now = datetime.now(timezone.utc)
        return AskedQuestion(
            id=item["id"],
            topic=item["topic"],
            tier=cast(QuestionTier, int(item["tier"])),
            question_format=cast(QuestionFormat, item["question_format"]),
            level_tag=cast(CandidateLevel, item["level_tag"]),
            text=item["text"],
            timestamp=now.isoformat(),
        )

    def _prepare_scenario_question(self, state: State) -> AskedQuestion:
        """
        Selects a scenario matching current level. Encodes it as an AskedQuestion
        (uses tier=2, question_format='diagnose_and_fix' to fit schema).
        """
        level = cast(CandidateLevel, state["level_estimation"])
        asked_ids = {ev["asked"]["id"] for ev in state.get("events", []) if ev.get("asked")}
        cand: Optional[Dict[str, Any]] = None

        for sc in self._scenario_catalog:
            if sc.get("level_tag") == level and sc.get("id") not in asked_ids:
                cand = sc
                break

        if not cand:
            state["scenario_done"] = True
            # Force a plan for a normal question
            self.evaluator.evaluate_turn(state)
            spec = cast(Dict[str, Any], state.get("next_question_spec") or {})
            bank_item = self._bank_by_id.get(spec.get("id", ""))
            if not bank_item:
                state["should_end"] = True
                state["phase"] = "conclusion"
                raise RuntimeError("No scenario and no viable normal question; interview ended.")
            return self._mk_asked_from_bank(bank_item)

        now = datetime.now(timezone.utc)
        return AskedQuestion(
            id=cand["id"],
            topic=cand.get("topic", "Scenario"),
            tier=cast(QuestionTier, 2),
            question_format=cast(QuestionFormat, "diagnose_and_fix"),
            level_tag=level,
            text=cand["text"],
            timestamp=now.isoformat(),
        )

    def _build_bank_index(self, items: List[BankItem]) -> BankIndex:
        idx: BankIndex = {}
        for it in items:
            key = (cast(CandidateLevel, it["level_tag"]), cast(str, it["topic"]), cast(QuestionTier, int(it["tier"])) )
            idx.setdefault(key, []).append(it)
        return idx

    # ---------- Conclusion helpers ----------

    def _generate_recommendations(self, level: str, improvements: List[str], topic_scores: Dict[str, float]) -> List[str]:
        recs: List[str] = []

        if level == "beginner":
            recs.extend([
                "Master basic formulas (SUM, AVERAGE, COUNT) in various scenarios",
                "Practice cell referencing (relative vs absolute) until it's automatic",
                "Learn efficient navigation shortcuts (Ctrl+Arrow keys, Ctrl+Home/End)"
            ])
        elif level == "intermediate":
            recs.extend([
                "Focus on conditional logic - practice nested IF statements",
                "Master text functions (LEFT, RIGHT, MID, CONCATENATE)",
                "Build comfort with date/time calculations and formatting"
            ])
        elif level == "advanced":
            recs.extend([
                "Deepen VLOOKUP skills and learn XLOOKUP alternatives",
                "Create complex PivotTables with calculated fields",
                "Practice error handling with IFERROR and troubleshooting"
            ])
        elif level == "expert":
            recs.extend([
                "Master Power Query for data transformation workflows",
                "Build advanced dashboard solutions with slicers and dynamic ranges",
                "Explore What-If Analysis tools (Goal Seek, Solver, Data Tables)"
            ])

        weak_topics = [topic for topic, score in topic_scores.items() if score < 0.60]
        for topic in weak_topics[:2]:
            topic_display = topic.replace('_', ' ').title()
            recs.append(f"Strengthen {topic_display} through hands-on practice")
        return recs[:5]

    def _create_conclusion_summary(self, name: str, level: str, confidence: float,
                                   coverage: float, questions: int, strengths: List[str],
                                   improvements: List[str]) -> str:
        confidence_text = "high" if confidence >= 0.8 else "moderate" if confidence >= 0.6 else "initial"
        coverage_text = "comprehensive" if coverage >= 0.8 else "solid" if coverage >= 0.6 else "foundational"

        summary = f"""
🎉 Great job, {name}! Interview complete.

📊 Your Excel Assessment Results:
   • Skill Level: {level.title()} (with {confidence_text} confidence)
   • Topic Coverage: {coverage_text} ({coverage:.0%} of {level} topics explored)
   • Questions Completed: {questions}

💪 Your Excel Strengths:"""
        if strengths:
            for strength in strengths[:3]:
                summary += f"\n   • {strength}"
        else:
            summary += f"\n   • Solid foundational knowledge at {level} level"

        if improvements:
            summary += f"\n\n📈 Growth Opportunities:"
            for improvement in improvements[:3]:
                summary += f"\n   • {improvement}"

        summary += f"\n\n✨ You're well-positioned at the {level} level! Keep building on these strengths while working on the growth areas above."
        return summary.strip()