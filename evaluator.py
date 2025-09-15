from typing import Dict, Optional, Tuple, List, cast, Any
from state import (
    State, Event, CandidateLevel, KnowledgeVerdict,
    AskedQuestion, AnsweredQuestion, NextQuestionSpec
)
from planner import Planner
from utils import load_questions
import os
import json

try:
    from openai import OpenAI  # ChatGPT client
except Exception:
    OpenAI = None  # type: ignore


class Evaluator:
    """
    Scores the last Q/A, updates state (scores, coverage, streaks, level),
    appends an Event, decides flags (scenario/stop), and uses Planner to set
    the next_question_spec when appropriate.
    """

    def __init__(self, planner: Planner, config: Dict[str, Any]):
        self.planner = planner
        self.config = config

        threshold_cfg = self.config.get("thresholds", {})
        policy_cfg = self.config.get("policy", {})

        self.promote_thr: float = float(threshold_cfg.get("promote", 0.70))
        self.demote_thr: float = float(threshold_cfg.get("demote", 0.60))
        self.coverage_req: float = float(threshold_cfg.get("coverage_required", 0.70))

        self.good_streak_req: int = int(policy_cfg.get("scenario_trigger", {}).get("good_streak_at_level", 2))
        self.scenario_fallback_after: int = int(policy_cfg.get("scenario_trigger", {}).get("fallback_after_questions", 6))
        self.stop_after_scenario_if_cov: float = float(policy_cfg.get("stop_after_scenario_if_coverage", 0.80))
        self.max_questions: Optional[int] = policy_cfg.get("max_questions")
        self.min_questions_before_end: int = int(policy_cfg.get("min_questions_before_end", 0))

        confidence_mix = policy_cfg.get("confidence_mix", {"technical": 0.85, "soft": 0.15})
        self.conf_w_tech: float = float(confidence_mix.get("technical", 0.85))
        self.conf_w_soft: float = float(confidence_mix.get("soft", 0.15))

        # Number of early questions to personalize via AI (fallbacks to planner on failure)
        self.personalized_first_n: int = int(policy_cfg.get("personalized_first_n", 3))

        # LLM client setup
        self._chat_model = os.environ.get("EVALUATOR_MODEL", "gpt-4o-mini")
        self._chat_client = None
        if OpenAI is not None:
            try:
                self._chat_client = OpenAI()
            except Exception:
                self._chat_client = None

        # Cache rubric index: id -> rubric dict {correct|partial|incorrect: [phrases]}
        self._rubrics_by_id: Dict[str, Dict[str, List[str]]] = {}
        self._build_rubric_index()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def evaluate_turn(self, state: State) -> None:
        """
        Main entrypoint called after Interviewer (or Scenario) records last Q/A.
        Mutates `state` in-place:
          - scores the last answer
          - updates topic scores, coverage, streak, and level
          - appends an Event
          - sets flags (should_run_scenario, should_end)
          - plans the next_question_spec via Planner (unless ending or scenario)
        """
        # Guard: nothing to score (e.g., intro)
        if not state.get("last_question") or not state.get("last_answer"):
            state["next_question_spec"] = self._maybe_plan_next(state)
            return

        level_before: CandidateLevel = cast(CandidateLevel, state["level_estimation"])
        asked: AskedQuestion = cast(AskedQuestion, state["last_question"])
        answer: AnsweredQuestion = cast(AnsweredQuestion, state["last_answer"])
        topic: str = asked["topic"]

        # 1) Score answer (LLM with rubric guidance; robust fallback)
        verdict, tech_score = self._score_answer(answer_text=answer["text"], asked=asked)

        # 2) Soft skills (LLM-guided scoring + EMA updates)
        soft_score = self._score_soft_skills(state=state, answer_text=answer["text"], asked=asked)

        # 3) Update topic scores (EMA)
        self._update_topic_scores(state, topic, tech_score)

        # 4) Recompute coverage for current level
        state["topic_coverage"] = self._compute_topic_coverage(state)

        # 5) Update good streak
        if verdict in ("correct", "partial") and tech_score >= 0.70:
            state["good_streak_at_level"] = state.get("good_streak_at_level", 0) + 1
        else:
            state["good_streak_at_level"] = 0

        # 6) Level rollup + hysteresis (promote/demote)
        level_after = self._apply_level_hysteresis(state)

        # 7) Confidence + short reasoning
        level_rollup_score = self._compute_level_rollup(state)  # technical rollup for current level
        soft_mean = self._soft_mean(state)
        state["level_estimation_confidence"] = (
            self.conf_w_tech * float(level_rollup_score) + self.conf_w_soft * float(soft_mean)
        )
        state["level_estimation_reasoning"] = self._mk_reasoning(state, topic, verdict, tech_score, level_rollup_score)

        # 8) Flags: scenario + stop
        self._update_flags(state)

        # 9) Append Event
        self._append_event(
            state=state,
            verdict=verdict,
            tech_score=tech_score,
            soft_score=soft_score,
            level_before=level_before,
            level_after=cast(CandidateLevel, state["level_estimation"]),
            why_next=self._mk_why_next(state, topic, verdict, tech_score)
        )

        # 10) Plan next (unless ending or scenario)
        state["next_question_spec"] = self._maybe_plan_next(state)

    # ---------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------

    def _score_answer(self, answer_text: str, asked: AskedQuestion) -> Tuple[KnowledgeVerdict, float]:
        """
        Uses ChatGPT with rubric guidance to compute verdict and technical score.
        Fallback mapping if the client/key is unavailable or the model output is invalid:
            correct=1.0, partial=0.6, incorrect=0.0
        """
        text = (answer_text or "").strip()
        if not text:
            return cast(KnowledgeVerdict, "incorrect"), 0.0

        rubric = self._rubrics_by_id.get(asked.get("id", ""), {})
        llm_result = self._llm_score_with_rubric(asked=asked, answer_text=text, rubric=rubric)

        if llm_result:
            try:
                verdict = str(llm_result.get("verdict", "partial"))
                if verdict not in ("correct", "partial", "incorrect"):
                    verdict = "partial"
                tech_score = float(llm_result.get("tech_score", 0.6))
                tech_score = max(0.0, min(1.0, tech_score))
                return cast(KnowledgeVerdict, verdict), tech_score
            except Exception:
                pass  # fall through to heuristic fallback

        # Heuristic fallback (light rubric keyword matching)
        verdict = "incorrect"
        tech_score = 0.0
        if rubric:
            correct_hits = self._count_hits(text.lower(), rubric.get("correct", []))
            partial_hits = self._count_hits(text.lower(), rubric.get("partial", []))
            incorrect_hits = self._count_hits(text.lower(), rubric.get("incorrect", []))
            if correct_hits > 0 and incorrect_hits == 0:
                verdict, tech_score = "correct", 1.0
            elif partial_hits > 0 or correct_hits > 0:
                verdict, tech_score = "partial", 0.6
            else:
                verdict, tech_score = "incorrect", 0.0
        else:
            # No rubric → use a generic LLM scoring fallback instead of hardcoded verdicts
            llm_generic = self._llm_score_generic(asked=asked, answer_text=text)
            if llm_generic:
                try:
                    verdict = str(llm_generic.get("verdict", "partial"))
                    if verdict not in ("correct", "partial", "incorrect"):
                        verdict = "partial"
                    tech_score = float(llm_generic.get("tech_score", 0.6))
                    tech_score = max(0.0, min(1.0, tech_score))
                    return cast(KnowledgeVerdict, verdict), tech_score
                except Exception:
                    pass
            # Ultimate conservative fallback if LLM unavailable or invalid output
            return cast(KnowledgeVerdict, "incorrect"), 0.0

        return cast(KnowledgeVerdict, verdict), float(tech_score)

    def _score_soft_skills(self, state: State, answer_text: str, asked: AskedQuestion) -> float:
        """
        Uses ChatGPT to score soft skills (clarity, framing, integrity) and updates
        state['soft_skills_scores'] via EMA; returns an overall soft score in 0..1.
        Graceful fallback to neutral score if client/key unavailable.
        """
        text = (answer_text or "").strip()
        if not text:
            return 0.0

        rubric = self._rubrics_by_id.get(asked.get("id", ""), {})
        llm_result = self._llm_score_with_rubric(asked=asked, answer_text=text, rubric=rubric)

        if llm_result:
            soft_scores = llm_result.get("soft_scores", {})
            # Expected keys: clarity, framing, integrity (0..1)
            if isinstance(soft_scores, dict):
                self._update_soft_scores(state, {
                    "clarity": float(soft_scores.get("clarity", 0.6)),
                    "framing": float(soft_scores.get("framing", 0.6)),
                    "integrity": float(soft_scores.get("integrity", 0.6)),
                })
            overall = float(llm_result.get("soft_score", 0.6))
            return max(0.0, min(1.0, overall))

        # Fallback: neutral
        self._update_soft_scores(state, {"clarity": 0.6, "framing": 0.6, "integrity": 0.6})
        return 0.6

    # ---------------------------------------------------------------------
    # State math helpers
    # ---------------------------------------------------------------------

    def _update_topic_scores(self, state: State, topic: str, tech_score: float) -> None:
        """
        Topic EMA: new = alpha*tech + (1-alpha)*old, alpha=0.5
        """
        alpha = 0.5
        cur = float(state.get("topic_scores", {}).get(topic, 0.0))
        new = alpha * float(tech_score) + (1.0 - alpha) * cur
        state["topic_scores"][topic] = new

    def _compute_topic_coverage(self, state: State) -> float:
        """
        Computes coverage of current level's topics with a depth-aware measure:
          - A topic counts as covered if it has at least two interactions OR
            its average technical score across interactions is >= 0.60.
          coverage = sum(weight[t] for covered topics) / sum(weights)
        """
        level = cast(CandidateLevel, state["level_estimation"])
        weights = self._level_weights(level)
        if not weights:
            return 0.0

        # Gather technical scores per topic from past events
        scores_by_topic: Dict[str, List[float]] = {}
        for ev in state.get("events", []):
            asked = ev.get("asked") or {}
            topic = asked.get("topic")
            if not topic:
                continue
            tech = float(ev.get("tech_score", 0.0))
            scores_by_topic.setdefault(topic, []).append(tech)

        def is_topic_covered(topic: str, scores: List[float]) -> bool:
            if len(scores) >= 2:
                return True
            if not scores:
                return False
            return (sum(scores) / max(1, len(scores))) >= 0.60

        covered_weight = 0.0
        for topic, scores in scores_by_topic.items():
            if topic in weights and is_topic_covered(topic, scores):
                covered_weight += float(weights.get(topic, 0.0))

        total_weight = sum(weights.values()) or 1.0
        return max(0.0, min(1.0, covered_weight / total_weight))

    def _compute_level_rollup(self, state: State) -> float:
        """
        Σ weight_t * topic_scores[t] over current level topics.
        Missing topic_scores default to 0.0.
        """
        level = cast(CandidateLevel, state["level_estimation"])
        weights = self._level_weights(level)
        if not weights:
            return 0.0
        scores = state.get("topic_scores", {})
        return sum(float(weights[t]) * float(scores.get(t, 0.0)) for t in weights.keys())

    def _apply_level_hysteresis(self, state: State) -> CandidateLevel:
        """
        Applies promote/demote/stay and mutates state['level_estimation'] if needed.
        Uses rollup score for promotion but separate logic for demotion.
        """
        level = cast(CandidateLevel, state["level_estimation"])
        level_rollup_score = self._compute_level_rollup(state)
        coverage = state.get("topic_coverage", 0.0)
        
        # Promotion based on rollup score
        promote = level_rollup_score >= self.promote_thr and coverage >= self.coverage_req
        
        # Demotion based on recent performance, not rollup score
        recent_events = state.get("events", [])[-3:]  # Look at last 3 questions
        recent_scores = [float(ev.get("tech_score", 0.0)) for ev in recent_events 
                        if ev.get("asked", {}).get("level_tag") == level]
        
        # Only consider demotion if we have enough data at current level
        can_demote = len(recent_scores) >= 2
        demote = can_demote and sum(recent_scores) / max(1, len(recent_scores)) < self.demote_thr
        
        new_level = level
        if demote:
            new_level = self._level_step(level, -1)
        elif promote:
            new_level = self._level_step(level, +1)
        
        if new_level != level:
            state["level_estimation"] = new_level
            state["good_streak_at_level"] = 0
            # Recompute coverage under new level's topic set
            state["topic_coverage"] = self._compute_topic_coverage(state)
        
        return cast(CandidateLevel, state["level_estimation"])

    def _update_flags(self, state: State) -> None:
        """
        Sets should_run_scenario and should_end based on policy + current state.
        """
        # Scenario flag
        if not state.get("scenario_done", False):
            streak_ok = state.get("good_streak_at_level", 0) >= self.good_streak_req
            fallback_ok = state.get("questions_asked", 0) >= self.scenario_fallback_after
            state["should_run_scenario"] = bool(streak_ok or fallback_ok)
        else:
            state["should_run_scenario"] = False

        # End flag
        should_end = False
        if self.max_questions is not None and state.get("questions_asked", 0) >= int(self.max_questions):
            should_end = True
        if state.get("scenario_done", False):
            if state.get("questions_asked", 0) >= int(self.min_questions_before_end):
                if state.get("topic_coverage", 0.0) >= self.stop_after_scenario_if_cov:
                    should_end = True

        # You can add more guards (e.g., no viable next question) in _maybe_plan_next.
        state["should_end"] = should_end

    def _append_event(
        self,
        state: State,
        verdict: KnowledgeVerdict,
        tech_score: float,
        soft_score: float,
        level_before: CandidateLevel,
        level_after: CandidateLevel,
        why_next: str,
    ) -> None:
        ev: Event = {
            "turn": len(state.get("events", [])) + 1,
            "asked": cast(AskedQuestion, state["last_question"]),
            "answer": cast(AnsweredQuestion, state["last_answer"]),
            "tech_score": float(tech_score),
            "soft_score": float(soft_score),
            "verdict": verdict,
            "level_before": level_before,
            "level_after": level_after,
            "why_next_question": why_next,
        }
        state["events"].append(ev)

    # ---------------------------------------------------------------------
    # Planning integration
    # ---------------------------------------------------------------------

    def _maybe_plan_next(self, state: State) -> Optional[NextQuestionSpec]:
        """
        Returns a NextQuestionSpec or None. Also sets should_end=True if no viable item remains.
        Skips planning when a scenario is pending.
        """
        if state.get("should_end"):
            return None
        if state.get("should_run_scenario") and not state.get("scenario_done"):
            return None

        # AI-personalized planning for the first N questions
        try:
            if int(state.get("questions_asked", 0)) < int(self.personalized_first_n):
                ai_spec = self._ai_plan_next(state)
                if ai_spec is not None:
                    return ai_spec
        except Exception:
            # Safety: never fail the interview due to AI planning errors
            pass

        spec = self.planner.plan_next_question(state)
        if spec is None:
            # No viable item left → end
            state["should_end"] = True
        return spec

    def _ai_plan_next(self, state: State) -> Optional[NextQuestionSpec]:
        """
        Ask the LLM to choose topic/tier/format for the next question given the candidate profile
        and current interview context. Then map to an unused bank item. Gracefully returns None
        if unavailable or invalid.
        """
        if not self._chat_client or not os.environ.get("OPENAI_API_KEY", ""):
            return None

        level = state.get("level_estimation")
        weights = self._level_weights(cast(CandidateLevel, level))
        if not weights:
            return None

        candidate = state.get("candidate", {})
        asked_ids = {ev["asked"]["id"] for ev in state.get("events", []) if ev.get("asked")}

        sys_prompt = (
            "You are an expert Excel interviewer. Given the candidate profile and current context, "
            "choose the next most relevant topic and difficulty. For the first few questions, bias "
            "toward their past experiences (experiences_text) but maintain level-weighted coverage "
            "and avoid repeating the same topic back-to-back. Return STRICT JSON: "
            '{"topic": "<one of the configured topics>", "tier": 1|2|3, '
            '"question_format": "explain_then_example|step_by_step|diagnose_and_fix|compare_and_choose"}.'
        )

        user_payload = {
            "candidate": {
                "name": candidate.get("name"),
                "years_experience": candidate.get("years_experience"),
                "domains": candidate.get("domains", []),
                "experiences_text": candidate.get("experiences_text", ""),
                "self_report_level": candidate.get("self_report_level"),
            },
            "level": level,
            "available_topics_with_weights": weights,
            "recent_events": [
                {
                    "topic": ev.get("asked", {}).get("topic"),
                    "tier": ev.get("asked", {}).get("tier"),
                    "verdict": ev.get("verdict"),
                    "tech_score": ev.get("tech_score"),
                } for ev in state.get("events", [])[-3:]
            ],
            "policy": {
                "goal": "personalize early but maintain coverage; avoid immediate repeats"
            }
        }

        try:
            resp = self._chat_client.chat.completions.create(
                model=self._chat_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            data = self._extract_json(content) or {}
            topic = str(data.get("topic", "")).strip()
            tier_val = int(data.get("tier", 1))
            fmt = str(data.get("question_format", "step_by_step")).strip()
        except Exception:
            return None

        # Validate outputs
        if topic not in weights:
            return None
        if tier_val not in (1, 2, 3):
            tier_val = 1
        if fmt not in ("explain_then_example", "step_by_step", "diagnose_and_fix", "compare_and_choose"):
            fmt = "step_by_step"

        # Map to a concrete unused bank item; backoff across tiers if needed
        from typing import cast as _cast
        from state import QuestionTier as _QT, QuestionFormat as _QF, CandidateLevel as _CL
        chosen = (
            self.planner._select_item(_cast(_CL, level), topic, _cast(_QT, tier_val), asked_ids)
            or self.planner._select_item(_cast(_CL, level), topic, _cast(_QT, 2), asked_ids)
            or self.planner._select_item(_cast(_CL, level), topic, _cast(_QT, 1), asked_ids)
            or self.planner._select_item(_cast(_CL, level), topic, _cast(_QT, 3), asked_ids)
        )
        if not chosen:
            return None

        return NextQuestionSpec(
            id=chosen["id"],
            topic=topic,
            tier=_cast(_QT, int(chosen.get("tier", tier_val))),
            question_format=_cast(_QF, fmt or chosen.get("question_format") or "step_by_step"),
            level_tag=_cast(_CL, level),
        )

    # ---------------------------------------------------------------------
    # Small utilities
    # ---------------------------------------------------------------------

    def _level_weights(self, level: CandidateLevel) -> Dict[str, float]:
        levels = self.config.get("levels", {})
        return levels.get(level, {}).get("topic_weights", {}) or {}

    def _level_step(self, level: CandidateLevel, delta: int) -> CandidateLevel:
        order = ["beginner", "intermediate", "advanced", "expert"]
        i = max(0, min(len(order) - 1, order.index(level) + delta))
        return cast(CandidateLevel, order[i])

    def _soft_mean(self, state: State) -> float:
        s = state.get("soft_skills_scores", {})
        if not s:
            return 0.0
        vals = [float(v) for v in s.values()]
        return sum(vals) / max(1, len(vals))

    def _mk_reasoning(self, state: State, topic: str, verdict: KnowledgeVerdict, tech_score: float, level_rollup_score: float) -> str:
        return (
            f"{topic}: {verdict} ({tech_score:.2f}). "
            f"Level rollup={level_rollup_score:.2f}, coverage={state.get('topic_coverage', 0.0):.2f}."
        )

    def _mk_why_next(self, state: State, topic: str, verdict: KnowledgeVerdict, tech_score: float) -> str:
        # Keep it short & audit-friendly
        if verdict == "incorrect" or tech_score < 0.60:
            return f"Weak signal on {topic}; reinforcing or downshifting tier."
        return f"Improving coverage; selecting next gap topic at current level."

    # ---------------------------------------------------------------------
    # LLM helpers + rubric index
    # ---------------------------------------------------------------------

    def _build_rubric_index(self) -> None:
        try:
            for item in load_questions():
                qid = item.get("id")
                rubric = item.get("rubric") or {}
                if qid and isinstance(rubric, dict):
                    # Normalize lists
                    self._rubrics_by_id[qid] = {
                        "correct": [str(x).lower() for x in rubric.get("correct", [])],
                        "partial": [str(x).lower() for x in rubric.get("partial", [])],
                        "incorrect": [str(x).lower() for x in rubric.get("incorrect", [])],
                    }
        except Exception:
            # If load_questions fails, keep index empty
            self._rubrics_by_id = {}

    def _count_hits(self, text_lower: str, phrases: List[str]) -> int:
        hits = 0
        for p in phrases or []:
            if p and p in text_lower:
                hits += 1
        return hits

    def _llm_score_with_rubric(self, asked: AskedQuestion, answer_text: str, rubric: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
        """
        Single-call LLM scoring for both technical and soft skills, guided by rubric.
        Returns a dict with fields: verdict, tech_score, soft_score, soft_scores{clarity,framing,integrity}.
        None if client/key unavailable or request fails.
        """
        if not self._chat_client:
            return None
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None

        # Compose structured instructions
        sys_prompt = (
            "You are an expert technical interviewer. "
            "Evaluate the candidate's answer against the provided question and rubric. "
            "Return STRICT JSON with fields: "
            '{"verdict": "correct|partial|incorrect", '
            '"tech_score": 0..1, '
            '"soft_score": 0..1, '
            '"soft_scores": {"clarity": 0..1, "framing": 0..1, "integrity": 0..1}, '
            '"rationale": "one short sentence"} '
            "The tech_score should reflect rubric alignment; soft_score reflects clarity, framing, and integrity."
        )

        user_payload = {
            "question": {
                "id": asked.get("id"),
                "topic": asked.get("topic"),
                "tier": asked.get("tier"),
                "level_tag": asked.get("level_tag"),
                "format": asked.get("question_format"),
                "text": asked.get("text"),
            },
            "rubric": rubric or {},
            "answer": answer_text,
            "instructions": {
                "verdict_scale": ["correct", "partial", "incorrect"],
                "tech_score_range": [0.0, 1.0],
                "soft_keys": ["clarity", "framing", "integrity"],
                "soft_score_is_mean": True,
            },
        }

        try:
            resp = self._chat_client.chat.completions.create(
                model=self._chat_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            data = self._extract_json(content)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _llm_score_generic(self, asked: AskedQuestion, answer_text: str) -> Optional[Dict[str, Any]]:
        """
        LLM scoring without a rubric. Evaluates correctness vs the question content alone.
        Returns dict with fields: verdict, tech_score (0..1), soft_score, soft_scores, rationale.
        None if client/key unavailable or request fails.
        """
        if not self._chat_client:
            return None
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None

        sys_prompt = (
            "You are an expert technical interviewer. Evaluate the candidate's answer "
            "for correctness against the question. No rubric is provided—use general domain "
            "knowledge and the question text. Return STRICT JSON with fields: "
            '{"verdict": "correct|partial|incorrect", '
            '"tech_score": 0..1, '
            '"soft_score": 0..1, '
            '"soft_scores": {"clarity": 0..1, "framing": 0..1, "integrity": 0..1}, '
            '"rationale": "one short sentence"}. '
            "Penalize answers that convey uncertainty like 'I don't know' or 'not sure' as incorrect with low tech_score."
        )

        user_payload = {
            "question": {
                "id": asked.get("id"),
                "topic": asked.get("topic"),
                "tier": asked.get("tier"),
                "level_tag": asked.get("level_tag"),
                "format": asked.get("question_format"),
                "text": asked.get("text"),
            },
            "answer": answer_text,
            "instructions": {
                "verdict_scale": ["correct", "partial", "incorrect"],
                "tech_score_range": [0.0, 1.0],
                "soft_keys": ["clarity", "framing", "integrity"],
                "soft_score_is_mean": True,
            },
        }

        try:
            resp = self._chat_client.chat.completions.create(
                model=self._chat_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            data = self._extract_json(content)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to parse JSON from the model output. Handles fenced blocks.
        """
        s = text.strip()
        # Strip markdown fences if present
        if s.startswith("```"):
            s = s.strip("`")
            # Remove possible language tags
            if "\n" in s:
                s = s.split("\n", 1)[1]
        try:
            return json.loads(s)
        except Exception:
            # Try to find first {...} block
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end+1])
                except Exception:
                    return None
        return None

    def _update_soft_scores(self, state: State, scores: Dict[str, float]) -> None:
        """
        EMA update for soft skills keys in state['soft_skills_scores'].
        """
        alpha = 0.5
        store = state.get("soft_skills_scores", {})
        for k, v in scores.items():
            cur = float(store.get(k, 0.0))
            store[k] = alpha * float(v) + (1.0 - alpha) * cur
        state["soft_skills_scores"] = store