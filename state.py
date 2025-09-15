from typing import List, Dict, TypedDict, Literal, Optional
from copy import deepcopy

Phase = Literal[
    "introduction",
    "information_gathering", 
    "question_answering",
    "conclusion"
]

CandidateLevel = Literal[
    "beginner",
    "intermediate",
    "advanced",
    "expert"
]

KnowledgeVerdict = Literal[
    "correct",
    "partial",
    "incorrect"
]

# explain_then_example → candidate explains the concept, then gives a formula.
# step_by_step → candidate outlines sequential steps.
# diagnose_and_fix → candidate analyzes a broken case and proposes a correction.
# compare_and_choose → candidate compares multiple options and picks the best one.
QuestionFormat = Literal[
    "explain_then_example",
    "step_by_step", 
    "diagnose_and_fix",
    "compare_and_choose"
]

# Tier 1 → basic concept or simple use case.
# Tier 2 → multi-step, edge-case, or more complex scenario.
# Tier 3 → diagnostic, failure mode, or advanced "what if" question.
QuestionTier = Literal[1, 2, 3]


class Candidate(TypedDict):
    name: str
    years_experience: int
    domains: List[str]
    experiences_text: str  # single free-text field
    self_report_level: Optional[CandidateLevel]

class NextQuestionSpec(TypedDict, total=False):
    id: str
    topic: str
    tier: QuestionTier
    question_format: QuestionFormat
    level_tag: CandidateLevel


class AskedQuestion(TypedDict):
    id: str
    topic: str
    tier: QuestionTier
    question_format: QuestionFormat
    level_tag: CandidateLevel
    text: str
    timestamp: str  # ISO-8601

class AnsweredQuestion(TypedDict):
    text: str
    timestamp: str  # ISO-8601
    latency_seconds: float # Time taken to answer the question

class Event(TypedDict):
    turn: int
    asked: AskedQuestion
    answer: Optional[AnsweredQuestion]
    tech_score: float
    soft_score: float
    verdict: KnowledgeVerdict
    level_before: CandidateLevel
    level_after: CandidateLevel
    why_next_question: str

class State(TypedDict):
    # phase of the interview
    phase: Phase

    # candidate context
    candidate: Candidate

    # question tracking
    questions_asked: int
    topic_coverage: float
    good_streak_at_level: int
    topic_scores: Dict[str, float]
    soft_skills_scores: Dict[str, float]

    # last question and answer
    last_question: Optional[AskedQuestion]
    last_answer: Optional[AnsweredQuestion]

    # level estimation
    level_estimation: CandidateLevel
    level_estimation_confidence: float
    level_estimation_reasoning: str

    # control flags
    scenario_done: bool
    should_run_scenario: bool
    should_end: bool

    # next question spec
    next_question_spec: Optional[NextQuestionSpec]

    # events
    events: List[Event]


class StateFactory:
    """Creates, normalizes, and validates interview State objects. Pure, no internal session storage."""

    def __init__(self):
        pass  # keep factory stateless

    # ---------- Public API ----------

    def normalize_candidate(self, candidate: Candidate) -> Candidate:
        years = max(0, int(candidate.get("years_experience", 0)))
        # normalize domains: strip, lowercase, drop empties, dedupe (stable)
        seen: set = set()
        norm_domains: List[str] = []
        for d in candidate.get("domains", []) or []:
            v = str(d).strip().lower()
            if v and v not in seen:
                seen.add(v)
                norm_domains.append(v)

        # normalize experiences: strip, lowercase, drop empties, dedupe (stable)
        experiences_text = str(candidate.get("experiences_text", "")).strip()

        # Handle self_report_level validation - ensure it's a valid level or None
        self_report = candidate.get("self_report_level")
        valid_levels = ["beginner", "intermediate", "advanced", "expert"]
        if self_report not in valid_levels:
            self_report = None

        return {
            "name": str(candidate.get("name", "")).strip(),
            "years_experience": years,
            "domains": norm_domains,
            "experiences_text": experiences_text,
            "self_report_level": self_report,
        }

    def initialize(
        self,
        candidate: Candidate,
        *,
        prior_level: Optional[CandidateLevel] = None
    ) -> State:
        """Build a fresh State with sane defaults; no side-effects."""
        c = self.normalize_candidate(candidate)
        
        # Smarter level initialization based on experience and self-report
        start_level: CandidateLevel = self._determine_starting_level(c, prior_level)

        state: State = {
            "phase": "introduction",
            "candidate": c,
            "questions_asked": 0,
            "topic_coverage": 0.0,
            "good_streak_at_level": 0,
            "last_question": None,
            "last_answer": None,
            "level_estimation": start_level,
            "level_estimation_confidence": self._initial_confidence(c, start_level),
            "level_estimation_reasoning": self._initial_reasoning(c, start_level),
            "topic_scores": {},
            "soft_skills_scores": {
                "clarity": 0.0,
                "framing": 0.0,
                "integrity": 0.0,
            },
            "scenario_done": False,
            "should_run_scenario": False,
            "should_end": False,
            "next_question_spec": None,
            "events": [],
        }
        self.validate_state(state)
        return state

    def _determine_starting_level(self, candidate: Candidate, prior_level: Optional[CandidateLevel]) -> CandidateLevel:
        """Smarter starting level determination"""
        if prior_level:
            return prior_level
            
        # Use self-report if available
        if candidate.get("self_report_level"):
            return candidate["self_report_level"]
        
        # Fall back to experience-based heuristic
        years = candidate.get("years_experience", 0)
        if years == 0:
            return "beginner"
        elif years <= 2:
            return "intermediate" 
        elif years <= 5:
            return "advanced"
        else:
            return "expert"

    def _initial_confidence(self, candidate: Candidate, level: CandidateLevel) -> float:
        """Set initial confidence based on how we determined the level"""
        if candidate.get("self_report_level"):
            # Higher confidence if user self-reported
            return 0.6
        else:
            # Lower confidence if we guessed from experience
            return 0.4

    def _initial_reasoning(self, candidate: Candidate, level: CandidateLevel) -> str:
        """Generate initial reasoning explanation"""
        if candidate.get("self_report_level"):
            return f"Started at {level} based on self-assessment"
        else:
            years = candidate.get("years_experience", 0)
            return f"Started at {level} based on {years} years of experience"

    def validate_state(self, state: State) -> None:
        # Required keys sanity
        required_keys = [
            "phase", "candidate", "questions_asked", "topic_coverage",
            "good_streak_at_level", "level_estimation",
            "level_estimation_confidence", "topic_scores",
            "soft_skills_scores", "scenario_done",
            "should_run_scenario", "should_end", "events"
        ]
        missing = [k for k in required_keys if k not in state]
        if missing:
            raise ValueError(f"State missing keys: {missing}")

        # Ranges
        tc = state["topic_coverage"]
        conf = state["level_estimation_confidence"]
        if not (0.0 <= tc <= 1.0):
            raise ValueError(f"topic_coverage out of range: {tc}")
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"level_estimation_confidence out of range: {conf}")
        if state["questions_asked"] < 0 or state["good_streak_at_level"] < 0:
            raise ValueError("Counters must be non-negative")

        # Phase/Level literals basic check (lightweight)
        valid_phases = ["introduction","information_gathering","question_answering","conclusion"]
        if state["phase"] not in valid_phases:
            raise ValueError(f"Invalid phase: {state['phase']}")
        
        valid_levels = ["beginner","intermediate","advanced","expert"]
        if state["level_estimation"] not in valid_levels:
            raise ValueError(f"Invalid level_estimation: {state['level_estimation']}")

    def reset_transients(self, state: State) -> State:
        """Clear per-turn fields safely; returns a shallow copy with transients reset."""
        s = deepcopy(state)
        s["last_question"] = None
        s["last_answer"] = None
        s["next_question_spec"] = None
        return s