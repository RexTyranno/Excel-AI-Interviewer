from typing import Dict, List, Optional, Tuple, Set, TypedDict, cast, Any
import random
from state import (
    State,
    NextQuestionSpec,
    CandidateLevel,
    QuestionTier,
    QuestionFormat,
    Event,
)


class BankItem(TypedDict, total=False):
    id: str
    level_tag: CandidateLevel
    topic: str
    tier: QuestionTier
    question_format: QuestionFormat
    text: str

BankIndex = Dict[Tuple[CandidateLevel, str, QuestionTier], List[BankItem]]

class Planner:
    """Essential, typed question planner: topic → tier → format → unused item."""

    def __init__(
        self,
        config: Dict[str, Any],
        rng_seed: Optional[int] = None,
        bank_index: Optional[BankIndex] = None,
    ):
        self._rng = random.Random(rng_seed)
        self._bank_index: BankIndex = bank_index or {}
        self._config: Dict[str, Any] = config

    def plan_next_question(self, state: State) -> Optional[NextQuestionSpec]:
        # Scenario in progress → planner yields no normal question
        if state.get("should_run_scenario") and not state.get("scenario_done"):
            return None

        level: CandidateLevel = state["level_estimation"]
        weights = self._get_level_weights(level)
        if not weights:
            return None

        asked_ids = self._asked_ids(state)
        last_by_topic = self._last_event_by_topic(state)

        # Simple recent-topic cool-down: avoid topics from the last 2 turns if possible
        recent_topics: List[str] = []
        for ev in state.get("events", [])[-2:]:
            asked = ev.get("asked") or {}
            t = asked.get("topic")
            if t:
                recent_topics.append(str(t))
        topics_ordered = self._topics_by_gap(state, weights)
        topics_iter = [t for t in topics_ordered if t not in set(recent_topics)] or topics_ordered

        for topic in topics_iter:
            tier = self._choose_tier(topic, last_by_topic)
            for t in self._tier_backoff(tier):
                item = self._select_item(level, topic, t, asked_ids)
                if item:
                    fmt = self._choose_format(t, item.get("question_format"))
                    return NextQuestionSpec(
                        id=item["id"],
                        topic=topic,
                        tier=t,
                        question_format=fmt,
                        level_tag=level,
                    )
        return None

    # ---------- internals ----------

    def _get_level_weights(self, level: CandidateLevel) -> Dict[str, float]:
        levels = self._config.get("levels", {})
        level_cfg = levels.get(level, {})
        return level_cfg.get("topic_weights", {}) or {}

    def _asked_ids(self, state: State) -> Set[str]:
        ids: Set[str] = set()
        for ev in state.get("events", []):
            asked = ev.get("asked")
            if asked and asked.get("id"):
                ids.add(asked["id"])
        return ids

    def _last_event_by_topic(self, state: State) -> Dict[str, Event]:
        res: Dict[str, Event] = {}
        for ev in state.get("events", []):
            asked = ev.get("asked")
            if asked and asked.get("topic"):
                res[asked["topic"]] = ev
        return res

    def _topics_by_gap(self, state: State, weights: Dict[str, float]) -> List[str]:
        topic_scores = state.get("topic_scores", {})
        recency: Dict[str, int] = {}
        for idx, ev in enumerate(state.get("events", [])):
            aq = ev.get("asked")
            if aq and aq.get("topic"):
                recency[aq["topic"]] = idx

        def key(topic: str):
            score = float(topic_scores.get(topic, 0.0))
            weight = float(weights.get(topic, 0.0))
            gap = (1.0 - score) * weight
            last_idx = recency.get(topic, -1)
            return (gap, weight, -last_idx)

        topics = list(weights.keys())
        self._rng.shuffle(topics)
        topics.sort(key=key, reverse=True)
        return topics

    def _choose_tier(self, topic: str, last_by_topic: Dict[str, Event]) -> QuestionTier:
        last_ev = last_by_topic.get(topic)
        if not last_ev:
            return cast(QuestionTier, 1)
        asked = last_ev.get("asked") or {}
        last_tier = int(asked.get("tier", 1))
        verdict = last_ev.get("verdict", "partial")
        tech = float(last_ev.get("tech_score", 0.0))
        if verdict in ("correct", "partial") and tech >= 0.70:
            return cast(QuestionTier, min(3, last_tier + 1))
        if verdict == "incorrect" or tech < 0.60:
            return cast(QuestionTier, max(1, last_tier - 1))
        return cast(QuestionTier, last_tier)

    def _choose_format(
        self,
        tier: QuestionTier,
        override: Optional[QuestionFormat],
    ) -> QuestionFormat:
        if override:
            return override
        defaults = self._config.get("format_defaults_by_tier", {})
        fmt = defaults.get(str(tier))
        return cast(QuestionFormat, fmt or "step_by_step")

    def _select_item(
        self,
        level: CandidateLevel,
        topic: str,
        tier: QuestionTier,
        asked_ids: Set[str],
    ) -> Optional[BankItem]:
        pool = self._bank_index.get((level, topic, int(tier)), [])
        for item in pool:
            iid = item.get("id")
            if iid and iid not in asked_ids:
                return item
        return None

    def _tier_backoff(self, primary: QuestionTier) -> List[QuestionTier]:
        if primary == 1:
            return [1, 2, 3]
        if primary == 2:
            return [2, 1, 3]
        return [3, 2, 1]

