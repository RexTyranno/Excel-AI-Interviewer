# Excel AI Interviewer

An adaptive Excel skills interviewer built with Streamlit and LangGraph. It asks Excel questions tailored to a candidate’s level, paraphrases for clarity, scores answers using rubrics and an LLM, and concludes with personalized recommendations.

- **Tech**: Python, Streamlit, LangGraph, OpenAI API
- **Core modules**: `streamlit_app.py`, `build_graph.py`, `Interviewer.py`, `evaluator.py`, `planner.py`, `state.py`, `utils.py`
- **Content**: `config/config.yaml` (weights/policy), `questions/bank.yaml` (questions & scenarios)

## Quickstart

### Prerequisites

- Python 3.10+
- OpenAI API key with access to `gpt-4o-mini` (default) or your configured models

### Setup

```bash
git clone <this-repo>
cd AI-Interviewer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure environment

```bash
export OPENAI_API_KEY=sk-...        # required
# Optional overrides:
export EVALUATOR_MODEL=gpt-4o-mini   # scoring model
export PARAPHRASE_MODEL=gpt-4o-mini  # rephrasing model
export CLARIFY_MODEL=gpt-4o-mini     # clarification model
```

### Run

```bash
streamlit run streamlit_app.py
```

Streamlit will block with an error if `OPENAI_API_KEY` is missing.

## What it does

1. Introduces the interview and gathers basic candidate info.
2. Selects the next question based on level, topic coverage, and recent performance.
3. Paraphrases the question (optional) and provides hints on request.
4. Scores answers (technical + soft skills) using rubrics and an LLM-backed fallback.
5. Adapts difficulty, optionally branches into a scenario, and stops when coverage/policy criteria are met.
6. Produces a concise conclusion with recommendations.

## Architecture

- `streamlit_app.py`: UI flow and interaction handlers (render intro, IG prompts, question, hint, submit).
- `build_graph.py`: LangGraph state machine:
  - Nodes: introduction → information_gathering → interviewer/scenario → evaluator → conclusion
  - Uses `interrupt` to exchange data with the Streamlit UI.
- `Interviewer.py`: Stateless orchestrator; ties together `Planner` and `Evaluator`, builds asked questions, records answers, and produces conclusion text.
- `planner.py`: Chooses the next question by topic/tier/format with coverage and recent-topic cool-down logic.
- `evaluator.py`: LLM-assisted scoring with rubric guidance, soft skills EMA, coverage computation, streak/hysteresis for level changes, scenario/stop flags.
- `state.py`: Typed state schema, initialization, validation, and transient resets.
- `llm_helpers.py`: Paraphrasing and clarification helpers (graceful no-op if client/key missing; the app itself requires a key).
- `utils.py`: YAML loaders for config and question bank.

## Configuration

### Profile: `config/config.yaml`

Controls level topic weights, defaults by tier, thresholds, and policy.

Example (abbreviated):

```yaml
levels:
  beginner:
    topic_weights:
      Basic_Formulas: 0.35
      Sort_Filter: 0.20
      # ...
format_defaults_by_tier:
  "1": explain_then_example
  "2": step_by_step
  "3": diagnose_and_fix

thresholds:
  promote: 0.50
  demote: 0.10
  coverage_required: 0.70

policy:
  max_questions: 20
  min_questions_before_end: 10
  scenario_trigger:
    good_streak_at_level: 4
    fallback_after_questions: 8
  stop_after_scenario_if_coverage: 0.95
  confidence_mix:
    technical: 0.85
    soft: 0.15
  personalized_first_n: 3
```

- **Topic weights**: Drives coverage; higher weight topics are prioritized until covered.
- **format_defaults_by_tier**: If the bank item doesn’t specify a format, default by tier is used.
- **thresholds**: Coverage and promotion/demotion tuning.
- **policy**: Scenario triggers, end conditions, and early AI personalization.

## Authoring questions

Questions and scenarios are defined in `questions/bank.yaml`.

- The app splits items by `kind: question` vs `kind: scenario`.
- For questions, the planner indexes by `(level_tag, topic, tier)`.
- Rubrics improve scoring quality; they’re optional but recommended.

Example question item:

```yaml
- kind: question
  id: q_beg_basic_sum_1
  level_tag: beginner
  topic: Basic_Formulas
  tier: 1
  question_format: explain_then_example
  text: "Explain how SUM works and give one example using a cell range."
  rubric:
    correct:
      - "sum("
      - "range"
      - "example"
    partial:
      - "add"
      - "total"
    incorrect:
      - "vlookup"
```

Example scenario item:

```yaml
- kind: scenario
  id: sc_int_text_cleanup_1
  level_tag: intermediate
  topic: Text_Functions
  text: "A report has inconsistent spacing and casing; describe how you'd clean it up."
```

Notes:

- `id` must be unique.
- `topic` should match a configured topic for the relevant `level_tag`.
- `tier` is 1..3; higher tiers are harder.
- Optional `question_format` overrides the tier default.
- `rubric` lists are simple keyword cues for guided scoring and fallbacks.

## Customization

- Tune `config/config.yaml` weights and thresholds for a different curriculum or pacing.
- Add/edit items in `questions/bank.yaml` to extend coverage or difficulty.
- Override models via environment variables (see Quickstart).
- Adjust UI text and controls in `streamlit_app.py`.

## Deployment

- Streamlit Community Cloud:
  - Add the repo, set `OPENAI_API_KEY` under Secrets.
  - Command: `streamlit run streamlit_app.py`
- Self-host:
  - Use any Python host with environment variables set; expose Streamlit on your preferred port.

## Troubleshooting

- “OpenAI API key is not configured”:
  - Ensure `OPENAI_API_KEY` is set in the environment/Secrets and restart.
- Model errors/timeouts:
  - Confirm model names exist for your key; try `gpt-4o-mini` (default).
- No questions appear / early end:
  - Verify `questions/bank.yaml` items for the candidate’s `level_tag`, ensure unique `id`s, and that topics exist in `config/config.yaml`.

## Roadmap (ideas)

- Richer scenario branching and multi-turn hints
- Exportable report (PDF/Markdown) of interview summary
- Admin panel to review events/scores

## License

Specify your preferred license (e.g., MIT). If you want, I can add the text and a `LICENSE` file.
