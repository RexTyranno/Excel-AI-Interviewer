from typing import TypedDict, cast
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from Interviewer import Interviewer
from state import State, AskedQuestion, StateFactory


class GraphState(TypedDict):
    state: State


def build_graph():
    interviewer = Interviewer(rng_seed=None)
    factory = StateFactory()

    # Build initial state (single source of truth lives in the graph)
    initial_state = factory.initialize({
        "name": "",
        "years_experience": 0,
        "domains": [],
        "experiences_text": "",  # new
        "self_report_level": None,
    })

    def introduction_node(g: GraphState) -> GraphState:
        s = g["state"]
        interrupt({
            "message": interviewer.get_introduction_message(),
            "kind": "introduction",
            "action": "Press Enter to continue..."
        })
        s["phase"] = "information_gathering"
        g["state"] = s
        return g

    def information_gathering_node(g: GraphState) -> GraphState:
        s = g["state"]
        candidate_info = interviewer.gather_candidate_information_interactive()
        s = interviewer.start_with_info(s, candidate_info)
        g["state"] = s
        return g

    def interviewer_node(g: GraphState) -> GraphState:
        s = g["state"]
        if s.get("should_end"):
            g["state"] = s
            return g
        if s.get("should_run_scenario") and not s.get("scenario_done"):
            return g  # routed by conditional edges

        asked: AskedQuestion
        asked, s = interviewer.ask_next(s)

        progress_info = interviewer.get_progress_info(s)
        qtext = interviewer.render_question_text(asked, s)
        answer_text = interrupt({
            "id": asked["id"],
            "question": qtext,
            "topic": asked["topic"],
            "level": asked["level_tag"],
            "tier": asked["tier"],
            "format": asked["question_format"],
            "kind": "normal",
            "progress": progress_info
        })
        s = interviewer.record_answer(s, cast(str, answer_text))
        g["state"] = s
        return g

    def scenario_node(g: GraphState) -> GraphState:
        s = g["state"]
        asked, s = interviewer.ask_next(s)

        progress_info = interviewer.get_progress_info(s)
        qtext = interviewer.render_question_text(asked, s)
        answer_text = interrupt({
            "id": asked["id"],
            "question": qtext,
            "topic": asked["topic"],
            "level": asked["level_tag"],
            "tier": asked["tier"],
            "format": asked["question_format"],
            "kind": "scenario",
            "progress": progress_info
        })
        s = interviewer.record_answer(s, cast(str, answer_text))
        # Mark that the upcoming evaluation is for a scenario turn
        s["__scenario_turn__"] = True
        g["state"] = s
        return g

    def evaluator_node(g: GraphState) -> GraphState:
        s = g["state"]
        in_scenario = bool(s.pop("__scenario_turn__", False))
        s = interviewer.evaluate(s, in_scenario_turn=in_scenario)
        g["state"] = s
        return g

    def conclusion_node(g: GraphState) -> GraphState:
        s = g["state"]
        data = interviewer.generate_conclusion(s)
        export = interviewer.build_final_export(s)
        interrupt({
            "message": data["summary"],
            "recommendations": data["recommendations"],
            "final_level": data["final_level"],
            "confidence": data["confidence"],
            "coverage": data["coverage"],
            "export": export,
            "kind": "conclusion"
        })
        s["phase"] = "conclusion"
        s["should_end"] = True
        g["state"] = s
        return g

    def route_from_interviewer(g: GraphState):
        s = g["state"]
        if s.get("should_end"):
            return "conclusion"
        if s.get("should_run_scenario") and not s.get("scenario_done"):
            return "scenario"
        return "evaluator"

    def route_from_evaluator(g: GraphState):
        s = g["state"]
        if s.get("should_end"):
            return "conclusion"
        return "interviewer"

    workflow = StateGraph(GraphState)
    workflow.add_node("introduction", introduction_node)
    workflow.add_node("information_gathering", information_gathering_node)
    workflow.add_node("interviewer", interviewer_node)
    workflow.add_node("scenario", scenario_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("conclusion", conclusion_node)

    workflow.set_entry_point("introduction")
    workflow.add_edge("introduction", "information_gathering")
    workflow.add_edge("information_gathering", "interviewer")

    workflow.add_conditional_edges(
        "interviewer",
        route_from_interviewer,
        {
            "scenario": "scenario",
            "evaluator": "evaluator",
            "conclusion": "conclusion",
        },
    )
    workflow.add_edge("scenario", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        route_from_evaluator,
        {
            "interviewer": "interviewer",
            "conclusion": "conclusion",
        },
    )
    workflow.add_edge("conclusion", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # Return graph and the initial state (single source of truth)
    return graph, initial_state