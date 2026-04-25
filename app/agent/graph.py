"""Compile the LangGraph workflow: router → retriever → prompt_builder → llm → validator."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.context import AgentContext
from app.agent.nodes import NodeBundle
from app.agent.state import GraphState


def build_graph(ctx: AgentContext):
    """
    Build and compile the stateful agent graph.

    Flow (strict): router → retriever → prompt_builder → llm → validator → END.
    """
    bundle = NodeBundle(ctx)
    g = StateGraph(GraphState)
    g.add_node("router", bundle.router_node)
    g.add_node("retriever", bundle.retriever_node)
    g.add_node("prompt_builder", bundle.prompt_builder_node)
    g.add_node("llm", bundle.llm_node)
    g.add_node("validator", bundle.validator_node)

    g.set_entry_point("router")
    g.add_edge("router", "retriever")
    g.add_edge("retriever", "prompt_builder")
    g.add_edge("prompt_builder", "llm")
    g.add_edge("llm", "validator")
    g.add_edge("validator", END)
    return g.compile()


__all__ = ["build_graph"]
