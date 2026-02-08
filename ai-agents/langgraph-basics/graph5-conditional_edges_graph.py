from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    num1: int
    num2: int
    num3: int
    num4: int
    final_ans1: int
    final_ans2: int
    operator1: int
    operator2: int


def addition_node1(state: AgentState) -> AgentState:
    state["final_ans1"] = state["num1"] + state["num2"]
    return state


def substraction_node1(state: AgentState) -> AgentState:
    state["final_ans1"] = state["num1"] - state["num2"]
    return state


def router_node1(state: AgentState) -> AgentState:
    if state["operator1"] == "+":
        return "addition_edge1"
    elif state["operator1"] == "-":
        return "substraction_edge1"


def addition_node2(state: AgentState) -> AgentState:
    state["final_ans2"] = state["num3"] + state["num4"]
    return state


def substraction_node2(state: AgentState) -> AgentState:
    state["final_ans2"] = state["num3"] - state["num4"]
    return state


def router_node2(state: AgentState) -> AgentState:
    if state["operator2"] == "+":
        return "addition_edge2"
    elif state["operator2"] == "-":
        return "substraction_edge2"


graph = StateGraph(AgentState)

# add nodes
graph.add_node("addition_node1", addition_node1)
graph.add_node("addition_node2", addition_node2)
graph.add_node("substraction_node1", substraction_node1)
graph.add_node("substraction_node2", substraction_node2)
graph.add_node("router_node1", lambda state: state)  # pass through function
graph.add_node("router_node2", lambda state: state)

# add edges
graph.add_edge(START, "router_node1")
graph.add_conditional_edges(
    "router_node1",
    router_node1,
    {"addition_edge1": "addition_node1", "substraction_edge1": "substraction_node1"},
)  # {edge_name: node_name}
graph.add_edge("addition_node1", "router_node2")
graph.add_edge("substraction_node1", "router_node2")
graph.add_conditional_edges(
    "router_node2",
    router_node2,
    {"addition_edge2": "addition_node2", "substraction_edge2": "substraction_node2"},
)
graph.add_edge("addition_node2", END)
graph.add_edge("substraction_node2", END)

app = graph.compile()

initial_state = AgentState(num1=2, num2=4, num3=5, num4=9, operator1="+", operator2="-")

result = app.invoke(initial_state)
print(result)
