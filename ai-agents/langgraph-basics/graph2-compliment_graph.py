from typing import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    message: str


def compliment_node(state: AgentState) -> AgentState:
    """This function changes the state and returns the changes"""
    state["message"] = (
        f"{state['name']}, you are doing amazing learning AI Agents and LangGraph"
    )

    return state


graph = StateGraph(AgentState)

# build the graph by adding nodes, start and end points
graph.add_node("complimentor", compliment_node)
graph.set_entry_point("complimentor")
graph.set_finish_point("complimentor")

# compile the graph into a workflow
app = graph.compile()

# run the workflow
result = app.invoke({"name": "John"})
print(result["message"])
