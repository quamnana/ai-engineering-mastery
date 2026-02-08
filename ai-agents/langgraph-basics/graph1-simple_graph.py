from typing import TypedDict
from langgraph.graph import StateGraph


# create state to be used across the graph
class AgentState(TypedDict):
    message: str


# create a node: this is basically a Python function
def greeting_node(state: AgentState) -> AgentState:
    state["message"] = f"Hello {state['message']}! How are you doing today?"

    return state


# instantiate the StateGraph with the state
graph = StateGraph(AgentState)

# build the graph by adding nodes, start and end points
graph.add_node("greetor", greeting_node)
graph.set_entry_point("greetor")
graph.set_finish_point("greetor")

# compile the graph into a workflow
app = graph.compile()

# run the workflow
result = app.invoke({"message": "John"})
print(result["message"])
