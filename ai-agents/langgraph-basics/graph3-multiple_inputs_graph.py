from typing import TypedDict
from langgraph.graph import StateGraph
import math


class AgentState(TypedDict):
    name: str
    operation: str
    values: list
    result: str


def multi_operation_node(state: AgentState) -> AgentState:
    """This function returns results based on different operations"""

    if state["operation"] == "+":
        state["result"] = f"Hi {state['name']}, your answer is: {sum(state['values'])}"
    elif state["operation"] == "*":
        state["result"] = (
            f"Hi {state['name']}, your answer is: {math.prod(state['values'])}"
        )
    else:
        state["result"] = f"Hi {state['name']}, you input an invalid operation"

    return state


graph = StateGraph(AgentState)

# build the graph by adding nodes, start and end points
graph.add_node("operator", multi_operation_node)
graph.set_entry_point("operator")
graph.set_finish_point("operator")

# compile the graph into a workflow
app = graph.compile()

# run the workflow
result = app.invoke({"name": "John", "values": [1, 2, 3, 4], "operation": "/"})
print(result["result"])
