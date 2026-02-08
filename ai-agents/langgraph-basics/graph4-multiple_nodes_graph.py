from typing import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    age: int
    skills: list
    message: str


def name_node(state: AgentState) -> AgentState:
    state["message"] = f"Hello {state['name']} welcome to our app."
    return state


def age_node(state: AgentState) -> AgentState:
    state["message"] = f"{state['message']} You are {state['age']} years old."
    return state


def skills_node(state: AgentState) -> AgentState:
    state["message"] = (
        f"{state['message']} You have skills in: {', '.join(state['skills'])}"
    )
    return state


graph = StateGraph(AgentState)

graph.add_node("name_node", name_node)
graph.add_node("age_node", age_node)
graph.add_node("skills_node", skills_node)

graph.set_entry_point("name_node")
graph.add_edge("name_node", "age_node")
graph.add_edge("age_node", "skills_node")
graph.set_finish_point("skills_node")

app = graph.compile()

results = app.invoke(
    {"name": "John", "age": 30, "skills": ["Python", "FastAPI", "LangGraph"]}
)

print(results["message"])
