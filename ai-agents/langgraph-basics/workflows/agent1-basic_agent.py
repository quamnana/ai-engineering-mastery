from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    message: List[HumanMessage]


llm = ChatOpenAI(model="gpt-4o")


def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"AI: {response.content} \n")
    return state


graph = StateGraph(AgentState)

graph.add_node("process_node", process_node)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

app = graph.compile()

user_input = input("User: ")
while user_input != "exit":
    response = app.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("User: ")
