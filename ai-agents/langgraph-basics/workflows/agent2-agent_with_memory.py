from typing import List, TypedDict, Union
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(model="gpt-4o")


def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response.content)  # add AI response to messages
    print(f"AI: {response.content} \n")
    return state


graph = StateGraph(AgentState)

graph.add_node("process_node", process_node)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

app = graph.compile()

chat_history = []
user_input = input("User: ")
while user_input != "exit":
    chat_history.append(HumanMessage(content=user_input))
    response = app.invoke({"messages": chat_history})
    chat_history = response["messages"]
    user_input = input("User: ")
