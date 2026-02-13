from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()


# declare a graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# create a tool
@tool
def add(num1: int, num2: int) -> int:
    """This is a function for adding 2 numbers"""
    return num1 + num2


@tool
def subtract(num1: int, num2: int) -> int:
    """This is a function for subtractng 2 numbers"""
    return num1 - num2


tools = [add, subtract]

# instantiate the OpenAI chat
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# create node
def agent_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are an AI assistant, that solves problems effectively"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# conditional node
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "continue_edge"
    else:
        return "end_edge"


# create graph
graph = StateGraph(AgentState)

# add node
graph.add_node("agent_node", agent_node)

# add tool node
tool_node = ToolNode(tools=tools)
graph.add_node("tools_node", tool_node)

# add edges
graph.add_edge(START, "agent_node")
graph.add_conditional_edges(
    "agent_node", should_continue, {"continue_edge": "tools_node", "end_edge": END}
)
graph.add_edge("tools_node", "agent_node")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 30 + 4 and also subtract 34 + 56")]}
print_stream(app.stream(input=inputs, stream_mode="values"))
