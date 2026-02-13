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

document_content = ""


# declare a graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# create a tool
@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


tools = [save, update]

# instantiate the OpenAI chat
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# create node
def agent_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """
    )

    user_input = input("User: ")
    human_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [human_message]

    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": list(state["messages"]) + [human_message, response]}


# conditional node
def should_continue(state: AgentState):
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    if not messages:
        return "continue_edge"

    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end_edge"  # goes to the end edge which leads to the endpoint

    return "continue_edge"


# create graph
graph = StateGraph(AgentState)

# add node
graph.add_node("agent_node", agent_node)

# add tool node
tool_node = ToolNode(tools=tools)
graph.add_node("tools_node", tool_node)

# add edges
graph.add_edge(START, "agent_node")
graph.add_edge("agent_node", "tools_node")

graph.add_conditional_edges(
    "tools_node", should_continue, {"continue_edge": "agent_node", "end_edge": END}
)


app = graph.compile()


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


def run_document_agent():
    print("\n ===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()


# Write an email to a client requesting for the user requirements for a project
# The client name is John and my name is Nana, company is WebDevs and contact is webdevs@gmail.com. I am the founder
