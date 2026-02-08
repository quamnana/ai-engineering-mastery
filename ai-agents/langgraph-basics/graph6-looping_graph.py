from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import random


class AgentState(TypedDict):
    player_name: str
    intro_message: str
    higher_bound: int
    lower_bound: int
    guesses: list[int]
    max_guesses: int
    attempts: int
    correct_guess: int
    current_guess: int


def setup_node(state: AgentState) -> AgentState:
    state["intro_message"] = (
        f"Hello {state['player_name']}, welcome to the guessing game"
    )
    state["correct_guess"] = random.randint(state["lower_bound"], state["higher_bound"])
    state["attempts"] = 0
    state["max_guesses"] = 7
    state["guesses"] = []

    return state


def guess_node(state: AgentState) -> AgentState:
    print("GUESSING...")
    state["current_guess"] = random.randint(state["lower_bound"], state["higher_bound"])
    state["guesses"].append(state["current_guess"])
    state["attempts"] = state["attempts"] + 1

    if state["current_guess"] < state["correct_guess"]:
        state["lower_bound"] = state["current_guess"]

    if state["current_guess"] > state["correct_guess"]:
        state["higher_bound"] = state["current_guess"]

    return state


def hint_node(state: AgentState) -> AgentState:
    if state["attempts"] == state["max_guesses"]:
        print("Sorry! You have maxed out your guesses")
        return "exit_edge"

    if state["current_guess"] == state["correct_guess"]:
        print(f"Correct! your guess is right: {state['current_guess']}")
        return "exit_edge"

    if state["current_guess"] < state["correct_guess"]:
        print(f"Current guess is lower: {state['current_guess']}")
        return "loop_edge"

    if state["current_guess"] > state["correct_guess"]:
        print(f"Current guess is higher: {state['current_guess']}")
        return "loop_edge"


graph = StateGraph(AgentState)

graph.add_node("setup_node", setup_node)
graph.add_node("guess_node", guess_node)
graph.add_node("hint_node", lambda state: state)

graph.add_edge(START, "setup_node")
graph.add_edge("setup_node", "guess_node")
graph.add_edge("guess_node", "hint_node")
graph.add_conditional_edges(
    "hint_node", hint_node, {"loop_edge": "guess_node", "exit_edge": END}
)

app = graph.compile()

initial_state = AgentState(
    player_name="John", lower_bound=1, higher_bound=20, guesses=[]
)
result = app.invoke({"player_name": "John", "lower_bound": 1, "higher_bound": 20})

print(result)
