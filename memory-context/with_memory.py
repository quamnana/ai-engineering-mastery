import json
from openai import OpenAI
import ollama


def create_initial_message():
    # initialize a messages array (context window) with the system prompt
    messages = [
        {
            "role": "system",
            "content": "Your name is Fred, you are a smart assistant, that has the ability to answer question briefly and informatively.",
        }
    ]

    return messages


def chatbot(user_input, messages, use_ollama=False):
    try:

        # append the user's input to the messages array (context window)
        messages.append({"role": "user", "content": user_input})

        if use_ollama:
            model_name = "llama3.2"

            # query the LLM to get your response.
            response = ollama.chat(model=model_name, messages=messages)

            chatbot_response = response["message"]["content"]

            # append the AI's response to the messages array (context window)
            messages.append({"role": "assistant", "content": chatbot_response})

            # automatically save messages to file (memory)
            save_conversation(messages)

            return chatbot_response
        else:
            model_name = "gpt-4o-mini"
            client = OpenAI()

            response = client.chat.completions.create(
                model=model_name, messages=messages
            )

            chatbot_response = response.choices[0].message.content

            messages.append({"role": "assistant", "content": chatbot_response})

            return chatbot_response
    except Exception as e:
        print("An error occured: ", e)


# using the summarization strategy to save token and also not fill up the context window
def summarize_messages(messages):
    last_messages = messages[-5:]  # last 5 messages
    summarization = " ".join([msg["content"][:50] for msg in last_messages])
    summary = f"Previous summarized conersations: {summarization}"
    return [{"role": "system", "content": summary}] + last_messages


def save_conversation(messages, file_name="./conversations.json"):
    try:
        with open(file_name, "w") as f:
            json.dump(messages, f)
    except Exception as e:
        print("An error occured when saving file: ", e)


def load_conversation(file_name="./conversations.json"):
    try:
        with open(file_name, "r") as f:
            return json.load(f)
    except Exception as e:
        print("An error occured when loading file: ", e)
        create_initial_message()


def main():
    print("Hi! I am Fred, a chat bot with memory.")
    print("\n Select a model")
    print("1. OpenAI gpt-4o-mini")
    print("2. Ollama -- local model (Llama3.2)")

    # model selection
    while True:
        model_choice = input("Select your choice of model (1 or 2): ")
        if model_choice in ["1", "2"]:
            break
        print("Please enter either 1 or 2")

    use_ollama = model_choice == "2"

    print("Chat session has begun...")
    print("Enter 'q' to quit")
    print("Enter 'c' to clear screen")
    print("Enter 'sm' to summarize messages")
    print("Enter 'sc' to save converation")
    print("Enter 'lc' to load conversation")

    messages = create_initial_message()

    has_begun_chat = False
    while True:
        if has_begun_chat:
            user_input = input("\nYou: ")
        else:
            user_input = input("What is on you mind today?: ")

        if not user_input:
            has_begun_chat = True
            continue

        if user_input.lower() == "q":
            print("Talk to you later!")
            break

        if user_input.lower() == "c":
            print("\033c", end="")
            continue

        if user_input.lower() == "sm":
            messages = summarize_messages(messages)
            print("Conversation has been summarized!")
            continue

        if user_input.lower() == "sc":
            save_conversation(messages)
            print("Conversation is saved!")
            continue

        if user_input.lower() == "lc":
            messages = load_conversation()
            print("Conversation is loaded!")
            continue

        response = chatbot(user_input, messages, use_ollama=use_ollama)
        print("\n Chatbot: ", response)
        has_begun_chat = True

        # Automatically summarize if conversation gets too long
        if len(messages) > 10:
            messages = summarize_messages(messages)
            print("\n(Conversation automatically summarized)")


main()
