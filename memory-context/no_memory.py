from openai import OpenAI
import ollama


def chatbot(input: str, use_ollama: bool = False):
    try:
        if use_ollama:
            model_name = "llama3.2"

            response = ollama.chat(
                model=model_name, messages=[{"role": "user", "content": input}]
            )

            return response["message"]["content"]
        else:
            model_name = "gpt-4o-mini"
            client = OpenAI()

            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": input}]
            )

            return response.choices[0].message.content
    except Exception as e:
        print("An error occured: ", e)


def main():
    print("I am a chat bot with no memory!")
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

    has_begun_chat = False
    while True:
        if has_begun_chat:
            user_input = input("You: ")
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

        response = chatbot(input=user_input, use_ollama=use_ollama)
        print("\n Chatbot: ", response)
        has_begun_chat = True


main()
