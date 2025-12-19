from transformers import pipeline, AutoTokenizer

model_name = "distilgpt2"


# creates a simple LLM using a small GPT-2 model
def create_simple_llm():
    generator = pipeline("text-generation", model=model_name, pad_token_id=50256)

    return generator


def generate_text(generator, prompt, max_length=100):
    response = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
    )
    generated_text = response[0]["generated_text"]

    return generated_text


def run_llm_demo():
    print("This is a demo to show basic text generation with a small LLM")

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Python is a programming language",
    ]

    print("Loading LLM....")
    generator = create_simple_llm()

    for prompt in prompts:
        print("Prompt: ", prompt)
        generated_text = generate_text(generator, prompt)
        print("Generated Text: ", generated_text)
        input("\n Press 'enter' to continue")


# allows users to input their own prompts to the llm
def interactive_demo():
    generator = create_simple_llm()

    print("Interactive Demo.. type your prompt or 'q' to quit")

    while True:
        prompt = input("\n Enter your prompt: ")
        if prompt.lower() == "q":
            break

        generated_text = generate_text(generator, prompt)
        print("Generated Text: ", generated_text)


# explains the LLM process with a simple example
def explain_process():
    print("\n🎓 How it works:")
    print("1. Input text → Tokenization → Numbers")
    print("2. Numbers → Model Processing → Prediction")
    print("3. Prediction → New Token → Output Text")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Hello World!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print("\n📝 Example Tokenization:")
    print(f"Original text: '{text}'")
    print(f"As tokens (numbers): {tokens}")
    print(f"Decoded back: '{decoded}'")


if __name__ == "__main__":
    print("Choose a demo:")
    print("1. Run basic demonstration")
    print("2. Interactive mode")
    print("3. Explain the process")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        run_llm_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        explain_process()
    else:
        print("You chose none of the options. Exiting.....")
