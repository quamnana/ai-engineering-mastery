from transformers import pipeline, AutoTokenizer


# creates a simple LLM using a small GPT-2 model
def create_simple_llm():
    model_name = "distilgpt2"

    generator = pipeline("text-generation", model=model_name, pad_token_id=50256)

    return generator


prompt = "Once upon a time.."

generator = create_simple_llm()
response = generator(prompt, max_length=100, num_return_sequence=1)
generated_text = response[0]["generated_text"]

print(generated_text)
