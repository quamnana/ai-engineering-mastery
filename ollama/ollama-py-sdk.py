import ollama
import base64

# list your current ollama models
models = ollama.list()

# chat example
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Tell me a short history of Ghana's independence"}
    ],
)

print(response["message"]["content"])


# chat example with streaming
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Tell me a short history of Ghana's independence"}
    ],
    stream=True,
)

print("Generated Text: ", end="", flush=True)
for chunk in response:
    if chunk:
        generated_text = chunk["message"]["content"]
        print(generated_text, end="", flush=True)


# chat with multimodal model

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()


response = ollama.chat(
    model="llava:7b",
    messages=[
        {
            "role": "user",
            "content": "Decribe this picture to me",
            "images": [b64],
        }
    ],
)
print(response["message"]["content"])
