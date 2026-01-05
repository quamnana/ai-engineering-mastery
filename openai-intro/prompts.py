from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


def generate_completions(prompt, stream=False):
    completions = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=stream,
        messages=[
            {"role": "system", "content": "You are a geography teacher."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    if stream:
        for chunk in completions:
            if chunk.choices[0].delta is not None:
                # print(chunk.choices[0].delta.content, end="")
                print(chunk.choices[0].delta.content or "", end="")
    else:
        response = completions.choices[0].message.content
        print(response)


prompts = [
    {
        "type": "Few-Shot Prompting",
        "prompt": """ These are some countries and their capital cities: 
                'Ghana' -> 'Accra',
                'USA' -> 'Washington DC'.

                'Norway' -> ?

                Please follow same format.
                """,
    },
    {
        "type": "Direct Prompting",
        "prompt": "What is the largest river in the world?",
    },
    {
        "type": "Instructional Prompting",
        "prompt": "Write a 100-word summary of rock formation in tropic regions.",
    },
    {
        "type": "Open-Ended Prompting",
        "prompt": "Write about what effects climate change would cause in the next 50 years",
    },
    {
        "type": "Chain-of-Thought Prompting",
        "prompt": "Solve this problem step-by-step. If it is 2am in Toronto, what will be the time is Sydney?",
    },
]

print(
    "Hi! My name is Fred, your intelligent geography assistant. I will be able to help you with your geography questions."
)
for idx, p in enumerate(prompts, start=1):
    prompt = p["prompt"]
    print(f"\n{idx}. {p['type']}")
    print(f"\nPrompt: {prompt}")
    print("\nResponse:")
    generate_completions(prompt, stream=True)
    print("\n")
