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


# few-shot learning
few_short_prompt = """ These are some countries and their capital cities: 
                'Ghana' -> 'Accra',
                'USA' -> 'Washington DC'.

                Now capital of: 'Norway'
                """
# generate_completions(few_short_prompt)

# direct prompting
direct_prompt = "What is the largest river in the world?"
# generate_completions(direct_prompt)

# chain-of-thought prompting
cot_prompt = "Solve this problem step-by-step. If it is 2am in Toronto, what will be the time is Sydney?"
# generate_completions(cot_prompt)

# instructional prompting
instruct_prompt = "Write a 300-word summary of rock formation in tropic regions."
# generate_completions(instruct_prompt)

# open-ended prompting
oe_prompt = "Write about what effects climate change would cause in the next 50 years"
generate_completions(oe_prompt, stream=True)
