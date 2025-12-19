from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

completions = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful history teacher."},
        {
            "role": "user",
            "content": "Tell me some fun facts about Ghana's independence. Use GenZ slangs for your presentation. Add a title ",
        },
    ],
)

response = completions.choices[0].message.content
print(response)
