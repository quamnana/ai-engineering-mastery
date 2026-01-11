from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# Basic Chat Model with OpenAI
model = init_chat_model("gpt-4.1")
response = model.invoke("Why do parrots talk?")

print(response)


# Prompt Template
prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

prompt_value = prompt_template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?",
    }
)

response = model.invoke(prompt_value)
print(response)
