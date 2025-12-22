import ollama
import logging
import json
from datetime import datetime
import uuid


def setup_logging():
    """Configure logging to save logs in JSON format"""
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    # Create a file handler for JSON logs
    file_handler = logging.FileHandler("./logging/logs.json")
    formatter = logging.Formatter("%(message)s")  # Log raw JSON
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler for human-readable logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    return logger


def chatbot(user_input, messages, session_id, logger):
    try:
        model_name = "llama3.2"

        # log user input with metadata
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "type": "user_input",
            "user_input": user_input,
            "metadata": {"session_id": session_id, "model": model_name},
        }
        logger.info(json.dumps(log_entry))

        # append user messages
        messages.append({"role": "user", "content": user_input})

        # generate a response using the API
        start_time = datetime.now()
        response = ollama.chat(model=model_name, messages=messages)
        end_time = datetime.now()

        # calculate response time
        response_time = (end_time - start_time).total_seconds()

        chabot_response = response["message"]["content"]

        # append chatbot/assistant messages
        messages.append({"role": "assistant", "content": chabot_response})

        # log model response and performance metrics
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "type": "model_response",
            "model_response": chabot_response,
            "metadata": {
                "session_id": session_id,
                "model": model_name,
                "response_time_seconds": response_time,
                "tokens_used": (
                    response.usage.total_tokens if hasattr(response, "usage") else None
                ),
            },
        }
        logger.info(json.dumps(log_entry))

        return chabot_response
    except Exception as e:
        # log any errors that occur
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "type": "error",
            "error_message": str(e),
            "metadata": {"session_id": session_id, "model": model_name},
        }
        logger.error(json.dumps(log_entry))
        return f"An error occured: {e}"


def main():
    print("This is a CHATBOT with logs")
    print("Enter 'q' to quit")
    print("Enter 'c' to clear screen")

    session_id = str(uuid.uuid4())
    logger = setup_logging()

    messages = [
        {
            "role": "system",
            "content": "Your name is Fred, you are a smart assistant, that has the ability to answer question briefly and informatively.",
        }
    ]

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

        response = chatbot(
            user_input=user_input,
            messages=messages,
            session_id=session_id,
            logger=logger,
        )
        print("\n Chatbot: ", response)
        has_begun_chat = True


main()
