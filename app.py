from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import TOOLS, FUNCTION_NAME_TO_MAP
import json
import re

load_dotenv();

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = [
];
messages.append(
    {
        "role":"system",
        "content":(
            "Be humble and keep every response under 20 words. "
            "When asked for the current date or time, call get_current_time "
            "and use its result; do not guess."
        )
    }
)

while True:
    user_input = input("\nArvind: ")

    if user_input.lower() in ["exit", "bye", "quit", "goodbye"]:
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        print("Tool calls:", msg.tool_calls)

        # Execute requested tools
        while msg.tool_calls:
            messages.append(msg.model_dump(exclude_none=True))

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name

                func_args = json.loads(
                    tool_call.function.arguments or "{}"
                )

                fn = FUNCTION_NAME_TO_MAP.get(func_name)

                if fn is None:
                    result = f"Unknown function: {func_name}"
                else:
                    try:
                        result = fn(**func_args)
                    except Exception as e:
                        result = f"Tool execution failed: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

            # Ask the model to process the tool result
            response = client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            msg = response.choices[0].message

        if msg.content:
            messages.append({
                "role": "assistant",
                "content": msg.content
            })

            print("Assistant:", msg.content)

    except Exception as e:
        print("Error:", e)