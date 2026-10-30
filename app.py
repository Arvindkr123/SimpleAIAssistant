from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import TOOLS, run_tool
from utilites.utility import log

load_dotenv();

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = [];
MAX_STEPS = 5;
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
        for step in range(MAX_STEPS):
            response = client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                break;

            log(f"Step {step+1}: model requested {len(msg.tool_calls)} tool calls")

            messages.append(msg);   

            for tool_call in msg.tool_calls:
                result = run_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            fallback ="Sorry, I could not finish this within the step limit."
            messages.append({
                    "role": "assistant",
                    "content": fallback
            })
            continue
        messages.append({
                "role": "assistant",
                "content": msg.content
        })
        print("Assistant:", msg.content or "")
    except Exception as e:
        print("Error:", e)