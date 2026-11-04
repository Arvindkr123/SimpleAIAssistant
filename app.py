from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import TOOLS, run_tool
from utilites.utility import log
from types import SimpleNamespace

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
            "You are a helpful, humble assistant. "
            "Answer general questions (programming, science, advice) from your own language. "
            "Use a tool only when the user asks for the current time, the weather, or their city. "
            "Keep answers under 100 words."
        )
    }
)

def build_tool_calls_payload(tool_calls):
    """
        Converts our collected tool calls into dict format.
        that the api expects inside the assistant message.
    """
    payload = []
    for tool_call in tool_calls:
        tool_call_dict = {
            "id":tool_call.id,
            "type":"function",
            "function":{
                "name":tool_call.function.name,
                "arguments":tool_call.function.arguments
            }
        }
        payload.append(tool_call_dict)
    return payload;

def stream_chat(history):
    streams = client.chat.completions.create(
                    model=os.getenv("MODEL"),
                    messages=history,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True
                )
    full_response = ""
    calls = {}
    label_printed = False
    try:
        for chunk in streams:
            if not chunk.choices:
                continue;
            delta = chunk.choices[0].delta;

            # Streaming the content::
            if delta.content:
                if not label_printed:
                    print("😊 Assistant: ", end="", flush=True);
                    label_printed=True;
                print(delta.content, end="", flush=True);
                full_response+=delta.content;

            # 2 Tools Calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments=""))
                )

                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name+= tc.function.name;
                if tc.function and tc.function.arguments:
                    obj.function.arguments+= tc.function.arguments;
    except KeyboardInterrupt:
        print();
        print("Stoped by you")
        calls ={}

    if label_printed:
        print();

    calls_list =[]
    for key in sorted(calls):
        calls_list.append(calls[key])
    return full_response, calls_list

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
            full_response, tool_calls = stream_chat(messages)

            if not tool_calls:
                break;
            
            log(f"Step {step+1}: model requested {len(tool_calls)} tool calls")

            messages.append({
                "role":"assistant",
                "content":full_response,
                "tool_calls":build_tool_calls_payload(tool_calls)
            })

            for tool_call in tool_calls:
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
    except Exception as e:
        print("Error:", e)