from datetime import datetime
from dotenv import load_dotenv
import os
import requests
import json
from utilites.utility import log
load_dotenv();

def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S');
def get_city_weather(city):
    url = f"{os.getenv("WEATHER_URL")}/{city}?format=3"
    response = requests.get(url,  timeout=10)
    response.raise_for_status();
    return response.text

def get_current_city():
    return "sahaswan Uttar Pradesh India"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time as formatted string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Returns the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":{
                        "type":"string",
                        "description":"City to get the weather for."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_city",
            "description": "Returns the current city of user, if user is asking like 'my current location' 'my location' 'my city'  'my current city'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


FUNCTION_NAME_TO_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "get_current_city": get_current_city
}


def run_tool(tool_call):
    func_name = tool_call.function.name
    func_args = {}
    try:
        fn = FUNCTION_NAME_TO_MAP.get(func_name)
        func_args = json.loads(tool_call.function.arguments);
        log(f"✅ {func_name} ({func_args})")
        result = fn(**func_args);
    except Exception as e:
        result = f"Error while running {func_name}:{e}"
    log(f" ➡ {result}")
    return result;

