from datetime import datetime
from dotenv import load_dotenv
import os
import requests

load_dotenv();

def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S');
def get_city_weather(city):
    url = f"{os.getenv("WEATHER_URL")}/{city}?format=3"
    response = requests.get(url,  timeout=10)
    return response.text


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
                "required": []
            }
        }
    }
]


FUNCTION_NAME_TO_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
}

