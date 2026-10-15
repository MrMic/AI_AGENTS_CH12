import os
from typing import Dict

from aiohttp import ClientSession
from dotenv import load_dotenv
from fastmcp import FastMCP

#--------------------------------
# ADAPTED FROM: https://github.com/adhikasp/mcp-weather
#--------------------------------

load_dotenv() #A

mcp = FastMCP("mcp-accuweather") #B

@mcp.tool(description="""Get weather conditions for a location.""") #C
async def get_weather_conditions(location: str) -> Dict:
    """Get weather conditions for a location."""
    api_key = os.getenv("ACCUWEATHER_API_KEY") #D  # None if unset -> API answers 401
    base_url = "http://dataservice.accuweather.com"

    async with ClientSession() as session:  # one session reused for both HTTP calls
        location_search_url = f"{base_url}/locations/v1/cities/search"
        params = { #E
            "apikey": api_key,
            "q": location,
        }
        async with session.get(location_search_url, params=params) as response:
            locations = await response.json() #F
            if response.status != 200:  # HTTP error (bad key, rate limit...)
                raise Exception(f"""Error fetching location data: {response.status}, {locations}""")
            if not locations or len(locations) == 0:  # search OK but empty list: unknown city
                raise Exception("Location not found")
        location_key = locations[0]["Key"] #G  # take best match; Key = AccuWeather city ID

        current_conditions_url = f"{base_url}/currentconditions/v1/{location_key}"
        params = { #H
            "apikey": api_key,
            "details": "true"  # adds humidity, precipitation flag, etc.
        }
        async with session.get(current_conditions_url, params=params) as response:
            current_conditions = \
               await response.json() #I
            
        if current_conditions and len(current_conditions) > 0:
            current = current_conditions[0] #J  # API returns a list with one observation
            current_data = {
                "temperature": {
                    "value": current["Temperature"]["Metric"]["Value"],  # Celsius (Metric block)
                    "unit": current["Temperature"]["Metric"]["Unit"]
                },
                "weather_text": current["WeatherText"],  # e.g. "Partly sunny"
                "relative_humidity": current.get("RelativeHumidity"),  # .get -> None if missing
                "precipitation": current.get("HasPrecipitation", False),  # default False if missing
                "observation_time": current["LocalObservationDateTime"]  # local time of the city
            }
        else:
            current_data = "No current conditions available"  # string fallback, not a dict

        return { #K
            "location": locations[0]["LocalizedName"], 
            "location_key": location_key, 
            "country": locations[0]["Country"]["LocalizedName"], 
            "current_conditions": current_data,
        }

if __name__ == "__main__":
    mcp.run(transport="streamable-http", #L  # HTTP transport; client URL = http://127.0.0.1:8020/accu-mcp-server
        host="127.0.0.1", 
        port=8020, path="/accu-mcp-server")
    
#A - Load environment variables
#B - Initialize FastMCP
#C - Define MCP tool
#D - Get AccuWeather API key
#E - Parameters for location search
#F - Get locations
#G - Get location key
#H - Current conditions parameters
#I - Get current conditions
#J - Format current conditions
#K - Return structured content
#L - Run MCP server
