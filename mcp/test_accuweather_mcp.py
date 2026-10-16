import asyncio

from fastmcp import Client
from fastmcp.client.transports.http import StreamableHttpTransport

transport = StreamableHttpTransport(url="http://localhost:8020/accu-mcp-server") #A
client = Client(transport) #B

async def main():
    # Connection is established here
    async with client:
        print(f"Client connected: {client.is_connected()}")
        tools = await client.list_tools() #C
        print(f"Tools available: {tools}")
        if (any(tool.name == "get_weather_conditions" for tool in tools)): #D
            result = await client.call_tool("get_weather_conditions", {"location": "Penzance, UK"}) #E
            print(f"Call result: {result}") #F

    # Connection is closed here automatically 
    print(f"Client connected: {client.is_connected()}")

if __name__ == "__main__":
    asyncio.run(main())

#A - Set up transport as streamable HTTP server agains the MCP server running on port 8020
#B - Create MCP client
#C - List tools exposed by the MCP server
#D - Check if tool exists
#E - Call tool
#F - Print result of the call
