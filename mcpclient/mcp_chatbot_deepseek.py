import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import sys
import requests
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import json


load_dotenv()  # load environment variables from .env

class MCPClient:
    # Initialize session and client objects
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env file")
        self.base_url = "https://api.deepseek.com/v1"  # DeepSeek API endpoint
        
        self.model = os.getenv("MODEL")
        self.client = OpenAI(api_key=self.deepseek_api_key, base_url = self.base_url)

    # we’ll implement the method to connect to an MCP server:
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        print(f"\n Debug: {response.tools}")

        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        print(f"\n Debug: {available_tools}")
        # Initial Deepseek api call
        # 调用 OpenAI API
        print(f"\nDebug: {messages}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools
        )

        print(f"\n Debug {response}")

        content = response.choices[0]

        # Process response and handle tool calls
        # final_text = []

        # Process response and handle tool calls
        if "tool_calls" == content.finish_reason:
            # for tool_call in content.message.tool_calls:
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Execute tool call
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")

            # final_text.append(f"[Called tool {tool_name} with args {tool_args}]")

            # Add tool response to conversation history
            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": result.content[0].text,
                "tool_call_id": tool_call.id
            })
            
            print(f"\n Debug: {messages}")
            
            # Get next response from DeepSeek
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages, 
            )
            return response.choices[0].message.content

        return content.message.content
            

    # Now we’ll add the chat loop and cleanup functionality:
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# Main Entry Point
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
