import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import sys
import requests
from openai import OpenAI
from dotenv import load_dotenv
import json


#load env from .env file
load_dotenv()

class MCPClient:
    # Initialize session and client objects
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env file")
        self.base_url = "https://api.deepseek.com/v1"  
        
        self.model = os.getenv("MODEL")
        self.client = OpenAI(api_key=self.deepseek_api_key, base_url = self.base_url)

    # Connect to an MCP server:
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server file {server_script_path} does not exist")

        # 确定执行命令和参数
        if server_script_path.endswith('.py'):
            command = "python"
            args = [server_script_path]
        elif server_script_path.endswith('.js'):
            command = "node"
            args = [server_script_path]
        else:
            # 二进制文件情况
            if not os.access(server_script_path, os.X_OK):
                raise PermissionError(f"File {server_script_path} is not executable")
            command = server_script_path
            args = []

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
                "role": "system",
                "content": "You are an autonomous Kubernetes assistant. When asked to perform actions, "
                        "automatically execute the necessary tools without asking for confirmation. "
                        "Only ask for clarification if the request is ambiguous or missing required parameters."
            },
            {
                "role": "user",
                "content": query
            }
        ]
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            tool_response = await self.session.list_tools()
            # print(f"\n Debug: {tool_response.tools}")

            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    # "input_schema": tool.inputSchema
                    "parameters": tool.inputSchema
                }
            } for tool in tool_response.tools]

            # 调用 OpenAI(Deepseek) API
            print(f"\nDebug[request message:]: {messages}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )

            print(f"\n Debug(deepseek response object): {response}")

            choice = response.choices[0]
            message = choice.message

            # Handle tool calls
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing tool arguments: {e}")
                        tool_args = {}

                    print(f"\n[Executing {tool_name} with args {tool_args}]")
                    try: 
                        result = await self.session.call_tool(tool_name, tool_args)
                        
                        # Add both the assistant message and tool response to history
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments
                                    },
                                    "type": "function"
                                }
                            ]
                        })
                        messages.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": str(result),
                            "tool_call_id": tool_call.id
                        })                        
                    except Exception as e:
                        return f"Error executing {tool_name}: {str(e)}"
            # If final response
            else:
                return message.content
        return "Maximum iterations reached without completing the request"

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

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_chatbot_deepseek.py <path_to_mcp_server> [support .py/.js/binary form]")
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
