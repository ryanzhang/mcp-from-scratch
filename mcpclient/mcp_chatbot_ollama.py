import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os
import sys
import requests
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MCPClient:
    # Initialize session and client objects
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("MODEL", "gemma3:12b")
        if not self.ollama_host:
            raise ValueError("OLLAMA_HOST not found in .env file or default not set")

    # Connect to an MCP server
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server file {server_script_path} does not exist")

        # Determine execution command and arguments
        if server_script_path.endswith('.py'):
            command = "python"
            args = [server_script_path]
        elif server_script_path.endswith('.js'):
            command = "node"
            args = [server_script_path]
        else:
            # Binary file case
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
        self.stdio, self.client = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.client))

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
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in tool_response.tools]

            # Call Ollama API
            try:
                response = ollama.chat(
                    model="llama3.1",
                    messages=messages,
                    tools=available_tools
                )
                print(f"\nDebug: Response: {response}")

                message = response["message"]
                # print(f"\nDebug[ollama response]: {response_data}")

                # Handle tool calls
                if "tool_calls" in message and message["tool_calls"]:
                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        try:
                            tool_args = tool_call["function"]["arguments"]
                        except KeyError:
                            print(f"Error: Invalid tool call format for {tool_name}")
                            tool_args = {}

                        print(f"\n[Executing {tool_name} with args {tool_args}]")
                        try:
                            result = await self.session.call_tool(tool_name, tool_args)

                            # Add both the assistant message and tool response to history
                            messages.append({
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [tool_call]
                            })
                            messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": str(result),
                                "tool_call_id": tool_call.get("id", "unknown")
                            })
                        except Exception as e:
                            return f"Error executing {tool_name}: {str(e)}"
                # If final response
                else:
                    return message["content"]
            except requests.RequestException as e:
                return f"Error calling Ollama API: {str(e)}"
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
        print("Usage: python mcp_chatbot_ollama.py <path_to_mcp_server> [support .py/.js/binary form]")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
