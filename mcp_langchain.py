"""
Steps 3-4 of the lab: load MCP tools into LangChain and run a
Document Analysis Agent against them.

This script:
1. Starts our MCP server (mcp_server.py) as a subprocess over stdio,
   via MultiServerMCPClient.
2. Loads its tools (list_sources, search_chunks, get_chunk) as LangChain
   tools.
3. Builds a small ReAct-style agent (LangGraph's create_react_agent) with
   those tools and an OpenAI chat model.
4. Runs one test query end-to-end and prints the full trace: which
   tool(s) were called, what evidence came back, and the final answer -
   this is the raw material for lab_proof.md.

Run with: python mcp_langchain.py
"""

import asyncio
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()  # expects OPENAI_API_KEY in .env

SYSTEM_PROMPT = """You are a document analysis assistant with access to two \
sources: the EU AI Act (a legal regulation) and a podcast transcript that \
discusses the EU HLEG Ethics Guidelines for Trustworthy AI.

Always use the search_chunks tool to find evidence before answering - never \
answer from general knowledge alone. When you answer, cite which source \
(and page/article/recital, if given) each piece of your answer came from. \
If the tools don't return relevant evidence for something, say so explicitly \
instead of guessing."""

TEST_QUERY = (
    "According to these documents, what are the three components of "
    "trustworthy AI, and does the EU AI Act codify any of them into legal "
    "obligations? Cite where each part of your answer comes from."
)


async def run_agent(query: str) -> None:
    client = MultiServerMCPClient(
        {
            "document_search": {
                "command": "python3",
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}\n")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    print("=" * 80)
    print("FULL MESSAGE TRACE (query -> tool calls -> evidence -> final answer)")
    print("=" * 80)
    for msg in result["messages"]:
        role = msg.__class__.__name__
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"\n[{role} -> TOOL CALL] {tc['name']}({json.dumps(tc['args'])})")
        elif role == "ToolMessage":
            print(f"\n[TOOL RESULT: {msg.name}]\n{msg.content}")
        else:
            print(f"\n[{role}]\n{msg.content}")

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent(TEST_QUERY))
