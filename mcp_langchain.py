"""
Steps 3-4 of the lab: load MCP tools into LangChain and run a
Document Analysis Agent against them.

Run with: python mcp_langchain.py
"""

import asyncio
import json
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()  # expects OPENAI_API_KEY in .env

SYSTEM_PROMPT_TEMPLATE = """You are a document analysis assistant with access to two \
sources: the EU AI Act (a legal regulation) and a podcast transcript that \
discusses the EU HLEG Ethics Guidelines for Trustworthy AI.

Below is the full podcast transcript, loaded as background context via an \
MCP *resource* (not a tool call) - it's small enough to always have on \
hand. The EU AI Act is 144 pages, far too large for that, so it stays \
behind the search_chunks tool instead.

--- PODCAST TRANSCRIPT (background context, loaded via MCP resource) ---
{podcast_transcript}
--- END BACKGROUND CONTEXT ---

For anything from the EU AI Act, always use the search_chunks tool to find \
evidence before answering - never answer from general knowledge alone. For \
the podcast transcript, you can answer directly from the background context \
above. When you answer, always cite which source (and page/article/recital, \
if given) each piece of your answer came from. If neither the background \
context nor the tools have relevant evidence for something, say so \
explicitly instead of guessing."""

def _mermaid_escape(text: str, limit: int = 60) -> str:
    """Truncate and sanitize text so it's safe to embed as a Mermaid node label."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return (
        text.replace('"', "'")
        .replace("[", "(").replace("]", ")")
        .replace("{", "(").replace("}", ")")
    )


def _extract_tool_text(content) -> str:
    """
    ToolMessage.content from an MCP call isn't a plain string - it's a list
    of content blocks like [{'type': 'text', 'text': '...'}]. Pull the
    actual text out instead of stringifying that whole structure (which is
    what produced the garbled "({'type': 'text'..." node labels before).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _summarize_evidence(text: str) -> str:
    """search_chunks returns a JSON list of chunk dicts - summarize it as
    'N chunk(s): id1, id2, ...' for a readable diagram node instead of
    dumping the full chunk text (which is already in the printed trace)."""
    try:
        data = json.loads(text)
        if isinstance(data, list) and data and "chunk_id" in data[0]:
            ids = [d.get("chunk_id", "?") for d in data]
            return f"{len(ids)} chunk(s): " + ", ".join(ids)
    except (json.JSONDecodeError, TypeError):
        pass
    return text


def _save_execution_trace(messages) -> None:
    """
    Builds a diagram of what actually happened on THIS run: the real tool
    calls the agent chose, with their real arguments, and a summary of the
    real evidence that came back - the decision path your teacher meant
    when they said "see what the agents are doing."
    """
    lines = ["flowchart TD"]
    prev_id = None
    node_count = 0

    def add_node(label: str, shape_open: str = "[", shape_close: str = "]") -> str:
        nonlocal prev_id, node_count
        node_id = f"n{node_count}"
        node_count += 1
        lines.append(f'    {node_id}{shape_open}"{_mermaid_escape(label)}"{shape_close}')
        if prev_id is not None:
            lines.append(f"    {prev_id} --> {node_id}")
        prev_id = node_id
        return node_id

    for msg in messages:
        role = msg.__class__.__name__
        if role == "HumanMessage":
            add_node(f"Query: {msg.content}", "([", "])")
        elif role == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                add_node(f"Agent decides: {tc['name']}({args_str})", "{{", "}}")
        elif role == "ToolMessage":
            text = _extract_tool_text(msg.content)
            summary = _summarize_evidence(text)
            add_node(f"Evidence from {msg.name}: {summary}")
        elif role == "AIMessage":
            add_node(f"Final answer: {msg.content}", "([", "])")

    mermaid_text = "\n".join(lines)
    with open("execution_trace.mmd", "w") as f:
        f.write(mermaid_text)
    print("\nSaved this run's actual decision path to execution_trace.mmd")
    print("(paste into https://mermaid.live to view)")


TEST_QUERY = (
    "According to these documents, what are the three components of "
    "trustworthy AI, and does the EU AI Act codify any of them into legal "
    "obligations?"
)


async def run_agent(query: str) -> None:
    client = MultiServerMCPClient(
        {
            "document_search": {
                "command": sys.executable,
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    resource_blobs = await client.get_resources(
        "document_search", uris=["docs://podcast-transcript"]
    )
    podcast_transcript = resource_blobs[0].as_string()
    print(f"Loaded 1 MCP resource: docs://podcast-transcript "
          f"({len(podcast_transcript)} chars, no tool call needed)\n")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(podcast_transcript=podcast_transcript)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(model, tools, system_prompt=system_prompt)

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

    _save_execution_trace(result["messages"])


if __name__ == "__main__":
    asyncio.run(run_agent(TEST_QUERY))