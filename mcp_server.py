"""
MCP server exposing the chunked documents from
chunks.jsonl as tools, instead of raw filesystem access.

The EU AI Act is a 144-page PDF.
Handing an agent a raw "read_file" tool on it would either blow the
context window or force the agent to guess where to look. Since
data_prep.py already split both documents into chunks tagged with their
source, page number, and legal structure (article/recital/chapter/annex),
exposing a *search* tool over those chunks keeps every retrieval step
small, targeted.

Search here is a simple keyword-overlap scorer (no embeddings, no extra
API calls).
"""

import json
import re
from collections import Counter

try:
    # mcp < 2.0 - this is the version pinned in requirements.txt, because
    # langchain-mcp-adapters 0.3.1 is not yet compatible with mcp 2.0's
    # transport changes.
    from mcp.server.fastmcp import FastMCP as MCPServerClass
except ImportError:
    # mcp >= 2.0 renamed FastMCP -> MCPServer (same decorator API).
    from mcp.server.mcpserver import MCPServer as MCPServerClass

import lab_config as config

mcp = MCPServerClass("document-search")

_CHUNKS: list[dict] = []


def _load_chunks() -> list[dict]:
    global _CHUNKS
    if _CHUNKS:
        return _CHUNKS
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        _CHUNKS = [json.loads(line) for line in f if line.strip()]
    return _CHUNKS


_WORD_RE = re.compile(r"[a-zA-Z']+")


def _keywords(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD_RE.findall(text))


def _score(query_words: Counter, chunk_words: Counter) -> int:
    """Simple overlap score: sum of matching word counts. No embeddings —
    good enough to demonstrate grounded retrieval without an extra API call
    inside the tool itself."""
    return sum(min(count, chunk_words[word]) for word, count in query_words.items())


@mcp.tool()
def list_sources() -> str:
    """List the available document sources and how many chunks each has."""
    chunks = _load_chunks()
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    return json.dumps(counts, indent=2)


@mcp.tool()
def search_chunks(query: str, source: str | None = None, top_k: int = 3) -> str:
    """
    Search the document chunks for the ones most relevant to `query`.

    Args:
        query: free-text question or keywords to search for.
        source: optional filter - "eu_ai_act" or "podcast_transcript".
                Leave empty to search across both.
        top_k: how many top-matching chunks to return (default 3).

    Returns a JSON list of matches, each with chunk_id, source, page,
    article/recital/chapter/annex (when applicable), and the chunk text -
    so the caller can see exactly where the evidence came from.
    """
    chunks = _load_chunks()
    if source:
        chunks = [c for c in chunks if c["source"] == source]

    query_words = _keywords(query)
    scored = []
    for c in chunks:
        s = _score(query_words, _keywords(c["text"]))
        if s > 0:
            scored.append((s, c))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:top_k]

    results = []
    for score, c in top:
        entry = {
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "score": score,
            "text": c["text"],
        }
        if c.get("page") is not None:
            entry["page"] = c["page"]
        for field in ("article", "recital", "chapter", "annex"):
            if c.get(field):
                entry[field] = c[field]
        results.append(entry)

    if not results:
        return json.dumps({"message": "No matching chunks found for this query."})
    return json.dumps(results, indent=2)


@mcp.tool()
def get_chunk(chunk_id: str) -> str:
    """Fetch one specific chunk's full text and metadata by its chunk_id."""
    chunks = _load_chunks()
    for c in chunks:
        if c["chunk_id"] == chunk_id:
            return json.dumps(c, indent=2)
    return json.dumps({"message": f"No chunk found with id {chunk_id}"})


# --- Resources (Step 5 of the lab) ---
#
# Unlike tools, resources aren't "called" by the model with arguments -
# the client fetches them upfront and decides how to use them (e.g. drop
# straight into the system prompt as background context). They're for
# data that's cheap enough to just always have on hand.

@mcp.resource("docs://sources")
def sources_resource() -> str:
    """Read-only overview of available document sources and chunk counts.
    Same information as the list_sources tool, but as a resource: no tool
    call needed, the client can just load this once and keep it around."""
    return list_sources()


@mcp.resource("docs://podcast-transcript")
def podcast_transcript_resource() -> str:
    """
    The full podcast transcript, verbatim. At ~16KB this is small enough
    to keep in context permanently as background - unlike the 144-page
    EU AI Act, which stays behind the search_chunks tool because loading
    it wholesale would blow the context window. This contrast is exactly
    why MCP distinguishes Resources (cheap, always-on context) from Tools
    (targeted, on-demand retrieval).
    """
    return config.PODCAST_FILE.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="stdio")