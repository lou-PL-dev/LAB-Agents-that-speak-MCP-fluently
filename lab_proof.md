# Lab Proof — Agents that speak MCP fluently

## Workflow

1. `data_prep.py` extracts `input/eu_ai_act.pdf` (144 pages) and
   `input/podcast_transcript.txt`, chunks both, and tags each chunk with
   its source, page number, and legal structure (article/recital/
   chapter/annex) → `chunks.jsonl` (429 chunks: 416 from the AI Act, 13
   from the podcast).
2. `mcp_server.py` exposes that data over MCP via three tools:
   `list_sources`, `search_chunks(query, source, top_k)`,
   `get_chunk(chunk_id)`.
3. `mcp_langchain.py` connects a LangGraph ReAct agent (GPT-4o-mini) to
   the server over stdio and answers a query, using only
   `search_chunks` evidence.

## Test query

> "According to these documents, what are the three components of
> trustworthy AI, and does the EU AI Act codify any of them into legal
> obligations? Cite where each part of your answer comes from."

## Retrieved evidence (verified via direct MCP tool call)

**Tool call:** `search_chunks(query="what are the three components of trustworthy AI lawful ethical robust", source="podcast_transcript", top_k=1)`

**Result — `podcast_transcript_002`:**
> "...the source material defines trustworthy AI as having three
> components... Think of it like a three-legged stool. The first leg is
> lawful... That's why the second leg of the stool is ethical... And the
> third leg? Robust..."

**Tool call:** `search_chunks(query="human oversight requirement obligations providers", source="eu_ai_act", top_k=1)`

**Result — `eu_ai_act_p021_2`** (page 21, recital 73):
> "...it is appropriate to provide for an enhanced human oversight
> requirement for those systems so that no action or decision may be
> taken by the deployer... unless this has been separately verified and
> confirmed by at least two natural persons..."

This shows the "ethical" (human oversight, tied to the "lawful"/"ethical"
components from the podcast) getting a concrete legal form in the AI Act
— a direct, page-cited link between the two sources.

## Final agent output

> _[Run `python3 mcp_langchain.py` locally and paste the agent's final
> answer here — this requires a live OpenAI API call, which this
> sandbox environment cannot make (network to `api.openai.com` is
> blocked here). Everything above this point — chunking, MCP server,
> tool calls, and retrieved evidence — was verified directly against
> the real documents.]_

## Failure / limitation

`search_chunks` uses simple keyword-overlap scoring (no embeddings) —
fast and fully inspectable, but it has no relevance threshold. Tested
with a deliberately unrelated query:

**Tool call:** `search_chunks(query="quarterly earnings stock price Tesla", top_k=3)`

**Result:** still returned a "top match" (`eu_ai_act_p027_1`, score 1)
about open-source AI licensing — clearly irrelevant to the query, but
the tool has no way to say "nothing relevant here" versus "weak match."
A production version would need a minimum-score cutoff (or a real
embedding-based similarity search) so the agent — and a human reviewing
its trace — can tell a genuine retrieval from a coincidental word
overlap.
