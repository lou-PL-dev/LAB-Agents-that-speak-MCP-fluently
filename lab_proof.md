# Lab Proof — Agents that speak MCP fluently

## Workflow

1. `data_prep.py` extracts `input/eu_ai_act.pdf` (144 pages) and
   `input/podcast_transcript.txt`, chunks both, and tags each chunk with
   its source, page number, and legal structure (article/recital/
   chapter/annex) → `chunks.jsonl`.
2. `mcp_server.py` exposes that data over MCP via three tools:
   `list_sources`, `search_chunks(query, source, top_k)`,
   `get_chunk(chunk_id)`.
3. `mcp_langchain.py` connects a LangChain agent (`create_agent`,
   GPT-4o-mini) to the server over stdio and answers a query, using only
   `search_chunks` evidence. It also saves `execution_trace.mmd`, a
   Mermaid diagram of the actual decision path for that run (real tool
   calls, real arguments, real evidence — not just the fixed
   architecture).

## Test query

> "According to these documents, what are the three components of
> trustworthy AI, and does the EU AI Act codify any of them into legal
> obligations?"

(Citation instructions live in the system prompt now, not the query
itself.)

## Tool calls made by the agent

1. `search_chunks(query="three components of trustworthy AI", source="podcast_transcript")`
2. `search_chunks(query="trustworthy AI legal obligations", source="eu_ai_act")`

The agent chose both calls itself, in parallel, without being told which
source covers which half of the question.

## Retrieved evidence

**From `podcast_transcript_001` / `podcast_transcript_002`** (score 4, 3):
> "...the source material defines trustworthy A.I. as having three
> components... Think of it like a three-legged stool. The first leg is
> lawful... That's why the second leg of the stool is ethical... And the
> third leg? Robust..."

**From `eu_ai_act_p001_2`** (page 1, recital 1) and `eu_ai_act_p002_2`
(page 2, recital 6) (score 4, 3): the Act's aim to promote "human centric
and trustworthy AI" and establish common rules for high-risk systems
"consistent with the Charter."

## Final agent answer (verbatim, from a verified run)

> According to the podcast transcript discussing the EU HLEG Ethics
> Guidelines for Trustworthy AI, the three components of trustworthy AI
> are:
>
> 1. **Lawful**: The AI must comply with all applicable regulations.
> 2. **Ethical**: The AI should adhere to ethical principles, even if the
>    law has not yet caught up with technological advancements.
> 3. **Robust**: The AI must be technically and socially robust, meaning
>    it should function reliably and not cause harm (unintentional harm
>    is still considered harm) (podcast transcript, chunks 001 and 002).
>
> Regarding the EU AI Act, it does codify aspects of trustworthy AI into
> legal obligations. The Act aims to ensure that AI systems are
> developed and used in accordance with fundamental rights and values,
> promoting human-centric and trustworthy AI. It establishes a uniform
> legal framework for the development, marketing, and use of AI systems,
> ensuring a high level of protection for health, safety, and
> fundamental rights (EU AI Act, recital 1 and 6). Specifically, it
> emphasizes the need for common rules for high-risk AI systems that
> align with the Ethics Guidelines for Trustworthy AI (EU AI Act,
> recital 7).
>
> Thus, while the Act does not explicitly list the three components, it
> incorporates their principles into legal obligations aimed at ensuring
> trustworthy AI.

**Grounding check**: every claim traces to a specific chunk_id returned
by `search_chunks` — the three-legged-stool framing to the podcast
chunks, and each "codified obligation" claim to a specific recital
number in the AI Act. No claim is unsupported by retrieved evidence.

## MCP Resources (Step 5)

`mcp_server.py` exposes two resources: `docs://sources` (lightweight
overview) and `docs://podcast-transcript` (full transcript, ~16KB).
`mcp_langchain.py` loads the transcript resource via
`client.get_resources(...)` and injects it directly into the system
prompt *before* the agent starts reasoning - confirmed via the real MCP
protocol (`ListResourcesRequest` / `ReadResourceRequest` in the logs),
not just as a local Python call.

**Observation**: despite the full podcast transcript already being in
context as background (no tool call needed for it), the agent still
called `search_chunks(source="podcast_transcript")` anyway on this run.
It didn't need to - the answer it gave matches what's directly readable
from the background context. This suggests the model defaults to its
"always search before answering" instruction literally, rather than
recognizing that some evidence is already in hand. It's a small but
real illustration of a Resources/Tools design trade-off: giving a model
both a resource *and* a tool over the same content doesn't guarantee
it'll prefer the free one.

## Additional testing (4 more queries, beyond the main test query)

To avoid over-indexing on one repeated query, four different query types
were tested against the full agent:

| Query type | Result |
|---|---|
| Out-of-scope ("EU AI Act on cryptocurrency mining?") | Agent correctly said the Act doesn't cover this, despite the tool returning 3 weak (score 1) matches - the system prompt's "say so explicitly instead of guessing" instruction worked at the agent level even where the tool's scorer couldn't distinguish relevance |
| Resource-only ("what is red teaming and why does the host like it?") | Answered entirely from the podcast-transcript *resource* in context - zero tool calls, confirming Resources can fully replace a tool call when the content is already in context and the query doesn't cue "search the documents" |
| Narrow legal query (migration/asylum AI systems) | Correctly retrieved and cited the exact relevant recitals (59, 60) on specific legal language, not just the broad terms used in the main test query |
| Multi-hop (podcast's "human in the loop" -> AI Act's oversight requirement) | Correctly connected a concept named in the podcast to the Act's actual human oversight recitals (26, 65, 71) - genuine cross-source reasoning, properly cited |

**Revised takeaway**: the earlier "fairness/accountability" wrong answer
(see below) looks more like a one-off triggered by a specific vague
query than a systemic failure - none of these four additional queries
reproduced anything similar. The underlying tool-level limitation (no
relevance threshold in `search_chunks`) is still real and worth fixing
for production use, but in practice the system prompt's explicit
"admit when there's no evidence" instruction acted as an effective
safety net at the agent level in every test run here.

## Failure / limitation

Two failure cases were found, both stemming from the same root cause:
`search_chunks` uses simple keyword-overlap scoring (no embeddings, no
relevance threshold).

**1. A real wrong answer, not a contrived one.** On an earlier run, a
slightly different agent-generated search query — "components of
trustworthy AI" instead of "three components of trustworthy AI" —
retrieved different chunks (podcast chunks 009 and 010, about
*requirements* 6 and 7: societal well-being and accountability) instead
of the chunk that actually explains what the three components are.
The agent then confidently answered "fairness, accountability, and
societal/environmental well-being" — wrong, but cited with the exact
same style and confidence as the correct answer in other runs. This is
the core risk this lab is about: a plausible, well-cited, *ungrounded*
answer that looks identical to a grounded one unless you check the
actual retrieved evidence.

**2. No "nothing relevant" signal.** Tested with a deliberately
unrelated query: `search_chunks(query="quarterly earnings stock price
Tesla", top_k=3)` still returned a "top match" (`eu_ai_act_p027_1`,
score 1) about open-source AI licensing — clearly irrelevant, but the
tool has no way to distinguish "nothing relevant" from "weak match."

**Takeaway**: a production version needs either a minimum-score cutoff
or real embedding-based similarity, *and* ideally the agent's search
query phrasing shouldn't be able to silently steer it toward wrong
evidence. Both failures were only visible because the trace shows the
actual chunk_ids and scores retrieved — confirming the value of keeping
retrieval inspectable rather than trusting the final answer at face
value.