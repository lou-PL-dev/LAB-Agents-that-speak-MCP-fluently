"""
Step 1 — Data preparation.

1. Extracts the EU AI Act PDF page by page (pdfplumber), keeping page numbers.
2. Loads the podcast transcript (produced by transcribe.py).
3. Chunks both texts with a splitter that respects paragraph/sentence
   boundaries instead of cutting mid-sentence at a fixed character count.
4. Attaches metadata to every chunk and writes everything to chunks.jsonl,
   which embeddings.py will read next.
"""

import json
import re
import pdfplumber
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

import lab_config as config

# tiktoken encoder matching the embedding model's tokenizer, used so
# CHUNK_SIZE_TOKENS actually means "tokens", not "characters".
_encoding = tiktoken.encoding_for_model(config.EMBEDDING_MODEL)


def _token_length(text: str) -> int:
    return len(_encoding.encode(text))


# Recursive splitter: tries paragraph breaks first ("\n\n"), then single
# newlines, then sentences, then words - only falling back to a hard cut
# if nothing else fits. This keeps chunks semantically coherent.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE_TOKENS,
    chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
    length_function=_token_length,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# --- Legal structure detection (for Step 5: metadata filtering) ---
#
# Heuristic, not perfect: pdfplumber's extracted text doesn't always
# preserve line breaks exactly where the PDF layout suggests, so these
# regexes will catch the clear cases (a header on its own line) but can
# miss or misfire on edge cases. Good enough to demonstrate filtering;
# not a substitute for a proper legal-XML parser of the EU AI Act.
_ARTICLE_RE = re.compile(r"(?:^|\n)Article\s+(\d+)\s*\n", re.MULTILINE)
_RECITAL_RE = re.compile(r"(?:^|\n)\((\d+)\)\s")
_CHAPTER_RE = re.compile(r"(?:^|\n)CHAPTER\s+([IVXLC]+)\b", re.MULTILINE)
_ANNEX_RE = re.compile(r"(?:^|\n)ANNEX\s+([IVXLC]+)\b", re.MULTILINE)


def extract_pdf_pages(pdf_path) -> list[dict]:
    """Returns a list of {"page": int, "text": str} for every non-empty page."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i, "text": text})
    print(f"Extracted {len(pages)} non-empty pages from {pdf_path.name}")
    return pages


def _find_section_events(full_text: str) -> dict:
    """
    Scans the FULL document text (all pages concatenated, in order) for
    section headers, returning {"article": [(offset, "13"), ...], "recital": [...], ...}
    sorted by offset. We scan the whole document at once (not page-by-page)
    because a section can span a page break, and we need the running
    "what section are we currently in" state to carry over correctly.
    """
    events = {
        "article": [(m.start(), m.group(1)) for m in _ARTICLE_RE.finditer(full_text)],
        "recital": [(m.start(), m.group(1)) for m in _RECITAL_RE.finditer(full_text)],
        "chapter": [(m.start(), m.group(1)) for m in _CHAPTER_RE.finditer(full_text)],
        "annex": [(m.start(), m.group(1)) for m in _ANNEX_RE.finditer(full_text)],
    }
    for kind in events:
        events[kind].sort()
    return events


def _value_at_offset(events: list[tuple[int, str]], offset: int):
    """Returns the value of the last event at or before `offset`, or None."""
    current = None
    for ev_offset, value in events:
        if ev_offset <= offset:
            current = value
        else:
            break
    return current


def chunk_eu_ai_act() -> list[dict]:
    """
    Chunks the EU AI Act PDF, keeping page number AND legal-structure
    metadata (article / recital / chapter / annex number) for each chunk,
    so Step 5 can filter e.g. "articles only, exclude recitals".
    """
    pages = extract_pdf_pages(config.PDF_FILE)

    # Build one big string with a record of each page's start offset, so we
    # can locate any chunk's absolute position in the whole document and
    # look up which section was "active" there.
    full_text = ""
    page_start_offsets = {}
    for page_data in pages:
        page_start_offsets[page_data["page"]] = len(full_text)
        full_text += page_data["text"] + "\n"

    section_events = _find_section_events(full_text)
    # Recitals only exist in the preamble, before the operative Articles
    # begin. Without this boundary, "last recital number seen" would keep
    # getting carried forward for the rest of the document (a bug we hit:
    # 327/328 chunks were getting tagged with a recital number, which is
    # obviously wrong for a 144-page document).
    first_article_offset = section_events["article"][0][0] if section_events["article"] else len(full_text)

    chunks = []
    for page_data in pages:
        page_chunks = _splitter.split_text(page_data["text"])
        search_cursor = 0  # avoids false-matching an identical earlier snippet on the same page

        for j, chunk_text in enumerate(page_chunks):
            # Locate this chunk's position within the page text to compute its
            # absolute offset in the full document.
            anchor = chunk_text[:40]
            local_offset = page_data["text"].find(anchor, max(0, search_cursor - config.CHUNK_OVERLAP_TOKENS * 4))
            if local_offset == -1:
                local_offset = page_data["text"].find(anchor)
            if local_offset == -1:
                local_offset = search_cursor  # fallback: best guess
            search_cursor = local_offset
            absolute_offset = page_start_offsets[page_data["page"]] + local_offset

            article = _value_at_offset(section_events["article"], absolute_offset)
            recital = _value_at_offset(section_events["recital"], absolute_offset) if absolute_offset < first_article_offset else None
            chapter = _value_at_offset(section_events["chapter"], absolute_offset)
            annex = _value_at_offset(section_events["annex"], absolute_offset)

            chunks.append({
                "chunk_id": f"{config.SOURCE_EU_AI_ACT}_p{page_data['page']:03d}_{j}",
                "text": chunk_text,
                "source": config.SOURCE_EU_AI_ACT,
                "page": page_data["page"],
                "article": article,   # e.g. "13", or None if not inside an article
                "recital": recital,   # e.g. "72", or None if not inside a recital
                "chapter": chapter,   # roman numeral, e.g. "III", or None
                "annex": annex,       # roman numeral, e.g. "III", or None
            })
    print(f"Created {len(chunks)} chunks from the EU AI Act")

    tagged_article = sum(1 for c in chunks if c["article"] is not None)
    tagged_recital = sum(1 for c in chunks if c["recital"] is not None)
    print(f"  -> {tagged_article} chunks tagged with an article number, {tagged_recital} tagged with a recital number")

    return chunks


def chunk_podcast() -> list[dict]:
    """Chunks the podcast transcript. No page numbers, so we index chunks sequentially."""
    transcript = config.PODCAST_FILE.read_text(encoding="utf-8")
    text_chunks = _splitter.split_text(transcript)
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunks.append({
            "chunk_id": f"{config.SOURCE_PODCAST}_{i:03d}",
            "text": chunk_text,
            "source": config.SOURCE_PODCAST,
            "page": None,  # no page concept for audio; sequence order stands in for position
        })
    print(f"Created {len(chunks)} chunks from the podcast transcript")
    return chunks


def build_all_chunks() -> list[dict]:
    """Runs both chunking pipelines and writes the combined result to chunks.jsonl."""
    all_chunks = chunk_eu_ai_act() + chunk_podcast()

    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nWrote {len(all_chunks)} total chunks to {config.CHUNKS_FILE}")
    return all_chunks


if __name__ == "__main__":
    chunks = build_all_chunks()
    print("\nExample chunk:")
    print(json.dumps(chunks[0], indent=2)[:500])