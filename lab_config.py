from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
PDF_FILE = INPUT_DIR / "eu_ai_act.pdf"
PODCAST_FILE = INPUT_DIR / "podcast_transcript.txt"
CHUNKS_FILE = BASE_DIR / "chunks.jsonl"

# --- Chunking ---
# EMBEDDING_MODEL is only used here to pick a matching tiktoken tokenizer,
# so CHUNK_SIZE_TOKENS means "tokens" rather than raw characters.
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

# --- Source labels ---
SOURCE_EU_AI_ACT = "eu_ai_act"
SOURCE_PODCAST = "podcast_transcript"