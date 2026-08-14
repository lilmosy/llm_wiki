# Dense index (B2)

- embedding model: `Qwen/Qwen3-Embedding-0.6B` (paper uses Qwen3-Embedding-8B — substituted)
- vectors: 60 passages x 1024 dims
- LLM calls to build: **0** (embedding only)

One vector per raw passage. No summarisation, no structure — this is the flat-chunk arm with a semantic retriever.
