"""Shared sentence embedder for the dense / graph baselines.

The paper (§4.2) uses Qwen3-Embedding-8B for every method that needs vectors;
we run Qwen3-Embedding-0.6B, the same family one size down. Whatever model
config.yaml names, it applies IDENTICALLY to every arm that uses embeddings, so
the between-arm comparison stays controlled -- that invariant, not the specific
model, is what the comparison rests on.

Nothing here is model-specific: the name comes from cfg["embed_model"], and this
module only adds what every caller needs regardless of model -- one cached load,
L2 normalisation so a dot product IS cosine, and a top-k helper.

The LLM-Wiki arm and the BM25 arm do NOT use this module (paper §3.2: retrieval
quality is not the bottleneck; both stay lexical).
"""
import hashlib
import os

import numpy as np

# Keyed by model name, not a single slot: comparing two embedders inside one
# process (which is how the MiniLM/Qwen swap was measured) would otherwise get
# the first model back for both and silently report identical numbers.
_MODELS = {}


def _model(name):
    if name not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


def encode(texts, cfg, prompt_name=None):
    """texts: list[str] -> np.ndarray (n, d), L2-normalised so dot == cosine.

    Batched because a 0.6B encoder on Apple MPS runs out of memory long before
    the corpus does: LightRAG alone embeds ~1,100 entity and relation strings in
    one call, which OOM'd at the default batch size. The batch size only affects
    peak memory, never the resulting vectors.
    """
    m = _model(cfg["embed_model"])
    kw = {"batch_size": int(cfg.get("embed_batch_size", 8)), "show_progress_bar": False}
    # Use the model's own template when it ships one -- Qwen3-Embedding declares
    # a `query` prompt, and its published numbers assume it.
    if prompt_name and prompt_name in (getattr(m, "prompts", None) or {}):
        kw["prompt_name"] = prompt_name
    v = m.encode(list(texts), **kw)
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def encode_query(texts, cfg):
    """Encode the QUERY side of a retrieval pair.

    Qwen3-Embedding is trained asymmetrically: documents are embedded bare, but
    queries are meant to carry a task instruction, and omitting it costs recall.
    Models without that convention ignore the prefix, so routing every query
    through here keeps one code path.

    Every arm that embeds a query must call this, and every arm that embeds a
    document must call encode(). Mixing the two would give one arm a different
    query representation from the others, which is exactly the confound the
    shared-embedder setup exists to prevent.

    The instruction text comes from the model itself, never from us: Qwen3
    declares prompts={'query': 'Instruct: ...', 'document': ''}, and its
    published numbers assume that exact wording. Models that declare nothing
    (all-MiniLM) get the text unchanged, which is right -- they were never
    trained with an instruction.
    """
    return encode(texts, cfg, prompt_name="query")


def encode_cached(texts, cfg, path):
    """encode(), but the matrix is written to `path` and reused next run.

    Document-side vectors are a pure function of (texts, model), yet three arms
    recomputed them on every load because their expensive stage was the LLM
    extraction, not the embedding. That held while the embedder was 23M
    parameters; at 0.6B it costs ~12 minutes and saturates the GPU per run.

    The cache key is the model name plus a hash of the texts themselves, so a
    changed corpus, a re-extracted graph, or a swapped embedder all invalidate
    it automatically. That is the failure this guards against: a stale
    384-dimension matrix silently surviving a move to a 1024-dimension model.
    """
    key = hashlib.sha256(
        ("\x00".join([cfg["embed_model"], *texts])).encode()).hexdigest()
    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["key"]) == key:
                return z["v"]
        except Exception:
            pass                     # unreadable or old format -> just rebuild
    v = encode(texts, cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, v=v, key=np.array(key), model=np.array(cfg["embed_model"]))
    return v


def top_k(query_vec, matrix, k):
    """Return [(index, score)] of the k most cosine-similar rows."""
    sims = matrix @ query_vec
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]
