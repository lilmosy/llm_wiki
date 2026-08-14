"""Query-time tools exposed to the agent (Section 3.2): wiki_search / wiki_read.

Thin wrappers over the Wiki store's search/read so the ReAct agent (agent.py)
treats them as callable tools. TOOLS holds the Anthropic tool-use schemas.
"""

TOOLS = [
    {"name": "wiki_search",
     "description": "Search the Wiki index by page names, aliases, tags, and summaries. Returns candidate pages.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "wiki_read",
     "description": "Batch-read wiki pages (by slug) or indices. Returned page content includes [[links]] to related pages for the next hop.",
     "input_schema": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]}},
]


def wiki_search(wiki, query, k):
    """Structure-first search over the index; returns candidate page hits."""
    return wiki.search(query, k)


def wiki_read(wiki, paths):
    """Read pages/indices by slug; returned content carries [[links]] for the next hop."""
    return wiki.read(paths)
