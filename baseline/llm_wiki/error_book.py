"""The Error Book: persistent self-correction store (error_book.yaml).

Five-stage lifecycle (paper Section 3.3):
  Discover  -> structural + source-grounded content validation
  Attribute -> map error type to a root cause
  Constrain -> a natural-language rule appended to future compile prompts
  Inject    -> active_constraints() feeds COMPILEWIKIPAGES
  (Verify & Close handled at finalize: a type with no new occurrences closes.)
Content errors are repaired by the periodic LLM layer during index construction.
"""
import os
import yaml

# type -> (root cause, injected constraint, layer)
_TAXONOMY = {
    "dangling_link": (
        "Emitted a wikilink to a page that is not present in the index.",
        "NEVER create a [[link]] to a page that is not present in the wiki index. "
        "Only link to entities you also emit as pages or that already exist.",
        "structural"),
    "malformed_ref": (
        "Cited a source id that is not in the provided source set.",
        "Only cite source ids that appear in the passage's own source id. Do not invent source ids.",
        "structural"),
    "incomplete_page": (
        "Produced a page missing a required section (summary or key facts).",
        "Every page MUST have a one-line summary and at least one key fact.",
        "structural"),
    "unsupported_fact": (
        "Stated a fact not grounded in the cited source digest.",
        "Do NOT add entity attributes unless supported by the cited source digest.",
        "content"),
    "cross_page_contradiction": (
        "Two compiled pages assert incompatible values for the same entity attribute.",
        "When updating an existing entity page, reconcile dates and relationships with its cited "
        "sources; do not preserve two incompatible values without qualification.",
        "content"),
    "unseen_overwrite": (
        "An update replaced existing page content without preserving its cited knowledge.",
        "When updating an existing page, merge supported facts and preserve all valid source references.",
        "structural"),
    "index_inconsistency": (
        "A page cannot be reliably located through the directory/index structure.",
        "Use one supported page type and keep title, aliases, tags, and directory entry consistent.",
        "structural"),
    # Not in the paper's seven-category taxonomy (Table 6): every category there
    # describes a page that WAS emitted. A passage that yields no page at all is
    # invisible to those checks, yet it silently removes evidence from the Wiki --
    # 15/80 passages on this corpus, including 5 of the 20 gold passages.
    "empty_emission": (
        "COMPILEWIKIPAGES returned no page for a source passage (truncated JSON, "
        "parse failure, or an empty list), so the passage's entities never entered the Wiki.",
        "Emit at least one page for every passage. If the passage is long, prefer fewer "
        "pages with complete fields over many partial pages, and never truncate the JSON.",
        "structural"),
}


class ErrorBook:
    def __init__(self, path):
        self.path = path
        self.entries = {}   # type -> entry dict
        self.log = []       # chronological occurrences (for the report)

    def discover(self, errors, batch_idx):
        """Attribute + Constrain: record each error type as an open entry."""
        for e in errors:
            self.log.append({"batch": batch_idx, **e})
            t = e["type"]
            if t not in self.entries:
                cause, rule, layer = _TAXONOMY.get(
                    t, ("unknown", "Avoid this error pattern.", "structural"))
                self.entries[t] = {
                    "type": t, "root_cause": cause, "constraint": rule,
                    "layer": layer, "status": "open", "count": 0,
                    "first_batch": batch_idx, "last_batch": batch_idx,
                }
            self.entries[t]["count"] += 1
            self.entries[t]["last_batch"] = batch_idx

    def active_constraints(self):
        """Inject: constraints from every open entry, appended to compile prompt."""
        return [en["constraint"] for en in self.entries.values() if en["status"] == "open"]

    def verify_and_close(self, last_batch, remaining_errors=()):
        """Close only error types absent from an explicit post-repair validation."""
        remaining = {e["type"] for e in remaining_errors}
        for en in self.entries.values():
            if en["type"] not in remaining:
                en["status"] = "closed"

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        yaml.safe_dump({"entries": list(self.entries.values()), "occurrences": self.log},
                       open(self.path, "w"), sort_keys=False, allow_unicode=True)
