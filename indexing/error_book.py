"""The Error Book: persistent self-correction store (error_book.yaml).

Five-stage lifecycle (paper Section 3.3), MVP scope = structural side only:
  Discover  -> validators.structural_validate
  Attribute -> map error type to a root cause
  Constrain -> a natural-language rule appended to future compile prompts
  Inject    -> active_constraints() feeds COMPILEWIKIPAGES
  (Verify & Close handled at finalize: a type with no new occurrences closes.)
Content-level errors are logged as open entries but not repaired (Layer 2 = future work).
"""
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

    def verify_and_close(self, last_batch):
        """A structural type not seen in the final batch is considered mitigated."""
        for en in self.entries.values():
            if en["layer"] == "structural" and en["last_batch"] < last_batch:
                en["status"] = "closed"

    def save(self):
        yaml.safe_dump({"entries": list(self.entries.values()), "occurrences": self.log},
                       open(self.path, "w"), sort_keys=False, allow_unicode=True)
