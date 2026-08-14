"""Structural/content validation for the Error Book (Algorithm 1, lines 4–6).

Structural checks compare pages against the wiki/source store deterministically.
Content checks use a separate LLM pass grounded in the raw cited sources.
"""
import json

from core.llm import complete_json


_CONTENT_SYS = """You are a strict Wiki quality validator. Check only the supplied
compiled pages against their cited raw source passages.

Report an unsupported_fact only when a Key Fact, summary, or relation claim is
not supported by ANY cited source. Report a cross_page_contradiction only when
two supplied compiled pages assert incompatible values for the same entity and
attribute. Do not flag wording differences, missing optional facts, or a claim
that is supported by one of several cited sources.

Return JSON exactly as:
{"errors": [{"type": "unsupported_fact|cross_page_contradiction", "slug": "existing page slug", "detail": "specific concise issue"}]}
"""


def structural_validate(wiki, updated_slugs):
    """Return a list of structural errors touching the just-updated pages."""
    errors = []
    for slug in updated_slugs:
        p = wiki.pages.get(slug)
        if not p:
            continue
        # dangling links: target page absent from the index
        for l in p["links"]:
            if l not in wiki.pages:
                errors.append({"type": "dangling_link", "slug": slug, "detail": l})
        # malformed reference: cited source pid not archived
        for s in p["sources"]:
            if s not in wiki.sources:
                errors.append({"type": "malformed_ref", "slug": slug, "detail": s})
        # incomplete page: required sections empty
        if not p["summary"].strip():
            errors.append({"type": "incomplete_page", "slug": slug, "detail": "empty summary"})
        if not p["facts"]:
            errors.append({"type": "incomplete_page", "slug": slug, "detail": "no key facts"})
        if p["type"] not in {"people", "media", "geography", "organizations", "concepts", "events", "misc", "history"}:
            errors.append({"type": "index_inconsistency", "slug": slug,
                           "detail": f"unknown directory type: {p['type']}"})
    return errors


def content_validate(wiki, updated_slugs, model):
    """LLM source-grounding and cross-page consistency check for one batch.

    The raw sources are deliberately included in the prompt; validating a page
    only against another LLM-written digest would merely reproduce its error.
    This is a batch-level call, not one call per page.
    """
    slugs = list(dict.fromkeys(s for s in updated_slugs if s in wiki.pages))
    if not slugs:
        return []
    pids = list(dict.fromkeys(
        pid for slug in slugs for pid in wiki.pages[slug].get("sources", [])
        if pid in wiki.sources
    ))
    pages = [{"slug": slug, "title": wiki.pages[slug]["title"],
              "summary": wiki.pages[slug]["summary"], "facts": wiki.pages[slug]["facts"],
              "links": [{"title": wiki.pages[l]["title"] if l in wiki.pages else l,
                         "relation": wiki.pages[slug].get("link_rel", {}).get(l, "")}
                        for l in wiki.pages[slug]["links"]],
              "sources": wiki.pages[slug]["sources"]}
             for slug in slugs]
    sources = [{"pid": pid, "title": wiki.sources[pid]["title"],
                "text": wiki.sources[pid]["article"]} for pid in pids]
    result = complete_json(_CONTENT_SYS,
                           "COMPILED PAGES:\n" + json.dumps(pages, ensure_ascii=False) +
                           "\n\nCITED RAW SOURCES:\n" + json.dumps(sources, ensure_ascii=False),
                           model, 2500)
    allowed = {"unsupported_fact", "cross_page_contradiction"}
    return [{"type": e["type"], "slug": e["slug"], "detail": str(e.get("detail", ""))[:500]}
            for e in result.get("errors", [])
            if isinstance(e, dict) and e.get("type") in allowed and e.get("slug") in slugs]


def code_autofix(wiki, errors):
    """Layer 1: deterministically repair structural errors. Returns #fixes."""
    fixed = 0
    for e in errors:
        p = wiki.pages.get(e["slug"])
        if not p:
            continue
        if e["type"] == "dangling_link" and e["detail"] in p["links"]:
            p["links"].remove(e["detail"])
            fixed += 1
        elif e["type"] == "malformed_ref" and e["detail"] in p["sources"]:
            p["sources"].remove(e["detail"])
            fixed += 1
    return fixed
