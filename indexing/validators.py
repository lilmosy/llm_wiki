"""STRUCTURALVALIDATE (Algorithm 1, line 4) + code auto-fix (Error Book Layer 1).

All checks here are deterministic and code-only -- no LLM. They compare the
compiled pages against external ground truth (the wiki filesystem / source set),
not against the model's own judgement.
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
    return errors


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
