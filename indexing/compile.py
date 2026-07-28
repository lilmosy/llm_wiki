"""Index-time Wiki compilation (Algorithm 1).

Per passage:  SELECTPAGES -> COMPILEWIKIPAGES -> validate -> Error Book -> apply.
Two LLM calls per passage (the paper's "compilation cost"). Dangling-link
code-autofix is deferred to finalization so that legitimate cross-page links
(compiled in a later batch) are not destroyed -- matching the paper's
"at finalization, a code-fix loop catches newly introduced errors."
"""
import json

from llm import complete, complete_json
from wiki import Wiki, slugify
from indexing.select_pages import select_pages
from indexing.validators import structural_validate, code_autofix
from indexing.error_book import ErrorBook

# Appendix E: the exact John V page, used as a few-shot output anchor.
_ANCHOR = """--- EXAMPLE OUTPUT PAGE (schema anchor, do not copy its content) ---
{
 "title": "John V, Prince of Anhalt-Zerbst",
 "type": "people",
 "aliases": ["John V of Anhalt-Zerbst", "Johann V von Anhalt-Zerbst"],
 "tags": ["German nobility", "House of Ascania", "Prince"],
 "summary": "German prince of the House of Ascania who ruled Anhalt-Zerbst from 1544",
 "facts": ["Born on 4 September 1504 in Dessau; died 4 February 1551 in Zerbst",
           "Second but eldest surviving son of Ernest I, Prince of Anhalt-Dessau"],
 "links": ["Ernest I, Prince of Anhalt-Dessau", "Karl I, Prince of Anhalt-Zerbst"],
 "sources": ["p09"]
}"""


def compile_pages(passage, selected, wiki, constraints, model):
    sel_ctx = ""
    for slug in selected["update"]:
        p = wiki.pages[slug]
        sel_ctx += f"\n- existing page {slug}: {p['summary']} | facts: {p['facts']}"
    cons = "\n".join(f"- {c}" for c in constraints) or "- (none yet)"
    sys = ("You perform COMPILEWIKIPAGES: compile a raw passage into one or more "
           "structured, entity-centric Wiki pages. Reorganize the passage's facts "
           "around the ENTITIES it describes (one page per entity), gather facts onto "
           "each entity, and link related entities.")
    user = (f"{_ANCHOR}\n\nACTIVE CONSTRAINTS (from the Error Book):\n{cons}\n\n"
            f"RELEVANT EXISTING PAGES:{sel_ctx or ' (none)'}\n\n"
            f"PASSAGE (source id = {passage['pid']}):\n{passage['title']}: {passage['text']}\n\n"
            "Output a JSON list of page objects with keys: title, type "
            "(people|media|geography|organizations|concepts|events|misc), aliases, tags, "
            "summary (one line), facts (list), links (titles of related entities), "
            f"sources (must be [\"{passage['pid']}\"]). "
            "Create a page for each distinct entity in the passage. Reference other "
            "entities by their title in 'links'.")
    return complete_json(sys, user, model, 1500)


def apply_pages(page_objs, wiki):
    updated = []
    for po in page_objs:
        if not isinstance(po, dict) or "title" not in po:
            continue
        po["links"] = [slugify(t) for t in po.get("links", [])]   # titles -> slugs
        po["sources"] = po.get("sources", [])
        updated.append(wiki.upsert(po))
    return updated


def bidirectionalize(wiki):
    """Make links bidirectional where the target exists (paper: bidirectional wikilinks)."""
    for slug, p in list(wiki.pages.items()):
        for l in p["links"]:
            if l in wiki.pages and slug not in wiki.pages[l]["links"]:
                wiki.pages[l]["links"].append(slug)


def run_compile(corpus, cfg, wiki_root, eb_path):
    model = cfg["model"]
    wiki = Wiki(wiki_root)
    eb = ErrorBook(eb_path)
    trace = []   # per-passage record for the report

    # archive sources first (articles verbatim; digest = short LLM summary folded here)
    for x in corpus:
        digest = complete("Summarize this passage in 1-2 sentences for provenance.",
                          f"{x['title']}: {x['text']}", model, 200).strip()
        wiki.sources[x["pid"]] = {"title": x["title"], "article": x["text"], "digest": digest}

    batch_size = cfg["batch_size"]
    for bi in range(0, len(corpus), batch_size):
        batch = corpus[bi:bi + batch_size]
        binx = bi // batch_size
        for passage in batch:
            sel = select_pages(passage, wiki, model, cfg["select_k"])
            try:
                pages = compile_pages(passage, sel, wiki, eb.active_constraints(), model)
                if isinstance(pages, dict):
                    pages = [pages]
            except Exception as e:
                pages = []
            updated = apply_pages(pages, wiki)
            errs = structural_validate(wiki, updated)
            eb.discover(errs, binx)
            # immediate Layer-1 fix for non-link structural errors (malformed refs)
            code_autofix(wiki, [e for e in errs if e["type"] == "malformed_ref"])
            trace.append({"pid": passage["pid"], "batch": binx,
                          "selected": sel["update"], "pages_emitted": [p.get("title") for p in pages],
                          "errors": errs})

    # finalization: bidirectional links, then full structural fix (true danglings)
    bidirectionalize(wiki)
    final_errs = structural_validate(wiki, list(wiki.pages))
    eb.discover(final_errs, binx + 1)
    n_fixed = code_autofix(wiki, final_errs)
    eb.verify_and_close(binx + 1)
    eb.save()
    wiki.render_all()
    return wiki, eb, {"trace": trace, "final_errors": final_errs, "final_fixed": n_fixed}
