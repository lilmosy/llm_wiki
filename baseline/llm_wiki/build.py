"""Index-time Wiki compilation (Algorithm 1).

Per passage:  SELECTPAGES -> COMPILEWIKIPAGES -> validate -> Error Book -> apply.
Two LLM calls per passage (the paper's "compilation cost"). Dangling-link
code-autofix is deferred to finalization so that legitimate cross-page links
(compiled in a later batch) are not destroyed -- matching the paper's
"at finalization, a code-fix loop catches newly introduced errors."
"""
import json

from core.llm import complete, complete_json
from baseline.llm_wiki.wiki import Wiki, slugify
from baseline.llm_wiki.select_pages import select_pages
from baseline.llm_wiki.validators import structural_validate, content_validate, code_autofix
from baseline.llm_wiki.error_book import ErrorBook

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
 "links": [{"title": "Ernest I, Prince of Anhalt-Dessau", "relation": "father of John V"},
           {"title": "Karl I, Prince of Anhalt-Zerbst", "relation": "eldest son and successor of John V"}],
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
            "summary (one line), facts (list), links, "
            f"sources (must be [\"{passage['pid']}\"]). "
            "Create a page for each distinct entity in the passage. "
            "Each entry of 'links' is an object {\"title\": <related entity title>, "
            "\"relation\": <short phrase stating how that entity relates to THIS page, "
            "e.g. 'father of John V', 'director of this film', 'birthplace'>}. "
            "The relation phrase is what lets an agent decide which link to follow "
            "next, so it must name the relationship, never repeat the title.")
    # annotated links (title + relation) make the JSON noticeably longer than the
    # bare-title schema; 1500 truncated mid-object on real 2Wiki passages.
    return complete_json(sys, user, model, 3000)


def apply_pages(page_objs, wiki):
    updated = []
    for po in page_objs:
        if not isinstance(po, dict) or "title" not in po:
            continue
        # links arrive as {"title","relation"} (Appendix E annotates each wikilink
        # with how the target relates to this page). Bare strings still parse so a
        # model that ignores the schema does not lose the link.
        slugs, rel = [], {}
        for l in po.get("links", []):
            if isinstance(l, dict):
                t = l.get("title") or ""
                if not t:
                    continue
                s = slugify(t)
                rel[s] = (l.get("relation") or "").strip()
            else:
                s = slugify(str(l))
            slugs.append(s)
        po["links"], po["link_rel"] = slugs, rel
        po["sources"] = po.get("sources", [])
        updated.append(wiki.upsert(po))
    return updated


_REPAIR_SYS = """You repair compiled Wiki pages using ONLY their cited raw source
passages. Remove unsupported or contradictory facts; retain every supported
fact useful for multi-hop traversal. Preserve entity aliases and useful links,
but never link to an entity absent from the supplied Wiki index.

Return JSON exactly as {"pages": [{"slug": "existing slug", "type": "...",
"aliases": ["..."], "tags": ["..."], "summary": "...", "facts": ["..."],
"links": [{"title": "existing page title", "relation": "..."}]}]}.
Do not create pages. Return only pages whose slug is in REPAIR TARGETS."""


def apply_repaired_pages(page_objs, wiki):
    """Replace repaired page content while retaining complete source provenance."""
    updated = []
    for po in page_objs:
        if not isinstance(po, dict):
            continue
        slug = po.get("slug")
        if slug not in wiki.pages:
            continue
        links, rel = [], {}
        for link in po.get("links", []):
            if isinstance(link, dict):
                title = (link.get("title") or "").strip()
                if not title:
                    continue
                target = slugify(title)
                links.append(target)
                relation = (link.get("relation") or "").strip()
                if relation:
                    rel[target] = relation
            elif link:
                links.append(slugify(str(link)))
        po["links"], po["link_rel"] = links, rel
        updated.append(wiki.replace(slug, po))
    return updated


def llm_periodic_fix(wiki, errors, model):
    """Layer 2 repair for pages with source-grounded/content errors.

    This is called only at the configured periodic boundary or finalisation,
    never during query time. A single call repairs the affected page set.
    """
    slugs = list(dict.fromkeys(e["slug"] for e in errors if e.get("slug") in wiki.pages))
    if not slugs:
        return [], None
    pids = list(dict.fromkeys(
        pid for slug in slugs for pid in wiki.pages[slug].get("sources", [])
        if pid in wiki.sources
    ))
    pages = [{"slug": slug, "title": wiki.pages[slug]["title"],
              "type": wiki.pages[slug]["type"], "aliases": wiki.pages[slug]["aliases"],
              "tags": wiki.pages[slug]["tags"], "summary": wiki.pages[slug]["summary"],
              "facts": wiki.pages[slug]["facts"],
              "links": [{"title": wiki.pages[l]["title"] if l in wiki.pages else l,
                         "relation": wiki.pages[slug].get("link_rel", {}).get(l, "")}
                        for l in wiki.pages[slug]["links"]],
              "sources": wiki.pages[slug]["sources"]}
             for slug in slugs]
    source_docs = [{"pid": pid, "title": wiki.sources[pid]["title"],
                    "text": wiki.sources[pid]["article"]} for pid in pids]
    active_index = [{"slug": slug, "title": p["title"]}
                    for slug, p in wiki.pages.items()]
    try:
        result = complete_json(
            _REPAIR_SYS,
            "REPAIR TARGETS:\n" + json.dumps(slugs) +
            "\n\nDETECTED ERRORS:\n" + json.dumps(errors, ensure_ascii=False) +
            "\n\nCURRENT PAGES:\n" + json.dumps(pages, ensure_ascii=False) +
            "\n\nCITED RAW SOURCES:\n" + json.dumps(source_docs, ensure_ascii=False) +
            "\n\nAVAILABLE WIKI INDEX:\n" + json.dumps(active_index, ensure_ascii=False),
            model, 3000)
        return apply_repaired_pages(result.get("pages", []), wiki), None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def repair_missing_passages(wiki, missing_pids, corpus_by_pid, constraints, model):
    """Retry pages that vanished through an empty compiler emission.

    This is the Layer-2 analogue for a missing page: there is no existing page
    to rewrite, so the only grounded repair is to recompile its raw source with
    the Error Book's accumulated constraints. It is intentionally periodic,
    rather than a hidden immediate retry that would mask a failure in the trace.
    """
    repaired, failures = [], []
    force_emit = constraints + [
        "This source previously emitted no page. You MUST emit at least one complete "
        "page grounded in the source; prefer one complete page to an empty list."
    ]
    for pid in dict.fromkeys(missing_pids):
        passage = corpus_by_pid.get(pid)
        if not passage:
            continue
        try:
            pages = compile_pages(passage, {"update": []}, wiki, force_emit, model)
            if isinstance(pages, dict):
                pages = [pages]
            emitted = apply_pages(pages, wiki)
            if emitted:
                repaired.append(pid)
            else:
                failures.append({"pid": pid, "error": "compiler returned an empty page list"})
        except Exception as e:
            failures.append({"pid": pid, "error": f"{type(e).__name__}: {e}"})
    return repaired, failures


def missing_source_pids(wiki, corpus_by_pid):
    """Raw sources that still have no compiled page after a repair pass."""
    covered = {pid for page in wiki.pages.values() for pid in page.get("sources", [])}
    return sorted(set(corpus_by_pid) - covered)


def unique_repair_errors(errors):
    """One current repair request per (error type, page), never replay history."""
    seen, out = set(), []
    for e in errors:
        key = (e.get("type"), e.get("slug"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


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
    validation_trace, repair_trace = [], []
    repair_review_slugs = set()
    repair_every = int(cfg.get("error_book_revalidate_batches", 10))
    corpus_by_pid = {x["pid"]: x for x in corpus}

    # archive sources first (articles verbatim; digest = short LLM summary folded here)
    for x in corpus:
        digest = complete("Summarize this passage in 1-2 sentences for provenance.",
                          f"{x['title']}: {x['text']}", model, 200).strip()
        wiki.sources[x["pid"]] = {"title": x["title"], "article": x["text"], "digest": digest}

    batch_size = cfg["batch_size"]
    for bi in range(0, len(corpus), batch_size):
        batch = corpus[bi:bi + batch_size]
        binx = bi // batch_size
        batch_updated, batch_errors = [], []
        for passage in batch:
            sel = select_pages(passage, wiki, model, cfg["select_k"])
            fail = None
            try:
                pages = compile_pages(passage, sel, wiki, eb.active_constraints(), model)
                if isinstance(pages, dict):
                    pages = [pages]
            except Exception as e:
                pages, fail = [], f"{type(e).__name__}: {e}"
            updated = apply_pages(pages, wiki)
            errs = structural_validate(wiki, updated)
            # A passage that produced nothing leaves no page for the validators to
            # inspect, so it used to vanish silently -- the exception was swallowed
            # and the Error Book never saw it. Record it as its own error type.
            if not updated:
                errs = errs + [{"type": "empty_emission", "slug": passage["pid"],
                                "detail": fail or "compiler returned an empty page list"}]
            # Layer 1 runs every batch; content checks run once after the batch.
            code_autofix(wiki, [e for e in errs if e["type"] == "malformed_ref"])
            trace.append({"pid": passage["pid"], "batch": binx,
                          "selected": sel["update"], "pages_emitted": [p.get("title") for p in pages],
                          "compile_failure": fail, "errors": errs})
            batch_updated.extend(updated)
            batch_errors.extend(errs)

        # Source-grounded LLM validation is batch-level (Algorithm 1 line 5).
        # A validation failure is recorded rather than misreported as “no issue”.
        content_errors, validation_failure = [], None
        try:
            content_errors = content_validate(wiki, batch_updated, model)
        except Exception as e:
            validation_failure = f"{type(e).__name__}: {e}"
        batch_errors.extend(content_errors)
        eb.discover(batch_errors, binx)
        validation_trace.append({"batch": binx, "updated_slugs": list(dict.fromkeys(batch_updated)),
                                 "structural_errors": len(batch_errors) - len(content_errors),
                                 "content_errors": content_errors,
                                 "validation_failure": validation_failure})

        # Layer 2: periodically repair the pages that have actually failed a
        # validator. The period is 10 batches, matching the paper's setting.
        if repair_every and (binx + 1) % repair_every == 0:
            target_errors = unique_repair_errors([e for e in eb.log if e["type"] in
                                                  {"unsupported_fact", "cross_page_contradiction", "incomplete_page"}])
            repaired, repair_failure = llm_periodic_fix(wiki, target_errors, model)
            repair_review_slugs.update(repaired)
            missing_pids = missing_source_pids(wiki, corpus_by_pid)
            repaired_sources, missing_failures = repair_missing_passages(
                wiki, missing_pids, corpus_by_pid, eb.active_constraints(), model)
            repair_trace.append({"stage": "periodic", "batch": binx,
                                 "target_errors": target_errors, "repaired_slugs": repaired,
                                 "repair_failure": repair_failure,
                                 "repaired_source_pids": repaired_sources,
                                 "missing_page_failures": missing_failures})

    # Finalisation: paper-style three-round code-fix ↔ LLM-fix loop. It runs
    # only during indexing, targets only detected errors, and records every
    # remaining issue rather than treating a failed repair as a clean Wiki.
    bidirectionalize(wiki)
    missing_pids = missing_source_pids(wiki, corpus_by_pid)
    repaired_sources, missing_failures = repair_missing_passages(
        wiki, missing_pids, corpus_by_pid, eb.active_constraints(), model)
    if repaired_sources or missing_failures:
        repair_trace.append({"stage": "final-missing-pages", "repaired_source_pids": repaired_sources,
                             "missing_page_failures": missing_failures})
    bidirectionalize(wiki)
    final_errs = []
    for round_idx in range(3):
        structural = structural_validate(wiki, list(wiki.pages))
        n_fixed = code_autofix(wiki, structural)
        # Every newly compiled page already received a source-grounded batch
        # validation. At finalisation, revalidate only pages touched by an LLM
        # repair; re-sending the entire 156-passage Wiki would be both redundant
        # and liable to exceed the model context window.
        review = sorted(repair_review_slugs)
        if review:
            try:
                content = content_validate(wiki, review, model)
                validation_failure = None
            except Exception as e:
                content, validation_failure = [], f"{type(e).__name__}: {e}"
        else:
            content, validation_failure = [], None
        final_errs = structural + content
        eb.discover(final_errs, binx + 1)
        repaired, repair_failure = llm_periodic_fix(
            wiki, unique_repair_errors([e for e in final_errs if e["type"] in
                                        {"unsupported_fact", "cross_page_contradiction", "incomplete_page"}]), model)
        repair_review_slugs = set(repaired)
        repair_trace.append({"stage": "final", "round": round_idx + 1,
                             "structural_errors": structural, "content_errors": content,
                             "code_fixes": n_fixed, "repaired_slugs": repaired,
                             "validation_failure": validation_failure,
                             "repair_failure": repair_failure})
        bidirectionalize(wiki)
        if not final_errs and not repair_review_slugs:
            break
    # One final pass after the last repair; its result is the actual Error Book
    # state, not the errors observed before repair.
    structural = structural_validate(wiki, list(wiki.pages))
    n_fixed = code_autofix(wiki, structural)
    if repair_review_slugs:
        try:
            content = content_validate(wiki, sorted(repair_review_slugs), model)
        except Exception as e:
            content = []
            repair_trace.append({"stage": "final-postcheck", "validation_failure": f"{type(e).__name__}: {e}"})
    else:
        content = []
    remaining_missing = missing_source_pids(wiki, corpus_by_pid)
    final_errs = structural + content + [
        {"type": "empty_emission", "slug": pid,
         "detail": "no compiled page remains after periodic/final repair"}
        for pid in remaining_missing
    ]
    eb.verify_and_close(binx + 1, final_errs)
    eb.save()
    wiki.render_all()
    return wiki, eb, {"trace": trace, "validation_trace": validation_trace,
                      "repair_trace": repair_trace, "final_errors": final_errs,
                      "final_fixed": n_fixed}
