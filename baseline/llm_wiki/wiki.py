"""The compiled Wiki: on-disk Markdown tree (the human-readable artifact) plus
an in-memory manifest (internal index). Renders pages in the Appendix E schema.

This module is the shared substrate: indexing/ writes to it, retrieval/ reads
from it. Category directories are created on demand from the type the LLM
assigns during compilation (never hardcoded).
"""
import json
import os
import re

from rank_bm25 import BM25Okapi

# Unicode-aware, matching the BM25 arm: wiki_search tokenises page titles,
# aliases and tags with this, so an ASCII-only class would make accented pages
# unfindable by their own names.
_WORD = re.compile(r"\w+", re.UNICODE)


def slugify(title):
    s = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")
    return s or "page"


def toks(s):
    return _WORD.findall(s.lower())


class Wiki:
    def __init__(self, root):
        self.root = root
        self.pages = {}          # slug -> page dict
        self.sources = {}        # pid -> {title, article, digest}
        self._bm = None          # cached search index, invalidated on upsert
        os.makedirs(root, exist_ok=True)

    # ---- page mutation ----------------------------------------------------
    def upsert(self, page):
        """page: {title,type,aliases,tags,summary,facts,links,sources}.
        links are page slugs; sources are pids. Merges into existing page."""
        slug = slugify(page["title"])
        cur = self.pages.get(slug, {
            "slug": slug, "title": page["title"], "type": page.get("type", "misc"),
            "aliases": [], "tags": [], "summary": "", "facts": [], "links": [],
            "link_rel": {}, "sources": [],
        })
        cur["type"] = page.get("type", cur["type"])
        # link_rel[target_slug] = how the target relates to this page (Appendix E)
        cur.setdefault("link_rel", {})
        for k, v in (page.get("link_rel") or {}).items():
            if v and not cur["link_rel"].get(k):
                cur["link_rel"][k] = v
        if page.get("summary"):
            cur["summary"] = page["summary"]
        for f in ("aliases", "tags", "links", "sources"):
            for v in page.get(f, []):
                if v and v not in cur[f]:
                    cur[f].append(v)
        for fact in page.get("facts", []):
            if fact and fact not in cur["facts"]:
                cur["facts"].append(fact)
        self.pages[slug] = cur
        self._bm = None
        return slug

    def replace(self, slug, page):
        """Replace one existing page after an LLM content repair.

        Repairs must be able to REMOVE an unsupported fact.  `upsert()` only
        accumulates fields, which is correct during ingestion but would retain
        exactly the bad fact a repair was asked to delete.  Provenance remains
        monotonic: all previously cited source pids are retained, and only
        archived pids supplied by the repair may be added.
        """
        old = self.pages[slug]
        repaired = {
            "slug": slug,
            "title": old["title"],
            "type": page.get("type", old["type"]),
            "aliases": list(dict.fromkeys(page["aliases"] if "aliases" in page else old["aliases"])),
            "tags": list(dict.fromkeys(page["tags"] if "tags" in page else old["tags"])),
            "summary": page.get("summary") or old["summary"],
            "facts": list(dict.fromkeys(page["facts"] if "facts" in page else old["facts"])),
            "links": list(dict.fromkeys(page["links"] if "links" in page else old["links"])),
            "link_rel": page["link_rel"] if "link_rel" in page else old.get("link_rel", {}),
            "sources": list(dict.fromkeys(old["sources"] + [
                s for s in page.get("sources", []) if s in self.sources
            ])),
        }
        self.pages[slug] = repaired
        self._bm = None
        return slug

    def has(self, slug):
        return slug in self.pages

    def index_lines(self):
        """directory-index one-liners: 'People > slug (aliases) #tags'"""
        out = []
        for slug, p in sorted(self.pages.items()):
            al = ", ".join(p["aliases"][:3])
            tg = " ".join("#" + t.replace(" ", "-") for t in p["tags"][:4])
            out.append(f"- [{p['type']}] {slug} ({p['title']}; aliases: {al}) {tg}")
        return out

    # ---- rendering (Markdown files, Appendix E schema) --------------------
    def render_page_md(self, slug):
        p = self.pages[slug]
        lines = [
            "---",
            f"type: {p['type']}",
            f"aliases: [{', '.join(p['aliases'])}]",
            f"tags: [{', '.join(p['tags'])}]",
            "---",
            f"# {p['title']}",
            f"> {p['summary']}",
            "",
            "## Key Facts",
        ]
        lines += [f"- {f}" for f in p["facts"]] or ["- (none)"]
        lines += ["", "## Related Pages"]
        if p["links"]:
            rel = p.get("link_rel") or {}
            for l in p["links"]:
                tt = self.pages[l]["title"] if l in self.pages else l
                # the relation phrase is the traversal affordance: it tells the
                # agent WHY to follow this link. Fall back to the title when the
                # link came from bidirectionalisation (no stated relation).
                lines.append(f"- [[{l}]] -- {rel.get(l) or tt}")
        else:
            lines.append("- (none)")
        lines += ["", "## Related Sources"]
        lines += [f"- [[sources/digests/{s}]]" for s in p["sources"]] or ["- (none)"]
        return "\n".join(lines)

    def render_all(self):
        # clear old md (keep manifest)
        cats = {}
        for slug, p in self.pages.items():
            cats.setdefault(p["type"], []).append(slug)
        # pages + category indices
        for cat, slugs in cats.items():
            d = os.path.join(self.root, cat)
            os.makedirs(d, exist_ok=True)
            # Appendix E: the category index is the browsable listing, and it groups
            # entries under semantic headings ("German Nobility", "Chinese Film
            # Directors") so a large directory stays scannable. We group by the
            # page's most common leading tag -- no LLM call needed.
            for slug in slugs:
                open(os.path.join(d, slug + ".md"), "w").write(self.render_page_md(slug))
            groups = {}
            for slug in slugs:
                p = self.pages[slug]
                key = (p["tags"][0] if p["tags"] else "Other").title()
                groups.setdefault(key, []).append(slug)
            idx = [f"# {cat}", f"> {len(slugs)} pages", ""]
            for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
                idx.append(f"## {key}")
                for slug in sorted(groups[key]):
                    p = self.pages[slug]
                    al = ", ".join(p["aliases"][:2])
                    tg = " ".join("#" + t.replace(" ", "-") for t in p["tags"][:3])
                    idx.append(f"- [[{slug}]] ({al or p['title']}) -- {p['summary']} {tg}")
                idx.append("")
            open(os.path.join(d, "_index.md"), "w").write("\n".join(idx))
        # Global index (Appendix E): a knowledge overview plus a directory catalog
        # that DESCRIBES each directory. It is a router, not a page list -- the
        # paper's 2Wiki index covers 5,825 pages, so enumerating them here is not
        # an option. The browsable listing lives in each <cat>/_index.md.
        _WHAT = {
            "concepts": "theories, methods, genres, categories, abstract ideas",
            "events": "historical events, periods, battles",
            "geography": "cities, villages, countries, airports, landmarks",
            "history": "historical events, periods, medieval history",
            "media": "films, albums, songs, TV shows, books, creative works",
            "organizations": "schools, universities, bands, companies, dynasties",
            "people": "biographies of historical figures, artists, politicians",
            "misc": "entries that did not fit another directory",
        }
        g = ["# Wiki Directory Overview",
             f"> Knowledge Overview: compiled encyclopedic knowledge base over "
             f"{len(self.sources)} source passages, organised as {len(self.pages)} "
             f"interlinked pages across {len(cats)} directories. Read a directory's "
             f"_index.md for its full page listing, then read the pages themselves.",
             "", "## Directory Catalog"]
        for cat, slugs in sorted(cats.items()):
            g.append(f"- {cat}/ -- {_WHAT.get(cat, 'compiled entity pages')} "
                     f"({len(slugs)} pages; listing: `{cat}/_index.md`)")
        g.append("- sources/ -- paragraph digests and original archives")
        open(os.path.join(self.root, "index.md"), "w").write("\n".join(g))
        # sources
        for sub in ("digests", "articles"):
            os.makedirs(os.path.join(self.root, "sources", sub), exist_ok=True)
        for pid, s in self.sources.items():
            open(os.path.join(self.root, "sources", "articles", pid + ".md"), "w").write(
                f"# {s['title']}\n\n{s['article']}")
            open(os.path.join(self.root, "sources", "digests", pid + ".md"), "w").write(
                f"# {s['title']} (digest)\n\n{s['digest']}")
        # manifest (internal)
        json.dump({"pages": self.pages, "sources": self.sources},
                  open(os.path.join(self.root, "_manifest.json"), "w"), indent=1)

    @classmethod
    def load(cls, root):
        w = cls(root)
        m = json.load(open(os.path.join(root, "_manifest.json")))
        w.pages, w.sources = m["pages"], m["sources"]
        return w

    # ---- retrieval tools --------------------------------------------------
    def _bm25(self):
        """BM25 over each page's structured surface, cached until pages change.
        Fields are weighted by repeating their tokens (BM25Okapi has no field
        weights): title x3, aliases/tags x2, then summary and facts once."""
        if self._bm is None:
            slugs = list(self.pages)
            docs, fields = [], []
            for s in slugs:
                p = self.pages[s]
                t = toks(p["title"])
                al = sum([toks(a) for a in p["aliases"]], [])
                tg = sum([toks(x) for x in p["tags"]], [])
                docs.append(t * 3 + al * 2 + tg * 2 + toks(p["summary"])
                            + toks(" ".join(p["facts"])))
                fields.append((set(t), set(al), set(tg)))
            self._bm = (slugs, fields, BM25Okapi(docs) if docs else None)
        return self._bm

    def search(self, query, k=5):
        """wiki_search: BM25 over the structured surface plus an exact-field bonus.

        The previous scorer counted overlapping tokens, so a page sharing only a
        stopword ('a') scored the same integer as a page sharing a rare name, and
        ties were broken by slug order -- at 214 pages that filled the results
        with noise. BM25 gives rare terms their weight and removes the ties."""
        q = toks(query)
        slugs, fields, bm = self._bm25()
        if not bm or not q:
            return []
        raw, qs = bm.get_scores(q), set(q)
        scored = []
        for i, s in enumerate(slugs):
            t, al, tg = fields[i]
            bonus = 2.0 * len(qs & t) + 1.5 * len(qs & al) + 0.5 * len(qs & tg)
            scored.append((raw[i] + bonus, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{"slug": s, "title": self.pages[s]["title"], "type": self.pages[s]["type"],
                 "score": round(float(sc), 2), "summary": self.pages[s]["summary"]}
                for sc, s in scored[:k] if sc > 0]

    def read(self, paths):
        """wiki_read: return rendered markdown for pages / indices."""
        out = []
        for path in paths:
            slug = path.split("/")[-1].replace(".md", "")
            if slug in self.pages:
                out.append(self.render_page_md(slug))
            elif path.endswith("_index.md") or path == "index.md":
                fp = os.path.join(self.root, path)
                out.append(open(fp).read() if os.path.exists(fp) else f"(no such index: {path})")
            elif "digests" in path or "articles" in path:
                fp = os.path.join(self.root, path if path.endswith(".md") else path + ".md")
                out.append(open(fp).read() if os.path.exists(fp) else f"(no such source: {path})")
            else:
                out.append(f"(page not found: {path})")
        return "\n\n---\n\n".join(out)
