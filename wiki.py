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

_WORD = re.compile(r"[a-z0-9]+")


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
        os.makedirs(root, exist_ok=True)

    # ---- page mutation ----------------------------------------------------
    def upsert(self, page):
        """page: {title,type,aliases,tags,summary,facts,links,sources}.
        links are page slugs; sources are pids. Merges into existing page."""
        slug = slugify(page["title"])
        cur = self.pages.get(slug, {
            "slug": slug, "title": page["title"], "type": page.get("type", "misc"),
            "aliases": [], "tags": [], "summary": "", "facts": [], "links": [], "sources": [],
        })
        cur["type"] = page.get("type", cur["type"])
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
            for l in p["links"]:
                tt = self.pages[l]["title"] if l in self.pages else l
                lines.append(f"- [[{l}]] -- {tt}")
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
            idx = [f"# {cat}", f"> {len(slugs)} pages", ""]
            for slug in sorted(slugs):
                p = self.pages[slug]
                open(os.path.join(d, slug + ".md"), "w").write(self.render_page_md(slug))
                idx.append(f"- [[{slug}]] ({p['title']}) " + " ".join("#" + t.replace(" ", "-") for t in p["tags"][:3]))
            open(os.path.join(d, "_index.md"), "w").write("\n".join(idx))
        # global index
        g = ["# Wiki Directory Overview", "> Compiled knowledge base (LLM-Wiki MVP)", "", "## Directory Catalog"]
        for cat, slugs in sorted(cats.items()):
            g.append(f"- {cat}/ -- {len(slugs)} pages")
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
    def search(self, query, k=5):
        """wiki_search: prioritise structured signals (name/alias/tag/summary),
        fall back to BM25 over page content."""
        q = set(toks(query))
        scored = []
        for slug, p in self.pages.items():
            struct = set(toks(p["title"])) | set(sum([toks(a) for a in p["aliases"]], [])) \
                | set(sum([toks(t) for t in p["tags"]], [])) | set(toks(p["summary"]))
            s = len(q & struct)
            if s:
                scored.append((s, slug))
        if scored:
            scored.sort(reverse=True)
            hits = [slug for _, slug in scored[:k]]
        else:  # content fallback: BM25 over facts+summary
            slugs = list(self.pages)
            corpus = [toks(self.pages[s]["summary"] + " " + " ".join(self.pages[s]["facts"])) for s in slugs]
            if not corpus:
                return []
            bm = BM25Okapi(corpus)
            order = sorted(range(len(slugs)), key=lambda i: bm.get_scores(list(q))[i], reverse=True)
            hits = [slugs[i] for i in order[:k]]
        return [{"slug": s, "title": self.pages[s]["title"],
                 "type": self.pages[s]["type"], "summary": self.pages[s]["summary"]} for s in hits]

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
