"""Step 1: source public-domain human prose from Project Gutenberg.

Downloads a curated list of English-language books published before 1929
(all in the US public domain), strips the Project Gutenberg boilerplate,
cleans typographic artefacts, and cuts each book into paragraph-sized
passages of 80-300 words. Output is one JSONL row per passage.

Only English-language originals are used. Translations are excluded on
purpose: the translation itself can carry a newer copyright even when the
original work is public domain.

Run:
    python pipeline/01_source_gutenberg.py --out data/passages_raw.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("source")

# (gutenberg_id, category). Titles/authors are read from the file header at
# download time and stored with each passage, so the list stays auditable.
BOOKS: list[tuple[int, str]] = [
    # fiction
    (1342, "fiction"),   # Pride and Prejudice, Austen (1813)
    (158, "fiction"),    # Emma, Austen (1815)
    (11, "fiction"),     # Alice's Adventures in Wonderland, Carroll (1865)
    (1661, "fiction"),   # The Adventures of Sherlock Holmes, Doyle (1892)
    (98, "fiction"),     # A Tale of Two Cities, Dickens (1859)
    (1400, "fiction"),   # Great Expectations, Dickens (1861)
    (730, "fiction"),    # Oliver Twist, Dickens (1838)
    (2701, "fiction"),   # Moby-Dick, Melville (1851)
    (74, "fiction"),     # The Adventures of Tom Sawyer, Twain (1876)
    (76, "fiction"),     # Adventures of Huckleberry Finn, Twain (1884)
    (345, "fiction"),    # Dracula, Stoker (1897)
    (84, "fiction"),     # Frankenstein, Shelley (1818)
    (768, "fiction"),    # Wuthering Heights, E. Bronte (1847)
    (1260, "fiction"),   # Jane Eyre, C. Bronte (1847)
    (145, "fiction"),    # Middlemarch, Eliot (1871)
    (16, "fiction"),     # Peter Pan, Barrie (1911)
    (25344, "fiction"),  # The Scarlet Letter, Hawthorne (1850)
    (219, "fiction"),    # Heart of Darkness, Conrad (1899)
    (36, "fiction"),     # The War of the Worlds, Wells (1898)
    (35, "fiction"),     # The Time Machine, Wells (1895)
    (43, "fiction"),     # Strange Case of Dr Jekyll and Mr Hyde, Stevenson (1886)
    (120, "fiction"),    # Treasure Island, Stevenson (1883)
    (174, "fiction"),    # The Picture of Dorian Gray, Wilde (1890)
    (2814, "fiction"),   # Dubliners, Joyce (1914)
    (64317, "fiction"),  # The Great Gatsby, Fitzgerald (1925)
    (55, "fiction"),     # The Wonderful Wizard of Oz, Baum (1900)
    (113, "fiction"),    # The Secret Garden, Burnett (1911)
    (514, "fiction"),    # Little Women, Alcott (1868)
    (45, "fiction"),     # Anne of Green Gables, Montgomery (1908)
    (1952, "fiction"),   # The Yellow Wallpaper, Gilman (1892)
    (244, "fiction"),    # A Study in Scarlet, Doyle (1887)
    (541, "fiction"),    # The Age of Innocence, Wharton (1920)
    (2775, "fiction"),   # The Good Soldier, Ford (1915)
    (271, "fiction"),    # Black Beauty, Sewell (1877)
    # non-fiction: essays, speeches, letters, autobiography, science, philosophy
    (205, "nonfiction"),    # Walden, Thoreau (1854)
    (16643, "nonfiction"),  # Essays, First Series, Emerson (1841)
    (20203, "nonfiction"),  # Autobiography of Benjamin Franklin (1791)
    (23, "nonfiction"),     # Narrative of the Life of Frederick Douglass (1845)
    (1228, "nonfiction"),   # On the Origin of Species, 1st ed., Darwin (1859)
    (1404, "nonfiction"),   # The Federalist Papers (1788)
    (147, "nonfiction"),    # Common Sense, Paine (1776)
    (34901, "nonfiction"),  # On Liberty, Mill (1859)
    (3207, "nonfiction"),   # Leviathan, Hobbes (1651)
    (5827, "nonfiction"),   # The Problems of Philosophy, Russell (1912)
    (408, "nonfiction"),    # The Souls of Black Folk, Du Bois (1903)
    (1080, "nonfiction"),   # A Modest Proposal, Swift (1729)
    (2130, "nonfiction"),   # Speeches and Addresses, Lincoln (collection)
    (3300, "nonfiction"),   # The Wealth of Nations, Smith (1776)
    (1232, "nonfiction"),   # (placeholder, skipped if it turns out to be a translation)
    (3600, "nonfiction"),   # (placeholder, skipped if it turns out to be a translation)
]

# Some older Gutenberg files carry no "Title:"/"Author:" header lines. Fallback values.
FALLBACK_META = {
    730: ("Oliver Twist", "Charles Dickens"), 84: ("Frankenstein", "Mary Wollstonecraft Shelley"),
    1260: ("Jane Eyre", "Charlotte Bronte"), 145: ("Middlemarch", "George Eliot"),
    25344: ("The Scarlet Letter", "Nathaniel Hawthorne"), 36: ("The War of the Worlds", "H. G. Wells"),
    120: ("Treasure Island", "Robert Louis Stevenson"), 64317: ("The Great Gatsby", "F. Scott Fitzgerald"),
    45: ("Anne of Green Gables", "L. M. Montgomery"), 244: ("A Study in Scarlet", "Arthur Conan Doyle"),
    271: ("Black Beauty", "Anna Sewell"), 205: ("Walden", "Henry David Thoreau"),
    1228: ("On the Origin of Species", "Charles Darwin"), 3300: ("The Wealth of Nations", "Adam Smith"),
}

# Books whose Gutenberg text is a translation are skipped even if listed.
KNOWN_TRANSLATIONS = {1232, 3600}

UA = {"User-Agent": "gohumanize-demo-humanizer/0.1 (dataset build; contact via GitHub)"}


def fetch(book_id: int) -> str | None:
    for url in (
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="replace")
        except Exception as e:  # noqa: BLE001
            log.debug("miss %s: %s", url, e)
    return None


def header_field(text: str, name: str) -> str:
    head = text[:20000].replace("\r", "")
    m = re.search(rf"^{name}:\s*(.+)$", head, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # older files only carry "The Project Gutenberg eBook of TITLE, by AUTHOR" on line 1
    m = re.search(r"Project Gutenberg'?s? eBook of (.+?)(?:, by (.+?))?\s*$", head.split("\n")[0], re.I)
    if m:
        return (m.group(1) if name.lower() == "title" else (m.group(2) or "")).strip()
    return ""


def strip_envelope(text: str) -> str:
    s = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I | re.S)
    if s:
        text = text[s.end():]
    e = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I | re.S)
    if e:
        text = text[: e.start()]
    return text


HEADING_RE = re.compile(
    r"^\s*(chapter|book|part|volume|letter|section|act|scene|epilogue|prologue|preface|contents|"
    r"introduction|appendix|footnotes?|illustrations?|[ivxlcdm]+\.?|\d+\.?)\b.*$",
    re.I,
)


def clean_paragraph(p: str) -> str:
    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r"\[Illustration[^\]]*\]", "", p, flags=re.I)
    p = re.sub(r"\[\d+\]|\{\d+\}", "", p)           # footnote markers
    p = re.sub(r"_([^_]{1,200})_", r"\1", p)         # _italics_
    p = p.replace("--", " - ").replace("—", " - ")
    p = re.sub(r"\s+-\s+", " - ", p)
    p = re.sub(r"\s([,.;:!?])", r"\1", p)
    p = re.sub(r"\s{2,}", " ", p).strip()
    return p


def is_prose(p: str) -> bool:
    words = p.split()
    if len(words) < 15:
        return False
    if HEADING_RE.match(p) and len(words) < 12:
        return False
    letters = sum(c.isalpha() for c in p)
    if letters / max(len(p), 1) < 0.75:
        return False
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    if caps / len(words) > 0.2:
        return False
    if re.search(r"gutenberg|e-?text|transcriber|proofread|copyright|all rights reserved", p, re.I):
        return False
    return True


def passages(book_text: str, lo: int, hi: int) -> list[str]:
    """Group consecutive clean paragraphs until the passage is within [lo, hi] words."""
    paras = [clean_paragraph(p) for p in re.split(r"\n\s*\n", book_text)]
    paras = [p for p in paras if p and is_prose(p)]
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for p in paras:
        w = len(p.split())
        if w > hi:
            buf, n = [], 0
            continue
        if n + w > hi:
            if lo <= n <= hi:
                out.append("\n\n".join(buf))
            buf, n = [], 0
        buf.append(p)
        n += w
        if n >= lo:
            out.append("\n\n".join(buf))
            buf, n = [], 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-book", type=int, default=80, help="max passages kept per book")
    ap.add_argument("--word-min", type=int, default=80)
    ap.add_argument("--word-max", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cache", type=Path, default=Path("data/gutenberg_cache"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []
    manifest: list[dict] = []

    for book_id, category in BOOKS:
        if book_id in KNOWN_TRANSLATIONS:
            log.info("skip %d (translation)", book_id)
            continue
        cache_file = args.cache / f"{book_id}.txt"
        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8")
        else:
            text = fetch(book_id)
            if text is None:
                log.warning("FAILED download %d", book_id)
                manifest.append({"gutenberg_id": book_id, "status": "download_failed"})
                continue
            cache_file.write_text(text, encoding="utf-8")
            time.sleep(1.0)  # be polite to gutenberg.org
        title = header_field(text, "Title") or FALLBACK_META.get(book_id, ("", ""))[0]
        author = header_field(text, "Author") or FALLBACK_META.get(book_id, ("", ""))[1]
        language = header_field(text, "Language")
        translator = header_field(text, "Translator")
        if translator or language.lower() not in ("english", ""):
            log.warning("skip %d: translator=%r language=%r", book_id, translator, language)
            manifest.append({"gutenberg_id": book_id, "title": title, "status": "skipped_translation"})
            continue
        body = strip_envelope(text)
        cands = passages(body, args.word_min, args.word_max)
        rng.shuffle(cands)
        kept = 0
        for p in cands:
            h = hashlib.sha1(p.lower().encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            rows.append({
                "id": f"pg{book_id}-{h}",
                "gutenberg_id": book_id,
                "title": title,
                "author": author,
                "category": category,
                "word_count": len(p.split()),
                "text": p,
            })
            kept += 1
            if kept >= args.per_book:
                break
        manifest.append({"gutenberg_id": book_id, "title": title, "author": author,
                         "category": category, "candidates": len(cands), "kept": kept, "status": "ok"})
        log.info("%-6d %-45.45s %-22.22s cand=%5d kept=%3d", book_id, title, author, len(cands), kept)

    rng.shuffle(rows)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out.parent / "books_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    n_f = sum(1 for r in rows if r["category"] == "fiction")
    log.info("wrote %d passages (%d fiction, %d nonfiction) to %s", len(rows), n_f, len(rows) - n_f, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
