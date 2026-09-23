"""Step 1b: modern human prose from US federal sources (public domain).

The literary half of the dataset comes from pre-1929 books, which is why the model
handles novels well and modern business or explanatory prose poorly: on text unlike
anything it saw in training it often plays safe and returns the input unchanged.

Works of the US federal government are not subject to copyright in the United States
(17 U.S.C. 105), so agency writing is the one large source of MODERN human prose that
can be redistributed under a permissive licence. This script collects it, cleans it and
cuts it into passages the same size as the literary ones.

Sources (all US federal), chosen so no single register dominates:
  NASA news releases and science features  - clear explanatory prose for a general reader
  Federal Reserve speeches                 - argued, analytical prose
  Department of Energy articles            - policy and programme writing
  NIST news                                - technical writing made accessible

Run:
    python pipeline/01b_source_gov.py --out data/passages_gov.jsonl --per-source 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gov")

UA = "gohumanize-open-humanizer/0.1 (dataset build; +https://github.com/GoHumanize-ai/gohumanize-open-humanizer)"
PAUSE = 1.0  # one request per second, these are public services

FEEDS = [
    ("nasa-news", "https://www.nasa.gov/news-release/feed/", "NASA"),
    ("nasa-science", "https://science.nasa.gov/feed/", "NASA"),
    ("energy", "https://www.energy.gov/rss/articles.xml", "Department of Energy"),
    # Voice of America is a federal broadcaster, so its own reporting is public domain, and it
    # writes plain journalistic prose rather than press-release prose. Articles that draw on
    # AP, Reuters or AFP copy are NOT public domain and are dropped whole, see WIRE below.
    ("voa", "https://www.voanews.com/api/epiqq", "Voice of America"),
]

# VOA marks anything built on agency copy with a line like "Some information for this report
# came from The Associated Press." That copy is under the agency's rights, not public domain,
# so the whole article is skipped rather than the line alone.
WIRE = re.compile(r"(information for this report came from|"
                  r"(associated press|reuters|agence france|afp) contributed)", re.I)
# Paginated listings, used when the feeds alone do not give enough articles.
# (name, url template, pages, agency, link pattern, site root for relative links)
LISTINGS = [
    ("nasa-news", "https://www.nasa.gov/news-release/page/{}/", range(1, 15), "NASA",
     re.compile(r'href="(https://www\.nasa\.gov/[^"]*?)"'), ""),
    ("fed-speeches", "https://www.federalreserve.gov/newsevents/speech/{}-speeches.htm",
     range(2026, 2021, -1), "Federal Reserve",
     re.compile(r'href="(/newsevents/speech/\w+\d{8}a\.htm)"'), "https://www.federalreserve.gov"),
    # The plain /news-events/news listing ignores ?page= and returns the same twelve items;
    # the search view paginates properly, 25 articles a page.
    ("nist-news", "https://www.nist.gov/news-events/news/search?page={}", range(0, 14),
     "National Institute of Standards and Technology",
     re.compile(r'href="(/news-events/news/20\d\d/\d\d/[a-z0-9-]{8,})"'), "https://www.nist.gov"),
]

# Lines that are navigation, credits or boilerplate rather than prose.
BOILERPLATE = re.compile(
    r"(share this|read more|download|credit:|image credit|media contact|for more information|"
    r"follow us|subscribe|privacy policy|accessibility|last updated|related (terms|links)|"
    r"contact us|newsletter|^\s*by\s+\w+\s+\w+\s*$|return to top|watch the video|view (the )?(pdf|full)|"
    r"speech given|watch live|©|all rights reserved|"
    # Federal site banners, which appear on every page of several agencies.
    r"secure \.gov websites|official website of the united states|https:// means|"
    r"locked padlock|sensitive information only on official|an official website|"
    r"before sharing sensitive information)", re.I)
FOOTNOTE = re.compile(r"^\s*(\d+\.|\[\d+\]|see, e\.g\.|footnote)", re.I)

# Press-release datelines ("GAITHERSBURG, Md. - ", "WASHINGTON - "). They are boilerplate, and a
# model trained on targets that open this way learns to invent a dateline for any text that smells
# like an announcement, which is a fabrication rather than a rewrite.
DATELINE = re.compile(r"^[A-Z][A-Z.\- ]{2,25}(,\s*(?:D\.C\.|[A-Z][a-z]{1,3}\.|[A-Z]{2}))?\s*[—–]\s*")


class Paragraphs(HTMLParser):
    """Collect the text of <p> elements, skipping scripts, styles, navigation and captions."""

    SKIP = {"script", "style", "nav", "header", "footer", "aside", "figcaption", "form", "button"}

    def __init__(self) -> None:
        super().__init__()
        self.depth_skip = 0
        self.in_p = False
        self.buf: list[str] = []
        self.out: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.depth_skip += 1
        elif tag == "p" and not self.depth_skip:
            self.in_p = True
            self.buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.depth_skip:
            self.depth_skip -= 1
        elif tag == "p" and self.in_p:
            self.in_p = False
            text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            if text:
                self.out.append(text)

    def handle_data(self, data: str) -> None:
        if self.in_p and not self.depth_skip:
            self.buf.append(data)


def get(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", "ignore")
        except (urllib.error.URLError, TimeoutError) as error:
            log.warning("%s: %s (attempt %d)", url, error, attempt + 1)
            time.sleep(2 * (attempt + 1))
    return ""


def feed_links(url: str) -> list[str]:
    xml = get(url)
    links = re.findall(r"<link>\s*(?:<!\[CDATA\[)?\s*(https?://[^<\]\s]+)", xml)
    return [l for l in links if "/feed" not in l][1:]  # first <link> is the channel itself


def listing_links(base: str, pages: range, pattern: re.Pattern[str], root: str) -> list[str]:
    found: list[str] = []
    for page in pages:
        html = get(base.format(page))
        if not html:
            continue
        for href in pattern.findall(html):
            url = href if href.startswith("http") else root + href
            if url not in found:
                found.append(url)
        time.sleep(PAUSE)
    return found


def article_text(url: str) -> list[str]:
    html = get(url)
    if not html:
        return []
    if WIRE.search(html):  # partly agency copy, so not ours to redistribute
        return []
    parser = Paragraphs()
    parser.feed(html)
    keep = []
    for para in parser.out:
        para = DATELINE.sub("", para)
        if len(para.split()) < 25:  # headings, captions, one-line notes
            continue
        if BOILERPLATE.search(para) or FOOTNOTE.match(para):
            continue
        letters = sum(c.isalpha() or c.isspace() for c in para)
        if letters / max(len(para), 1) < 0.85:  # tables, figures, reference lists
            continue
        keep.append(para)
    return keep


def passages(paras: list[str], low: int = 80, high: int = 300) -> list[str]:
    """Group consecutive paragraphs into passages of `low` to `high` words."""
    out, buf, count = [], [], 0
    for para in paras:
        words = len(para.split())
        if count + words > high and count >= low:
            out.append("\n\n".join(buf))
            buf, count = [], 0
        buf.append(para)
        count += words
        if count >= high:
            out.append("\n\n".join(buf))
            buf, count = [], 0
    if count >= low:
        out.append("\n\n".join(buf))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/passages_gov.jsonl"))
    ap.add_argument("--per-source", type=int, default=120, help="max articles per source")
    ap.add_argument("--limit-articles", type=int, default=0, help="stop after N articles (smoke test)")
    ap.add_argument("--only", default="", help="comma-separated source names, to top up one source "
                                              "without re-fetching the others")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    articles: list[tuple[str, str, str]] = []  # (source, url, agency)
    for name, url, agency in FEEDS:
        if only and name not in only:
            continue
        links = feed_links(url)[: args.per_source]
        log.info("%s: %d articles from the feed", name, len(links))
        articles += [(name, l, agency) for l in links]
        time.sleep(PAUSE)
    for name, base, pages, agency, pattern, root in LISTINGS:
        if only and name not in only:
            continue
        links = [l for l in listing_links(base, pages, pattern, root)
                 if not any(l == u for _, u, _ in articles)][: args.per_source]
        log.info("%s: %d articles from the listing", name, len(links))
        articles += [(name, l, agency) for l in links]

    if args.limit_articles:
        articles = articles[: args.limit_articles]

    seen_ids: set[str] = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for n, (source, url, agency) in enumerate(articles, 1):
            paras = article_text(url)
            for text in passages(paras):
                pid = f"{source}-{hashlib.sha1(text.encode()).hexdigest()[:16]}"
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                fh.write(json.dumps({"id": pid, "text": text, "source": source, "agency": agency,
                                     "url": url, "category": "modern"}, ensure_ascii=False) + "\n")
                written += 1
            if n % 20 == 0:
                log.info("%d/%d articles, %d passages", n, len(articles), written)
            time.sleep(PAUSE)
    log.info("finished: %d passages from %d articles -> %s", written, len(articles), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
