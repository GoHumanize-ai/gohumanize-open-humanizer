"""Client tests against a stub endpoint. No API key, no model, no network.

    python3 packages/python/test_client.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gohumanize_open_humanizer import (  # noqa: E402
    DEFAULT_SAMPLES, Humanizer, _pick_most_rewritten, _word_overlap,
)

SOURCE = "It is worth noting that the committee ultimately reached a consensus after lengthy deliberation."
REWRITE = "The committee argued the point for hours, and in the end they agreed on it."
TOO_SHORT = "They agreed."

calls: dict[str, int] = {}


def stub(kind: str, port: int) -> HTTPServer:
    """multi: answers with `n` rewrites. single: ignores `n`, rewrites only on the third ask."""
    calls[kind] = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls[kind] += 1
            assert body["messages"][0]["role"] == "system"
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            text = body["messages"][-1]["content"]
            if kind == "multi":
                assert body.get("n") == DEFAULT_SAMPLES, body.get("n")
                outs = [text, text, REWRITE, TOO_SHORT, text]
            elif kind == "single":
                outs = [text] if calls[kind] < 3 else [REWRITE]
            else:  # never rewrites
                outs = [text]
            payload = json.dumps({"choices": [{"message": {"content": o}} for o in outs]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    assert _word_overlap(SOURCE, SOURCE) == 1.0
    assert _word_overlap(SOURCE, REWRITE) < 0.6
    # A rewrite far shorter than the input is dropped: it has lost content, not style.
    assert _pick_most_rewritten(SOURCE, [SOURCE, TOO_SHORT]) == SOURCE
    assert _pick_most_rewritten(SOURCE, [SOURCE, REWRITE]) == REWRITE
    assert _pick_most_rewritten(SOURCE, []) == ""
    quoted = "\u201c" + REWRITE + "\u201d"
    assert _pick_most_rewritten(SOURCE, [quoted]) == REWRITE, "quotes around the whole answer are removed"
    # Changes more than REWRITE, so only the quote rule can reject it.
    invented = "Hours of argument, then a deal. \u201cWe finally agreed on it,\u201d one member said."
    assert _word_overlap(SOURCE, invented) < _word_overlap(SOURCE, REWRITE)
    assert _pick_most_rewritten(SOURCE, [invented, REWRITE]) == REWRITE, "a rewrite that adds quotes loses"
    assert _pick_most_rewritten(SOURCE, [invented]) == invented, "kept when it is the only rewrite"
    print("quotes: wrapping removed, invented quotes avoided")
    num_src = "It is worth noting that the agency ultimately allocated $2.4 million to 12 projects in 2025."
    lost = "The agency gave a few million dollars to a dozen projects last year, after a long review of them."
    kept = "Last year, 2025, the agency put $2.4 million into 12 projects after its review."
    assert _word_overlap(num_src, lost) < _word_overlap(num_src, kept)
    assert _pick_most_rewritten(num_src, [lost, kept]) == kept, "a rewrite that keeps the numbers wins"
    assert _pick_most_rewritten(num_src, [lost, num_src]) == num_src, "keeping the numbers beats changing more"
    assert _pick_most_rewritten(num_src, [lost, num_src, kept]) == kept, "a real rewrite keeping them beats both"
    print("numbers: a rewrite keeping every figure preferred, even over a bigger change")
    print("selection helpers OK")

    servers = [stub("multi", 8171), stub("single", 8172), stub("stubborn", 8173)]
    try:
        out = Humanizer(base_url="http://127.0.0.1:8171/v1", api_key="k").humanize(SOURCE)
        assert out == REWRITE, out
        assert calls["multi"] == 1, calls["multi"]
        print("multi-sample endpoint: one request, returns the rewrite")

        out = Humanizer(base_url="http://127.0.0.1:8172/v1", api_key="k").humanize(SOURCE)
        assert out == REWRITE, out
        assert calls["single"] == 3, calls["single"]
        print("endpoint ignoring n: asks again until a rewrite comes back")

        out = Humanizer(base_url="http://127.0.0.1:8173/v1", api_key="k").humanize(SOURCE)
        assert out == SOURCE, out
        assert calls["stubborn"] == DEFAULT_SAMPLES, calls["stubborn"]
        print("endpoint that never rewrites: stops after", DEFAULT_SAMPLES, "requests")

        before = calls["multi"]
        single = Humanizer(base_url="http://127.0.0.1:8171/v1", api_key="k", samples=1)
        # samples=1 sends no `n`, which the multi stub asserts on, so use the plain stub.
        single.base_url = "http://127.0.0.1:8173/v1"
        single.humanize(SOURCE)
        assert calls["multi"] == before and calls["stubborn"] == DEFAULT_SAMPLES + 1
        print("samples=1: exactly one request, no retries")

        doc = Humanizer(base_url="http://127.0.0.1:8171/v1", api_key="k").humanize_document(
            f"{SOURCE}\n\n{SOURCE}")
        assert doc.count(REWRITE) == 2, doc
        print("humanize_document: each paragraph rewritten separately")
    finally:
        for server in servers:
            server.shutdown()
    print("all client tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
