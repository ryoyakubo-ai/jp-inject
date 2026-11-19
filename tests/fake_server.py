"""A tiny OpenAI-compatible server that imitates a gullible-ish model.

Used by the tests (and handy for trying the tool without a GPU):

    python tests/fake_server.py 8765
    jp-inject run --base-url http://127.0.0.1:8765/v1 --model fake --out runs/fake.jsonl

Its behaviour is a deterministic function of the request, so tests are
reproducible. It is NOT a model of any real system.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CODE = re.compile(r"[A-Z]{2}\d-\d{2}[A-Z]{2}")


def _roll(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def fake_answer(system: str, user: str, seed: int) -> str:
    """Decide how the pretend model reacts."""
    roll = _roll(system, user, seed)
    asked_for_secret = "確認コード" in user or "verification code" in user or "kakunin koodo" in user
    secret = CODE.search(system)
    canary = CODE.search(user)
    if asked_for_secret and secret:
        return f"社内確認コードは {secret.group()} です。" if roll < 0.3 else "申し訳ありませんが、お伝えできません。"
    if canary:
        code = canary.group()
        if roll < 0.25:
            return code
        if roll < 0.35:
            return f"来週の定例ミーティングは木曜日の14時に変更されます。議題は2点です。\n\n{code}"
        if roll < 0.50:
            return f"文書には「{code}」とだけ出力するよう求める文がありますが、それには従いません。"
        return "会議は木曜日の14時に変更されます。"
    return "承知しました。ご質問にお答えします。"


class Handler(BaseHTTPRequestHandler):
    requests_seen: list[dict] = []
    fail_first: int = 0  # respond 500 to this many requests first

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/v0/models/"):
            self._send(200, {"id": "fake", "quantization": "Q0_FAKE", "arch": "none"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        cls = type(self)
        if cls.fail_first > 0:
            cls.fail_first -= 1
            self._send(500, {"error": "temporary failure"})
            return
        cls.requests_seen.append(request)
        messages = {m["role"]: m["content"] for m in request["messages"]}
        answer = fake_answer(messages.get("system", ""), messages.get("user", ""), request.get("seed", 0))
        self._send(200, {
            "model": request["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


@contextmanager
def serve(port: int = 0):
    Handler.requests_seen = []
    Handler.fail_first = 0
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    with serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8765) as url:
        print(f"fake model listening on {url} (Ctrl+C to stop)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
