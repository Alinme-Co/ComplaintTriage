"""Local web demo for complaint triage with OpenRouter decision models.

Usage: OPENROUTER_API_KEY=sk-or-v1-... python server.py   then open http://127.0.0.1:8765
The key stays on this server; the browser never sees it.
"""
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from classify import call_jev, route

HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 8765))
ROOT = os.path.dirname(os.path.abspath(__file__))
PRIMARY = "perplexity/pplx-decider-v1.3:free"  # best on the 40-complaint benchmark, see report.md
FALLBACK = "typesafe/jev-1.14"
MODELS = [PRIMARY, FALLBACK, "openai/gpt-6-luna-decisions", "google/gemini-decision-1", "liquid/d1", "cloudflare/clef-flash"]
MAX_BODY = 8 * 1024 * 1024
KEY = os.environ.get("OPENROUTER_API_KEY")


def classify(payload):
    text = str(payload.get("text") or "").strip()[:4000]
    image = payload.get("image") or ""
    product = str(payload.get("product_type") or "medicine")[:40]
    model = payload.get("model") if payload.get("model") in MODELS else PRIMARY
    if image and not re.fullmatch(r"data:image/(png|jpeg);base64,[A-Za-z0-9+/=]+", image):
        raise ValueError("Photo must be a PNG or JPEG.")
    if not text and not image:
        raise ValueError("Describe the complaint or add a photo.")
    state = [f"Customer complaint about {product}:"]
    if text:
        state.append(text)
    if image:
        state += ["Photo attached by the customer:", {"type": "image_url", "image_url": {"url": image}}]
    start, used = time.time(), model
    try:
        resp = call_jev({"id": model}, KEY, model=model, state=state)
    except Exception:
        if model == FALLBACK:
            raise
        used = FALLBACK
        resp = call_jev({"id": FALLBACK}, KEY, model=FALLBACK, state=state)
    r, reasons = route(text, resp["answers"])
    return {"model": used, "fallback": used != model, "answers": resp["answers"], "route": r, "reasons": reasons,
            "cost": resp.get("usage", {}).get("cost"), "ms": round((time.time() - start) * 1000)}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send(200, open(os.path.join(ROOT, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if path == "/api/models":
            return self.send(200, {"models": MODELS, "default": PRIMARY})
        m = re.fullmatch(r"/images/([\w-]+\.jpg)", path)
        if m and os.path.isfile(os.path.join(ROOT, "images", m[1])):
            return self.send(200, open(os.path.join(ROOT, "images", m[1]), "rb").read(), "image/jpeg")
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/classify":
            return self.send(404, {"error": "Not found"})
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            return self.send(413, {"error": "Photo too large. Use one under 6 MB."})
        try:
            self.send(200, classify(json.loads(self.rfile.read(size) or b"{}")))
        except ValueError as e:
            self.send(400, {"error": str(e)})
        except Exception as e:
            self.send(502, {"error": f"The decision model did not answer: {str(e)[:200]}"})


if __name__ == "__main__":
    if not KEY:
        raise SystemExit("Set OPENROUTER_API_KEY first.")
    print(f"Complaint triage demo on http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
