"""Send complaint photos (no text) to several OpenRouter decision models and print their answers side by side.

Usage: python image_test.py images/*.jpg      (needs OPENROUTER_API_KEY)
"""
import base64
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from classify import QUESTIONS, call_jev

MODELS = [
    "typesafe/jev-1.14",
    "openai/gpt-6-luna-decisions",
    "google/gemini-decision-1",
    "liquid/d1",
    "cloudflare/clef-flash",
    "perplexity/pplx-decider-v1.3:free",
]
QS = {k: v for k, v in QUESTIONS.items() if k != "adverse_event"}  # a photo cannot show a patient reaction
QS["damage_visible"] = {"type": "noul", "instructions": "Does the photo show a visible defect in the medicine or its packaging?"}


def ask(model, path, key):
    url = "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()
    state = ["Photo attached to a customer complaint about a medicine:", {"type": "image_url", "image_url": {"url": url}}]
    try:
        return call_jev({"id": f"{model} {path}"}, key, model=model, state=state, questions=QS)
    except Exception as e:  # one failing model should not hide the others
        return {"error": str(e)[:160]}


def fmt(resp):
    if "error" in resp:
        return "ERROR " + resp["error"]
    a = resp["answers"]
    if any(v.get("type") == "refusal" for v in a.values()):
        return "refused: " + ", ".join(k for k, v in a.items() if v.get("type") == "refusal")
    tags = ",".join(t[4:] for t in a if t.startswith("tag_") and a[t]["noul"] >= 0.5)
    conf = a["primary_category"].get("confidence")
    return (f"{a['primary_category']['choice']} ({conf if conf is None else round(conf, 2)}) / {a['subcategory']['choice']}"
            f" | tags={tags or '-'} | damage={a['damage_visible']['noul']:.2f} | severity={a['severity']['score']:.1f}"
            f" | ${resp.get('usage', {}).get('cost') or 0:.6f}")


def main():
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("Set OPENROUTER_API_KEY.")
    jobs = [(m, p) for p in sys.argv[1:] for m in MODELS]
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda j: ask(*j, key), jobs))
    for (m, p), r in zip(jobs, out):
        print(f"{os.path.basename(p):<22} {m:<36} {fmt(r)}")


if __name__ == "__main__":
    main()
