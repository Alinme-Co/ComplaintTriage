"""Classify synthetic pharmaceutical product complaints with TypeSafe Jev (OpenRouter Decisions API).

Usage:
    python classify.py complaints.json      # live run, needs OPENROUTER_API_KEY
    python classify.py --selftest           # offline checks, no network

Jev only suggests labels. Urgent escalation is decided by deterministic rules in this file,
and every complaint still goes to a human QA reviewer.
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.14")
WORKERS = 8
CONF_MIN = 0.8  # below this the primary category goes to human review
RETRY_CODES = {429, 500, 502, 503, 524, 529}

CATEGORIES = {
    "physical": "Something is wrong with the appearance or physical condition of the medicine itself: broken, chipped, discolored, odd smell, abnormal texture.",
    "packaging": "Something is wrong with the container or packaging: damaged blister, leaking or cracked bottle, broken seal, closure fault, wrong pack.",
    "labeling": "Printed information is wrong, conflicting, missing or unreadable: strength, expiry, instructions, leaflet.",
    "contamination": "Something unexpected is in or on the medicine: foreign particle, hair, glass, mold or microbial growth, residue.",
    "quantity": "The amount is wrong: missing tablets, under-filled bottle, missing strip or component.",
    "other": "Not a product quality issue, or too vague to tell which defect is meant.",
}
SUBCATEGORIES = {
    "broken_tablets": "Tablets broken, chipped or crumbling.",
    "discoloration": "Color differs from normal.",
    "unusual_odor": "Unusual smell.",
    "abnormal_texture": "Abnormal texture, stickiness or viscosity.",
    "damaged_blister": "Blister or foil damaged.",
    "leaking_bottle": "Bottle leaking or cracked.",
    "broken_seal": "Seal broken, missing or tampered.",
    "damaged_closure": "Cap or closure does not work.",
    "incorrect_packaging": "Wrong product or pack inside the carton.",
    "wrong_strength": "Conflicting or wrong strength on labels.",
    "missing_expiry": "Expiry date missing.",
    "incorrect_instructions": "Instructions wrong, conflicting or not understandable.",
    "unreadable_label": "Label cannot be read.",
    "foreign_particle": "Visible foreign particle, hair, glass, specks.",
    "suspected_microbial": "Mold, fluffy growth or other signs of microbial contamination.",
    "unexpected_residue": "Film, powder or residue that should not be there.",
    "missing_tablets": "Fewer tablets than stated.",
    "incorrect_fill": "Bottle under- or over-filled.",
    "incomplete_package": "A strip or component is missing.",
    "other": "None of the above or unclear.",
}
TAG_QUESTIONS = {
    "physical": "Does the complaint describe a physical defect of the medicine itself (broken, discolored, odor, texture)?",
    "packaging": "Does the complaint describe a defect of the packaging, container, seal or closure?",
    "labeling": "Does the complaint describe wrong, conflicting, missing or unreadable printed information?",
    "contamination": "Does the complaint report something foreign or unexpected in or on the medicine?",
    "quantity": "Does the complaint report a wrong amount of medicine or a missing component?",
}
URGENT_SUBCATS = {"wrong_strength", "foreign_particle", "suspected_microbial", "incorrect_packaging", "broken_seal"}

# Hard rules run on the raw text, independent of the model.
HARD_RULES = {
    "contamination word": re.compile(r"\b(particle|speck|hair|glass|metal|insect|mou?ld|fungus|growth|fib(er|re))", re.I),
    "adverse event word": re.compile(r"\b(rash|hives|vomit\w*|allerg\w*|dizz\w*|swell\w*|seizure|hospitali[sz]ed|emergency|faint\w*|breath\w*)", re.I),
    "seal/tamper": re.compile(r"tamper\w*|\bseal\w*\b.{0,40}\b(broken|missing|open)", re.I),
}


def hard_rules(text):
    reasons = [name for name, rx in HARD_RULES.items() if rx.search(text)]
    by_unit = {}
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml)\b", text, re.I):
        by_unit.setdefault(unit.lower(), set()).add(float(value))
    if any(len(v) > 1 for v in by_unit.values()):
        reasons.append("conflicting strength/dose")
    return reasons


def build_questions():
    q = {
        "primary_category": {"type": "choice", "instructions": "Which category best describes the main problem in the complaint?", "criteria": CATEGORIES},
        "subcategory": {"type": "choice", "instructions": "Which subcategory best describes the main problem in the complaint?", "criteria": SUBCATEGORIES},
        "sufficient_info": {"type": "noul", "instructions": "Does the complaint describe a specific, concrete problem with enough detail to start a quality classification?"},
        "adverse_event": {"type": "noul", "instructions": "Does the complaint mention that a person had a symptom, reaction or health harm?"},
        "severity": {"type": "score", "instructions": "How serious is the potential quality impact described?", "criteria": [
            "Minor: cosmetic or convenience issue, no plausible effect on the dose or patient.",
            "Major: could affect dose, quality or correct use.",
            "Critical: could directly harm a patient (wrong strength, contamination, reaction reported)."]},
    }
    for tag, text in TAG_QUESTIONS.items():
        q["tag_" + tag] = {"type": "noul", "instructions": text}
    return q


QUESTIONS = build_questions()


def call_jev(item, key, model=MODEL, state=None, questions=QUESTIONS):
    body = json.dumps({
        "model": model,
        "state": state or [f"Customer complaint about {item['product_type']} (batch: {item['batch'] or 'not given'}):", item["text"]],
        "questions": questions,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or attempt == 5:
                raise RuntimeError(f"{item['id']}: HTTP {e.code} {e.read()[:300]!r}") from None
            wait = float(e.headers.get("Retry-After") or min(30, 2 ** attempt + random.random()))
            time.sleep(wait)


def choice_conf(ans):
    return ans.get("confidence") if ans.get("confidence") is not None else max(ans.get("probabilities", {0: 0}).values())


def route(text, answers):
    """Return (route, reasons). Rules first; the model can only add urgency, never remove it."""
    reasons = hard_rules(text)
    if answers["adverse_event"].get("noul", 0) >= 0.5:
        reasons.append("model: adverse event")
    if answers["subcategory"].get("choice") in URGENT_SUBCATS:
        reasons.append("model: urgent subcategory " + answers["subcategory"]["choice"])
    if reasons:
        return "URGENT_QA", reasons
    if choice_conf(answers["primary_category"]) < CONF_MIN or answers["sufficient_info"].get("noul", 0) < 0.5:
        return "HUMAN_REVIEW", ["low confidence or insufficient info"]
    return "QA_REVIEW", []


def summarize(items, responses):
    n = len(items)
    rows, cost = [], 0.0
    for it, resp in zip(items, responses):
        a = resp["answers"]
        cost += resp.get("usage", {}).get("cost") or 0.0
        r, why = route(it["text"], a)
        rows.append((it, a, r, why))
    acc = lambda f: sum(f(it, a) for it, a, _, _ in rows) / n
    print(f"model={MODEL} items={n}")
    print(f"primary category accuracy: {acc(lambda it, a: a['primary_category']['choice'] == it['category']):.1%}")
    print(f"subcategory accuracy:      {acc(lambda it, a: a['subcategory']['choice'] == it['subcategory']):.1%}")
    print(f"sufficient_info accuracy:  {acc(lambda it, a: (a['sufficient_info']['noul'] >= 0.5) == it['sufficient_info']):.1%}")
    print("tag (threshold 0.5)  precision  recall")
    metrics = {}
    for tag in TAG_QUESTIONS:
        tp = sum(a["tag_" + tag]["noul"] >= 0.5 and tag in it["tags"] for it, a, _, _ in rows)
        fp = sum(a["tag_" + tag]["noul"] >= 0.5 and tag not in it["tags"] for it, a, _, _ in rows)
        fn = sum(a["tag_" + tag]["noul"] < 0.5 and tag in it["tags"] for it, a, _, _ in rows)
        p, rc = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        metrics[tag] = (p, rc)
        print(f"  {tag:<18} {p:>9.0%} {rc:>7.0%}")
    urgent = [x for x in rows if x[0]["urgent"]]
    flagged = [x for x in rows if x[2] == "URGENT_QA"]
    hit = sum(x[2] == "URGENT_QA" for x in urgent)
    print(f"urgent recall: {hit}/{len(urgent)}  urgent precision: {sum(x[0]['urgent'] for x in flagged)}/{len(flagged)}")
    print(f"routes: " + ", ".join(f"{k}={sum(x[2] == k for x in rows)}" for k in ("URGENT_QA", "HUMAN_REVIEW", "QA_REVIEW")))
    for it, a, r, why in rows:
        if it["urgent"] and r != "URGENT_QA":
            print(f"  MISSED URGENT {it['id']}: {it['text']}")
    print(f"total cost ${cost:.6f}  per 1,000 complaints ${cost / n * 1000:.4f}")
    return metrics, hit, len(urgent)


def selftest():
    assert "conflicting strength/dose" in hard_rules("Box says 10 mg, bottle says 20 mg.")
    assert "conflicting strength/dose" in hard_rules("take 5 ml twice daily but label says 15 ml")
    assert not hard_rules("The 100 ml bottle is only about two thirds full.")
    assert hard_rules("I found a small black particle inside.") == ["contamination word"]
    assert "adverse event word" in hard_rules("My son vomited after the dose.")
    assert not hard_rules("Hospital pharmacy: tablets were broken.")
    assert "seal/tamper" in hard_rules("The seal under the cap was already broken.")
    assert not hard_rules("One blister cavity was empty but sealed.")

    def fake(cat, sub, conf=0.95, ae=0.1, info=0.9, tags=()):
        a = {"primary_category": {"type": "choice", "choice": cat, "confidence": conf},
             "subcategory": {"type": "choice", "choice": sub, "probabilities": {sub: conf, "other": 1 - conf}},
             "sufficient_info": {"type": "noul", "noul": info}, "adverse_event": {"type": "noul", "noul": ae},
             "severity": {"type": "score", "score": 1.0}}
        for t in TAG_QUESTIONS:
            a["tag_" + t] = {"type": "noul", "noul": 0.9 if t in tags else 0.1}
        return {"answers": a, "usage": {"cost": 0.00002}}

    assert route("tablets broken", fake("physical", "broken_tablets")["answers"])[0] == "QA_REVIEW"
    assert route("tablets broken", fake("physical", "broken_tablets", conf=0.6)["answers"])[0] == "HUMAN_REVIEW"
    assert route("pills odd", fake("other", "other", info=0.2)["answers"])[0] == "HUMAN_REVIEW"
    assert route("felt unwell", fake("other", "other", ae=0.8)["answers"])[0] == "URGENT_QA"
    # model says harmless and confident, but the hard rule still escalates
    assert route("a hair in the tablet", fake("physical", "broken_tablets")["answers"])[0] == "URGENT_QA"

    items = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "complaints.json"), encoding="utf-8"))
    assert len(items) == 40 and len({i["id"] for i in items}) == 40
    for i in items:
        assert i["category"] in CATEGORIES and i["subcategory"] in SUBCATEGORIES, i["id"]
        assert set(i["tags"]) <= set(TAG_QUESTIONS), i["id"]
    perfect = [fake(i["category"], i["subcategory"], info=0.9 if i["sufficient_info"] else 0.1, tags=i["tags"]) for i in items]
    metrics, _, _ = summarize(items, perfect)
    assert all(m == (1.0, 1.0) for m in metrics.values())
    print("selftest OK")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY (an OpenRouter key, sk-or-...) or run --selftest.")
    path = sys.argv[1] if len(sys.argv) > 1 else "complaints.json"
    items = json.load(open(path, encoding="utf-8"))
    with ThreadPoolExecutor(WORKERS) as pool:
        responses = list(pool.map(lambda it: call_jev(it, key), items))
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump([{"complaint_id": it["id"], **r} for it, r in zip(items, responses)], f, indent=1)
    summarize(items, responses)


if __name__ == "__main__":
    main()
