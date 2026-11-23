# Pharmaceutical Product Complaint Classification with Jev — research notes

Researched 2026-11-14. Items marked **(unconfirmed)** could not be verified from a primary source.

## 1. What Jev is

- Jev is a decision model by TypeSafe, served on OpenRouter. TypeSafe calls it a "System One" model: it returns typed answers with calibrated probabilities and generates no text, rationale or explanation. [1][3]
- Model IDs: `typesafe/jev-1.14` (text + images, adds Box), `typesafe/jev-1.13` (text), alias `~typesafe/jev-latest`. [1][4]
- Endpoint: `POST https://openrouter.ai/api/alpha/decisions`, body `{model, state, questions}`. Response: `{id, model, provider, answers, usage}`; `answers` is keyed by your question IDs. [1][5]
- Price: $0.042 per million input tokens, output free; context 262,144 tokens. Text over ~32 KB or any image switches to a slower, less accurate long-context mode. [1][3]
- Launch date: sources disagree (Sep 15 vs Sep 18, 2026) **(unconfirmed)**.

### Question primitives [1][5][6]

| Primitive | Asks | Request | Answer |
|---|---|---|---|
| `choice` | Which one option? | `instructions`, `criteria` {option: description} | `choice`, `probabilities`, `confidence` |
| `noul` | Is this true? | `instructions`, optional `criteria` {true, false} | `noul` = P(yes) |
| `score` | Where on an ordered scale? | `instructions`, `criteria` [low → high] | `score`, `probabilities`, `confidence`, `legend` |
| `box` | Where in the image? (1.14 only) | `instructions` | `box` {top, bottom, left, right}, `confidence` |

A question can also come back as `refusal`. A `choice` always picks an option, so include `other`.

### Published accuracy (by OpenRouter, not independent)

- Banking77 (3,080 utterances, 77 intents): Jev 1.13 81.0% accuracy vs Claude Opus 5 84.4%; Jev 13x faster at the median and 1/22 the cost. On the 58% of items with confidence ≥ 0.99, Jev was 96.3% accurate; under 0.5 confidence, 29.6%. [2]
- TweetTopic cookbook: 0.8 confidence threshold kept 114/122 correct and sent 28 of 150 items to a person; ~$0.026 per 1,000 items. [7]

## 2. Related use cases found

| Use case | Source | Pattern that maps to complaint triage |
|---|---|---|
| Support ticket triage | TypeSafe docs, awesome-jev [6][8] | `choice` for queue, `score` for severity, `noul` for specific flags; routing in code |
| Classify & tag text at scale | OpenRouter cookbook [7] | one `choice` for the exclusive category + one `noul` per tag; per-tag thresholds tuned on 100–200 labeled items |
| Insurance claims triage | TypeSafe use-case map, awesome-jev [6][8] | independent `noul`s (coverage, missing docs, fraud); middle probability band → human |
| Hierarchical classification | awesome-jev [8] | narrow label when confident, roll up to parent when less, else human review |
| LLM guardrails / medical advice detection | TypeSafe use-case map [6] | thresholds in code decide pass / review / block |
| Scanned vaccination record | TypeSafe use-case map [6] | doubtful readings flagged for a nurse (image input — relevant for complaint photos) |
| Cross-border B2B case triage | GitHub `maraichr/jev-triage` [9] | ~1,500–2,000 input tokens per case, under 1/100 cent |
| Hard rules before the model | awesome-jev (jev-engineering, Cribrix) [8] | regex rules run first; model can only add, not remove, escalation |
| Healthcare intake routing | Ziplyne, Medium guides [10][11] | routing aid only; high threshold; human queue as the default |

**No Jev example for pharmaceutical complaints or pharmacovigilance was found.** The use case in this tutorial is new.

### Pharma complaint triage prior art (other LLMs, not Jev)

Several GitHub prototypes (PharmaQMS, PharmaGuard-AI, PharmaCopilot, Medicine-Complaint-ai) classify product quality complaints with LangGraph + Groq-hosted LLMs. Common ideas: ICH Q9-style Critical / Major / Minor risk, completeness check, duplicate detection, expensive root-cause/CAPA step only for Major/Critical, deterministic regex fallback, human review before saving. Several come from the same hiring assessment, so they are not independent. [12][13]

Useful rule from clinical-trial protocols: a product quality complaint that may be linked to an adverse event runs through both the complaint and the adverse-event procedures. The demo therefore asks an `adverse_event` `noul` and escalates on it. [14] (seen in search summary; exact protocol page **unconfirmed**)

## 3. How the demo maps the tutorial to Jev

`classify.py` sends one request per complaint with 10 questions:

- `primary_category` — `choice` over physical / packaging / labeling / contamination / quantity / other
- `subcategory` — `choice` over 19 subcategories + other
- `tag_<category>` — 5 `noul`s, so one complaint can carry several categories
- `sufficient_info` — `noul` (tutorial Question 4)
- `adverse_event` — `noul`
- `severity` — `score` Minor / Major / Critical

Routing is deterministic code, not model output:

1. Hard rules on the raw text (conflicting mg/ml values, contamination words, adverse-event words, seal/tamper) → `URGENT_QA`.
2. Model adverse event ≥ 0.5, or an urgent subcategory (wrong strength, foreign particle, microbial, wrong pack, broken seal) → `URGENT_QA`. The model can add urgency but never remove a rule hit.
3. Primary confidence < 0.8 or insufficient info → `HUMAN_REVIEW`.
4. Otherwise `QA_REVIEW`. Every complaint still reaches a human; the model never decides on batch safety or recall.

`complaints.json` holds 40 synthetic complaints with reference labels, including multi-defect (002, 038, 040), vague (033–035), non-quality (037) and urgent cases (12 total).

## 4. Results

- `python classify.py --selftest` passes: rule checks, routing checks, dataset validation, and metric code verified with perfect canned answers (100% / 12 of 12 urgent).
### Live run — `typesafe/jev-1.14`, 2026-11-14, 40 complaints

| Measure | Result |
|---|---|
| Primary category accuracy | 100% (40/40) |
| Subcategory accuracy | 92.5% (37/40) |
| Sufficient-info accuracy | 100% |
| Urgent recall / precision (rules + model) | 12/12, 12/12 |
| Urgent caught by hard rules alone | 11/12 |
| Urgent caught by model alone | 11/12, 0 false urgent |
| Routes | 12 URGENT_QA, 7 HUMAN_REVIEW, 21 QA_REVIEW |
| Cost | $0.0021 total, ~$0.052 per 1,000 (~1,240 input tokens each) |

Per-tag `noul` at a fixed 0.5 threshold:

| Tag | Precision | Recall |
|---|---|---|
| physical | 68% | 93% |
| packaging | 53% | 91% |
| labeling | 100% | 100% |
| contamination | 35% | 100% |
| quantity | 67% | 100% |

Observations:

- The single-label `choice` questions are strong; the multi-label `noul` tags over-fire at 0.5. Examples: "contamination" fires (0.6–0.89) on discoloration, odor and label complaints; "quantity" fires (0.9) on "10 mg vs 20 mg"; "packaging" fires on most label complaints.
- Fix per the OpenRouter cookbook: tune a threshold per tag on a labeled sample and reword tags that overlap. 40 items is too few to tune reliably.
- Subcategory misses: 033 vague "looks different" → discoloration; 039 two defects → discoloration (reference: unreadable label); 040 tamper + residue → broken_seal. All three were routed to human review or urgent anyway.
- The rules and the model each missed a different urgent case, which supports running both.
- Low primary-category confidence (< 0.8) appeared on the multi-defect complaints (002, 038, 039, 040) — confidence works as a "needs a human" signal here.

These are 40 synthetic, mostly clear-cut complaints written by the same author as the reference labels. Expect lower numbers on real complaints.

Re-run: `OPENROUTER_API_KEY=sk-or-v1-... python classify.py complaints.json`. Raw answers are saved to `results.json`.

### Model comparison — same 40 text complaints, same questions and rules

| Model | Primary | Subcat | Contam. tag precision | Human-review routes | Cost / 1,000 |
|---|---|---|---|---|---|
| perplexity/pplx-decider-v1.3:free | 100% | 95% | 78% | 8 | $0 |
| typesafe/jev-1.14 | 100% | 92.5% | 35% | 7 | $0.052 |
| liquid/d1 | 97.5% | 90% | 41% | 9 | $0.014 |
| openai/gpt-6-luna-decisions | 95% | 95% | 54% | 5 | $0.191 |
| google/gemini-decision-1 | 90% | 95% | 44% | 8 | $0.108 |
| cloudflare/clef-flash | 87.5% | 85% | 100% | 22 | $0.037 |

All six reached 12/12 urgent recall with the rules. **Chosen for the demo: Perplexity Decider v1.3 (free)**, best overall accuracy and tag precision; Jev 1.14 is the automatic fallback. Free-tier rate limits and data-use terms for the free model were not checked **(unconfirmed)** — check them before using real complaints.

### Photo-only test — 6 decision models, 2 photos, 2026-11-14

OpenRouter lists 41 decision models on the same Decisions API (`/api/v1/models?output_modalities=decisions`); most accept images. `image_test.py` sends each photo with no complaint text, same questions as `classify.py` (minus `adverse_event`) plus a `damage_visible` `noul`.

| Photo | Model | Primary (conf) / subcategory | Tags ≥ 0.5 | Cost |
|---|---|---|---|---|
| broken_tablets.jpg | typesafe/jev-1.14 | physical (0.86) / broken_tablets | physical, quantity | $0.00021 |
| | openai/gpt-6-luna-decisions | physical (0.75) / broken_tablets | physical, quantity | $0.00033 |
| | google/gemini-decision-1 | physical (0.90) / broken_tablets | physical | $0.00021 |
| | liquid/d1 | physical (0.77) / broken_tablets | physical | $0.00006 |
| | cloudflare/clef-flash | physical (0.59) / broken_tablets | physical | $0.00005 |
| | perplexity/pplx-decider-v1.3:free | physical (0.86) / broken_tablets | physical | free |
| degraded_blister.jpg | typesafe/jev-1.14 | contamination (0.41) / unexpected_residue | physical, packaging, contamination | $0.00016 |
| | openai/gpt-6-luna-decisions | contamination (0.30) / damaged_blister | physical, packaging, contamination | $0.00028 |
| | google/gemini-decision-1 | packaging (0.63) / damaged_blister | physical, packaging, contamination | $0.00021 |
| | liquid/d1 | physical (0.26) / damaged_blister | physical, packaging, contamination | $0.00004 |
| | cloudflare/clef-flash | physical (0.13) / damaged_blister | packaging | $0.00010 |
| | perplexity/pplx-decider-v1.3:free | physical (0.24) / damaged_blister | physical, packaging, contamination | free |

- Clear single defect (crushed tablet): all 6 agree on physical / broken tablets.
- Mixed defect (discolored, degraded tablets under damaged, stained foil): models split on the primary category, and every confidence is ≤ 0.63, so the `HUMAN_REVIEW` rule (< 0.8) would catch it. The multi-label tags (physical + packaging + contamination) describe it better than the single choice.
- Photo requests cost 4–8x a text request (image tokens). Severity scores stayed below "Critical", so images should not replace the deterministic text rules.
- Only 2 photos; no conclusion about which model is best. These are web example photos, not real complaints.

## 5. Caveats

- Accuracy figures above are published by OpenRouter itself; no independent benchmark was found.
- TypeSafe's terms for regulated use (GxP validation, data processing, BAA) were not found **(unconfirmed)**. Use synthetic data only until checked.
- awesome-jev says hosted Jev is text-only; OpenRouter's docs say 1.14 accepts PNG/JPEG images. The OpenRouter docs are newer.
- Probabilities vary slightly between runs; set thresholds as bands, tuned on a labeled sample.
- The taxonomy is illustrative, not an industry standard.

## Sources

1. [OpenRouter — Jev documentation](https://openrouter.ai/docs/guides/community/jev)
2. [OpenRouter — Is Jev as accurate as frontier models at classification?](https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/)
3. [OpenRouter — What is Jev?](https://openrouter.ai/blog/insights/what-is-jev/)
4. [OpenRouter — Jev 1.13 model page](https://openrouter.ai/typesafe/jev-1.13)
5. [OpenRouter — Decisions API reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md)
6. [TypeSafe docs — primitives](https://docs.typesafe.ai/primitives) and [use-case map](https://docs.typesafe.ai/concepts/use-case-map)
7. [OpenRouter cookbook — Classify and tag text at scale with Jev](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/jev-classification)
8. [GitHub — awesome-jev-by-typesafe](https://github.com/Anil-matcha/awesome-jev-by-typesafe)
9. [GitHub — maraichr/jev-triage](https://github.com/maraichr/jev-triage)
10. [Ziplyne — AI that doesn't talk: a guide to TypeSafe's Jev](https://ziplyne.agency/blog/ai-that-doesnt-talk-typesafe-jev-guide)
11. [Medium — Jev by TypeSafe](https://medium.com/@visrow/jev-by-typesafe-a-model-you-were-waiting-for-19e8fa8cb5cb)
12. [GitHub — suraj-vaishnav18/pharmaqms-ai](https://github.com/suraj-vaishnav18/pharmaqms-ai), [HarshitaSharmai/PharmaQMS](https://github.com/HarshitaSharmai/PharmaQMS-AI-Powered-Customer-Complaint-Management-System)
13. [GitHub — meetpokal1/PharmaGuard-AI](https://github.com/meetpokal1/PharmaGuard-AI), [prathmeshmdeshmane001/PharmaCopilot](https://github.com/prathmeshmdeshmane001/PharmaCopilot)
14. [ClinicalTrials.gov — protocol NCT04355117 (PQC definition)](https://cdn.clinicaltrials.gov/large-docs/17/NCT04355117/Prot_000.pdf)
