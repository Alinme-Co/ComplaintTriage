# ComplaintTriage — Pharmaceutical Complaint Classification and QA Routing

ComplaintTriage is an Alinme Python tutorial for classifying pharmaceutical product quality complaints from text and photos. It uses OpenRouter decision models, including TypeSafe Jev, to return structured predictions and confidence scores. Fixed escalation rules and model signals route each case to a quality assurance (QA) reviewer.

![ComplaintTriage banner with a tablet blister, carton, and QA checklist](docs/complaint-triage-banner.jpg)

[Quick start](#quick-start) · [Routing rules](#how-complaint-routing-works) · [Local API](#local-api) · [Benchmark](#synthetic-complaint-benchmark) · [License](#license)

## Quick start

Use Python 3.10 or later. The tutorial uses the Python standard library.

```bash
git clone https://github.com/Alinme-Co/ComplaintTriage.git
cd ComplaintTriage
python classify.py --selftest
```

The self-test runs offline without credentials. To classify sample complaints or launch the demo, set `OPENROUTER_API_KEY` to your [OpenRouter API key](https://openrouter.ai/keys). In Bash, use `export OPENROUTER_API_KEY=your-key`. In PowerShell, use `$env:OPENROUTER_API_KEY = 'your-key'`.

```bash
python server.py                    # open http://127.0.0.1:8765
python classify.py complaints.json  # run the 40-case benchmark
python image_test.py images/broken_tablets.jpg images/degraded_blister.jpg
```

Set `JEV_MODEL` to a supported model ID such as `liquid/d1` to change the benchmark runner's model. Use synthetic complaints for live requests because complaint text and photos are sent to OpenRouter.

![Local demo showing a damaged blister routed to Human review with category probabilities and tags](docs/demo.png)

## How complaint routing works

The model answers ten typed questions per complaint. It selects a primary category and subcategory, estimates the probability of five defect tags, flags possible adverse events and insufficient information, and scores severity. Primary categories are physical, packaging, labeling, contamination, quantity, and other.

The [routing function](classify.py) evaluates these conditions in order:

| Route | Condition |
| --- | --- |
| `URGENT_QA` | A text rule fires, the model's adverse-event probability is at least 0.5, or the model selects an urgent subcategory. |
| `HUMAN_REVIEW` | No urgent signal, and primary-category confidence is below 0.8 or the probability of sufficient information is below 0.5. |
| `QA_REVIEW` | All other complaints. |

Text rules flag patterns such as conflicting strengths, foreign particles, adverse reactions, and broken seals. Urgent subcategories include wrong strength, foreign particles, microbial contamination, wrong pack, and broken seals. A model prediction cannot override an urgent text rule. QA staff confirm the classification and retain responsibility for batch safety, recalls, and regulatory reporting.

## Local API

The Python server listens on `127.0.0.1:8765` by default and keeps the OpenRouter key on the server.

| Endpoint | Contract |
| --- | --- |
| `GET /api/models` | Returns the supported model IDs and the default model. |
| `POST /api/classify` | Accepts JSON with `text`, `image` (a base64 PNG or JPEG data URL), or both. Optional fields are `product_type` and `model`. |

Example request body:

```json
{
  "text": "Two tablets in the blister arrived cracked.",
  "product_type": "medicine"
}
```

A successful response includes `model`, `fallback`, `answers`, `route`, `reasons`, `cost`, and `ms`. The web demo defaults to Perplexity Decider v1.3 and retries a failed model call with TypeSafe Jev 1.14 unless Jev was already selected. See [the server implementation](server.py) for the request and response details.

## Synthetic complaint benchmark

On November 14, 2026, all six tested models reached 12/12 urgent-case recall and 12/12 urgent-case precision when combined with the routing rules. The [dataset](complaints.json) contains 40 author-labeled synthetic complaints, including 12 urgent cases. Each model used the same questions and rules.

| Model | Primary category accuracy | Subcategory accuracy | Sufficient-info accuracy | Human-review routes | Recorded cost per 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `perplexity/pplx-decider-v1.3:free` | 100% | 95% | 97.5% | 8 | $0 |
| `typesafe/jev-1.14` | 100% | 92.5% | 100% | 7 | $0.052 |
| `liquid/d1` | 97.5% | 90% | 97.5% | 9 | $0.014 |
| `openai/gpt-6-luna-decisions` | 95% | 95% | 97.5% | 5 | $0.191 |
| `google/gemini-decision-1` | 90% | 95% | 97.5% | 8 | $0.108 |
| `cloudflare/clef-flash` | 87.5% | 85% | 95% | 22 | $0.037 |

The routing rules alone and Jev alone each caught 11 of the 12 urgent cases, with different misses. Model runs took 6–8 seconds per 40-complaint batch with eight workers. The cost and timing figures describe the recorded run.

<details>
<summary>Multi-label tag and photo results</summary>

Tag precision and recall at a 0.5 probability threshold, reported as percentages:

| Model | Physical | Packaging | Labeling | Contamination | Quantity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Perplexity Decider v1.3 | 76 / 93 | 56 / 91 | 100 / 100 | 78 / 100 | 86 / 100 |
| Jev 1.14 | 68 / 93 | 53 / 91 | 100 / 100 | 35 / 100 | 67 / 100 |
| Liquid D1 | 70 / 100 | 58 / 91 | 100 / 88 | 41 / 100 | 75 / 100 |
| GPT-6 Luna Decisions | 65 / 93 | 53 / 91 | 100 / 100 | 54 / 100 | 67 / 100 |
| Gemini Decision 1 | 68 / 93 | 59 / 91 | 100 / 100 | 44 / 100 | 67 / 100 |
| Clef Flash | 78 / 100 | 79 / 100 | 100 / 88 | 100 / 100 | 100 / 100 |

Several tags over-fired at this threshold. For example, the model could tag a strength mismatch as a quantity problem.

On two photo-only examples, all six models classified a crushed tablet next to a blister as a physical defect (confidence 0.59–0.90). They disagreed about the primary category for discolored tablets in a damaged blister. Confidence was at most 0.63, below the threshold for routine QA review. Photo requests cost 4–8 times as much as text requests in these tests.

</details>

See the [raw model responses](benchmarks/) and [decision model research notes](report.md) for evidence and methodology. This small synthetic test illustrates the workflow. Production quality and safety performance remain unvalidated.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`classify.py`](classify.py) | Decision questions, routing rules, OpenRouter client, benchmark runner, and offline self-test. |
| [`server.py`](server.py) and [`index.html`](index.html) | Local HTTP API and browser demo. |
| [`complaints.json`](complaints.json) and [`benchmarks/`](benchmarks/) | Labeled synthetic cases and recorded model responses. |
| [`image_test.py`](image_test.py) and [`images/`](images/) | Photo comparison script and two example photos used for testing. |
| [`report.md`](report.md) | Decision model API, research sources, and related use cases. |

## License

Alinme publishes this proprietary source code for viewing and educational reference. See the [license and usage terms](LICENSE.md) for permissions.
