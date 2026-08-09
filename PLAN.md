# ragdx — Retrieval Failure Diagnostician

> **Working name:** `ragdx`. Rename before first public commit.
>
> **This file is the spec.** Read it fully before writing code. Work milestone by
> milestone. Do not skip ahead. Do not add features not listed here — see
> **Non-Goals**.

---

## 1. Mission

Existing RAG evaluation tools give you a **score**. They tell you context
precision is 0.61. They do not tell you *why* it is 0.61 or what to change.

`ragdx` gives you a **diagnosis**. For every query where retrieval failed, it
identifies the root cause, clusters causes across the whole query set, and
reports the specific config change that would recover the most failures.

Example of the target output:

```
127 queries evaluated · 41 retrieval failures

  ROOT CAUSE                      FAILURES   RECOVERABLE BY
  ─────────────────────────────────────────────────────────────────────
  Rank cutoff (gold below k)         18      raising k to 12, or reranker
  Vocabulary mismatch                11      hybrid retrieval (BM25 + dense)
  Chunk boundary split                7      512→768 tokens, 128 overlap
  Embedding blind spot                3      domain-tuned embedder
  Metadata filter over-exclusion      2      relax `department` filter
  Unclassified (low confidence)       0

  Top single fix: enable hybrid retrieval → est. recovery 11/41 (27%)
```

---

## 2. The core idea: differential diagnosis by ablation

**This is the most important section in this document. The design lives or dies
here.**

Do **not** ask an LLM "why did retrieval fail?" LLM judges are biased and
unreliable, and a diagnostician that is confidently wrong is worse than no
diagnostician. Instead, diagnose the way a doctor rules out conditions: **run
counterfactual retrievals and see which one would have succeeded.**

For each failed query, re-run retrieval under a battery of ablations. The
ablation that recovers the gold chunk *names the failure mode*:

| Ablation | If gold chunk is recovered → | Recommended fix |
|---|---|---|
| Same retriever, `k = 100` | **Rank cutoff** — it was ranked, just too low | Add reranker, or raise k |
| Lexical (BM25) only | **Vocabulary mismatch** — dense missed the wording | Hybrid retrieval |
| Dense only (when lexical was used) | **Paraphrase gap** | Hybrid retrieval |
| Filters removed | **Metadata / filter failure** | Fix filter logic |
| Alternate chunking (larger / overlapping) | **Chunk boundary split** | Re-chunk with overlap |
| Nothing recovers it | **Embedding blind spot** | Domain-tuned embedder |
| Gold *was* in top-k but answer is wrong | **Generation failure**, not retrieval | Prompt / grounding work |

Almost every classification above is **deterministic** — it is arithmetic over
ranks and set membership, not an opinion. That is the entire trust argument for
this tool, and it is the thing that differentiates it. Preserve it.

An LLM judge is used in **exactly two** places, and nowhere else:

1. Generating synthetic golden questions from corpus chunks (Milestone 3).
2. Faithfulness scoring on the generation plane, only when the gold chunk *was*
   retrieved (Milestone 6).

Every LLM-derived value carries a confidence and can **abstain**. When the
classifier cannot determine a cause, it reports `unclassified`. Never guess.

---

## 3. Non-Goals (for Phases 1 and 2)

Do not build any of these. If tempted, stop and re-read this list.

- No web UI, no dashboard server, no database. Report is a static HTML file.
- No production traffic sampling, no monitoring, no alerting. (Phase 3.)
- No inline / blocking evaluation endpoint. Evaluation is always offline here.
- No support beyond **two** framework adapters (LangChain, LlamaIndex) plus the
  generic trace-file adapter. Do not add more.
- No agentic or multi-hop retrieval support. Single-turn, single-retrieval only.
- No fine-tuning, no embedding training.
- No generation-plane metrics beyond faithfulness and answer relevance.
- No auth, no multi-tenancy, no telemetry.

---

## 4. Tech constraints

- Python 3.11+, `uv` for dependency management.
- `pydantic` v2 for all schemas. `typer` for CLI. `jinja2` for the report.
- `rank-bm25` for lexical ablation. `numpy` for vector math.
- Adapters live behind optional extras: `ragdx[langchain]`, `ragdx[llamaindex]`.
  The core package must import and run with neither installed.
- **The full test suite must run offline** — no network, no API keys. Use the
  fixture corpus and a stub embedder/judge (Milestone 2).
- Determinism is a hard requirement: fixed seeds everywhere, on-disk cache for
  embeddings and judge calls keyed by content hash. Two runs on the same input
  produce byte-identical JSON output.
- `ruff` + `mypy --strict` clean. `pytest` with coverage on the classifier
  module at 90%+.

---

## 5. Repo layout

```
ragdx/
├── PLAN.md                     ← this file
├── README.md
├── pyproject.toml
├── src/ragdx/
│   ├── schema.py               # Trace, Chunk, Golden, Diagnosis (pydantic)
│   ├── adapters/
│   │   ├── base.py             # Retriever protocol
│   │   ├── trace_file.py       # generic JSONL trace ingest
│   │   ├── langchain.py        # optional extra
│   │   └── llamaindex.py       # optional extra
│   ├── goldens/
│   │   ├── synthesize.py       # LLM-generated goldens from corpus
│   │   ├── importer.py         # load human-labeled goldens
│   │   └── store.py            # versioned golden set on disk
│   ├── ablations/
│   │   ├── base.py             # Ablation protocol
│   │   ├── rank_cutoff.py
│   │   ├── lexical.py
│   │   ├── filters.py
│   │   ├── chunking.py
│   │   └── registry.py         # ordered battery
│   ├── diagnose/
│   │   ├── classifier.py       # ← the core. ablation results → cause
│   │   ├── cluster.py
│   │   └── recommend.py        # cause clusters → ranked fixes
│   ├── judge/
│   │   ├── base.py             # LLM judge protocol + stub impl
│   │   └── faithfulness.py
│   ├── report/
│   │   ├── render.py
│   │   └── template.html.j2
│   ├── ci/
│   │   ├── baseline.py
│   │   ├── gate.py
│   │   └── junit.py
│   ├── cache.py
│   └── cli.py
├── tests/
│   ├── fixtures/corpus/        # planted-failure corpus (Milestone 2)
│   ├── fixtures/expected/
│   └── test_*.py
└── .github/workflows/
```

---

## 6. Data model

Define these in `schema.py` **first**, before any other code. Everything else
depends on them.

```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    char_start: int  # offset in source doc — REQUIRED for boundary detection
    char_end: int
    metadata: dict[str, Any] = {}


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int  # 0-indexed


class Golden(BaseModel):
    golden_id: str
    query: str
    gold_doc_id: str
    gold_char_start: int  # evidence span in the SOURCE DOC, not the chunk
    gold_char_end: int
    expected_answer: str | None = None
    origin: Literal["synthetic", "human"]
    synth_confidence: float | None = None


class Trace(BaseModel):
    trace_id: str
    query: str
    retrieved: list[RetrievedChunk]
    answer: str | None = None
    config_snapshot: dict[str, Any]  # k, retriever type, filters, chunk params


class AblationResult(BaseModel):
    ablation_name: str
    recovered: bool
    recovered_at_rank: int | None


class Diagnosis(BaseModel):
    golden_id: str
    outcome: Literal["hit", "retrieval_failure", "generation_failure"]
    cause: FailureCause | None  # enum; None when outcome == "hit"
    confidence: float
    ablation_results: list[AblationResult]
    evidence: str  # human-readable one-liner explaining the call
```

**Critical detail:** gold evidence is stored as a **character span in the source
document**, never as a chunk ID. Chunk IDs change the moment you re-chunk, which
makes the chunking ablation impossible. Spans survive re-chunking. Get this
right at the start; retrofitting it later means rewriting the whole ablation
layer.

---

# PHASE 1 — Developer / pre-production diagnostic

**Deliverable:** `ragdx run --config ragdx.yaml` produces `report.html` +
`report.json` containing clustered root causes and ranked fixes.

---

### Milestone 1 — Skeleton and schemas

- `pyproject.toml`, `uv` lockfile, ruff + mypy + pytest configured.
- `schema.py` complete, with round-trip serialization tests.
- `cli.py` with `run`, `goldens`, `ci` subcommands stubbed (exit 1, "not
  implemented").
- Retriever protocol in `adapters/base.py`:
  `retrieve(query: str, k: int, filters: dict | None) -> list[RetrievedChunk]`.

**Acceptance:** `uv run ragdx --help` works; `pytest` green; `mypy --strict`
clean.

---

### Milestone 2 — Fixture corpus with planted failures

**Do this before the classifier.** You cannot validate a diagnostician without
inputs whose correct diagnosis is known in advance.

Build a small synthetic corpus (~20 docs) with **deliberately planted** failures,
one per failure mode:

- A doc with a table whose header and data rows land in different chunks under
  the default chunker → *chunk boundary*.
- A doc using domain jargon with a query using the plain-English synonym →
  *vocabulary mismatch*.
- A doc whose gold chunk reliably ranks 8–15 → *rank cutoff*.
- A doc tagged with metadata that the default filter excludes → *filter failure*.
- Several docs where retrieval works fine → *true negatives, guard against a
  classifier that cries wolf*.

Ship a `StubEmbedder` (deterministic hash-based vectors) and `StubJudge` (returns
canned responses) so all of this runs offline.

Write `tests/fixtures/expected/diagnoses.json` — the ground-truth diagnosis for
each fixture query.

**Acceptance:** fixture corpus loads, stub retriever runs, expected-diagnoses
file is committed. This file is the yardstick for every later milestone.

---

### Milestone 3 — Golden set generation

Two paths in:

1. **Synthetic** (`goldens/synthesize.py`): sample a chunk, ask the LLM to write
   a question answerable *only* from that chunk, then **verify** — re-ask the LLM
   to answer the question given only that chunk, and given a random other chunk.
   Keep the golden only if it succeeds on the first and fails on the second.
   This verification step is not optional; without it roughly a third of
   synthetic goldens are answerable from anywhere in the corpus and your failure
   rate becomes noise.
2. **Human-labeled** (`goldens/importer.py`): CSV/JSONL with query + evidence
   span.

Golden sets are versioned on disk (`goldens/v1.jsonl` + a manifest with corpus
hash, generator model, and timestamp). Warn loudly if the corpus hash has changed
since the golden set was built.

**Acceptance:** `ragdx goldens build --corpus ./docs --n 50` produces a
versioned, verified golden set. Rejection rate is logged.

---

### Milestone 4 — Ablation engine

Implement each ablation behind the `Ablation` protocol. Each declares its cost
so the runner can order cheap-and-deterministic before expensive.

Run order matters — **first match wins**, so order from most specific to least:

1. `filters_removed` (cheapest, most specific)
2. `rank_cutoff` (single retrieval at k=100)
3. `lexical_only` / `dense_only`
4. `alternate_chunking` (most expensive — requires re-indexing)

Short-circuit: once an ablation recovers the gold chunk, stop and record. Cache
re-indexing aggressively; the chunking ablation is the slow one.

**Acceptance:** each ablation has unit tests against the fixture corpus. Running
the full battery on 50 fixture queries completes in under 30s with the stub
embedder.

---

### Milestone 5 — Classifier (the core)

`diagnose/classifier.py` maps ablation results → `Diagnosis`.

Rules:

- Gold chunk in top-k → `hit`. No ablations run. (Fast path — most queries.)
- Gold chunk not in top-k → run battery, classify by first recovering ablation.
- No ablation recovers → check similarity distribution. If gold chunk similarity
  is far below retrieved distractors, call `embedding_blind_spot` with moderate
  confidence. Otherwise `unclassified`.
- Confidence: 1.0 for deterministic recoveries; lower for distribution-based
  calls; `unclassified` when below threshold.
- Every `Diagnosis` must populate `evidence` with a plain-English one-liner
  (`"gold chunk recovered at rank 11 with k=100; current k=5"`). Users will not
  trust a bare label.

**Acceptance — this is the gate for the whole project:** running the classifier
against the fixture corpus reproduces `tests/fixtures/expected/diagnoses.json`
with ≥95% agreement, **and zero false positives on the true-negative fixtures.**
If you cannot hit this, do not proceed to reporting; fix the classifier.

---

### Milestone 6 — Generation plane (narrow)

Only for queries where the gold chunk **was** retrieved but the answer is wrong:
run a faithfulness judge (is the answer grounded in the retrieved context?) and
label `generation_failure`. This cleanly separates "retriever's fault" from
"generator's fault" — one of the most useful things the report says.

Keep it minimal. Do not build a general eval metric suite.

**Acceptance:** generation failures in the fixture set are correctly separated
from retrieval failures.

---

### Milestone 7 — Clustering, recommendations, report

- `cluster.py`: group diagnoses by cause; within `vocabulary_mismatch`, sub-group
  by the offending term where detectable.
- `recommend.py`: for each cluster emit the fix and an **estimated recovery
  count** (how many failures that single change would resolve). Rank fixes by
  recovery count ÷ implementation cost. This is the payoff of the whole tool.
- `report/`: single self-contained HTML file (inline CSS, no CDN). Summary table
  at top, then per-cause sections with up to 5 concrete failing examples each,
  showing query / expected evidence / what was actually retrieved. Also emit
  `report.json` — Phase 2 consumes it.

**Acceptance:** `ragdx run --config ragdx.yaml` on the fixture corpus produces a
report a stranger could act on without reading the source.

---

### Milestone 8 — Real adapters + README

- LangChain and LlamaIndex adapters, each ~50 lines wrapping a retriever into the
  protocol. Integration tests marked `@pytest.mark.network`, excluded by default.
- Generic `trace_file` adapter (JSONL of `Trace`) so any stack can feed the tool
  without an adapter at all. **This is the real answer to "works with any RAG
  platform"** — do not write more adapters.
- README: install, 5-minute quickstart, a real screenshot of the report, and an
  explicit **Limitations** section (single-turn only, synthetic goldens are
  imperfect, judge confidence caveats). Honest limitations sections build more
  trust than feature lists.

**Phase 1 is done when** someone can `pip install`, point it at their own corpus
and retriever, and get an actionable report in under 10 minutes.

---

# PHASE 2 — CI regression gate

**Deliverable:** the same engine running in CI, failing a build when retrieval
quality regresses.

Do not start Phase 2 until Phase 1 is genuinely complete and used at least once
on a real (non-fixture) corpus.

---

### Milestone 9 — Baselines

- `ragdx baseline save` snapshots current metrics + per-cause counts to
  `.ragdx/baseline.json`, pinned to a golden-set version and corpus hash.
- Baselines are committed to the repo. They are reviewable artifacts — a diff
  showing recall dropping 4 points is the point.
- Refuse to compare across mismatched golden-set versions. Fail loudly with
  instructions rather than silently comparing apples to oranges.

---

### Milestone 10 — The gate

`ragdx ci --baseline .ragdx/baseline.json --config ragdx.yaml`

- Exit 0 = pass, 1 = regression, 2 = error. Never exit 0 on error.
- Configurable thresholds: absolute floor (`recall@k >= 0.80`) and relative drop
  (`no more than 2% below baseline`). Support both; teams want both.
- **Handle noise.** LLM-judged metrics fluctuate between runs. Retrieval metrics
  are deterministic and should be gated strictly; judged metrics need a tolerance
  band and repeat-on-borderline. A flaky gate gets disabled within a week and
  then the whole tool is dead weight — treat flakiness as a P0 bug.
- Emit JUnit XML so CI systems render it natively.

---

### Milestone 11 — GitHub Action + PR comment

- Composite action in `.github/actions/ragdx/`.
- On PR: run gate, post a comment with the delta table — which causes got worse,
  which improved, and the example queries that flipped from hit to miss. The
  flipped-query list is what makes the comment worth reading.
- Cache embeddings between runs keyed by corpus hash; without this, CI cost kills
  adoption.
- Ship a working example workflow in the README.

---

### Milestone 12 — Calibration report

The credibility milestone, and the best interview story in the project.

- `ragdx calibrate --labeled ./human_labels.jsonl` compares classifier output
  against human-assigned causes and emits a confusion matrix plus per-cause
  precision/recall.
- Publish these numbers in the README. Stating "chunk-boundary detection: 0.91
  precision, 0.84 recall, n=120" is worth more than any feature claim, and
  nobody else in this space publishes it.
- If a cause scores badly, either fix it or mark it experimental. Do not ship a
  confident label you cannot defend.

---

## 7. Working agreement for the agent

- Implement **one milestone per PR/commit batch.** Stop at each acceptance
  criterion and report status before continuing.
- If an acceptance criterion cannot be met, **stop and say so.** Do not weaken
  the criterion to make it pass. The Milestone 5 gate especially is not
  negotiable.
- Do not add dependencies not listed in §4 without flagging it first.
- Do not add features from the Non-Goals list, even if they seem easy.
- Prefer deterministic logic over an LLM call every single time there is a
  choice. If you find yourself adding a judge call outside the two sanctioned
  places, stop and flag it.
- Write the test before the implementation for anything in `diagnose/`.

---

## 8. Deviations from spec (log)

Recorded here rather than made silently. See §7.

| # | Deviation | Reason |
|---|---|---|
| 1 | Added `pyyaml` to core dependencies (not listed in §4) | §5 / Milestone 1 mandate `ragdx run --config ragdx.yaml`; YAML parsing has no stdlib equivalent. |
| 2 | `mypy` has no `python_version` pin | numpy 2.5's bundled stubs use 3.12+ `type` statements, which fail to parse when mypy targets 3.11. mypy targets the running interpreter instead; the CI matrix covers 3.11–3.13. |
| 3 | Modules not in the §5 layout: `corpus.py`, `chunking.py`, `embedding.py`, `index.py`, `matching.py`, `spans.py`, `text.py` | §5 names the modules that carry features; these are the primitives underneath them. `matching.py` matters most: one definition of "did retrieval satisfy this golden", shared by the runner, every ablation and the classifier, so they cannot drift apart. |
| 4 | Fixture rank-cutoff ablation depth is 20, not the §2 illustration of 100 | The depth is a config value defaulting to 100. The fixture index is 213 chunks; at k=100 the ablation degenerates into "return half the corpus", which is not a fix anyone can ship. 20 is 4× the fixture's production `k`. |
| 5 | A retrieval "satisfies" a golden only when one chunk covers ≥75% of the evidence span | Plain overlap would score a boundary-split span as a hit, hiding the most actionable failure mode in the tool. See `matching.py`. |
| 6 | Two candidate `vocabulary_mismatch` fixtures were dropped rather than labelled | They landed outside the rerank depth *and* outside `k` on the lexical plane, so no ablation recovers them and no label is defensible. Ambiguous cases do not belong in a yardstick. See `tests/fixtures/README.md`. |
| 7 | No LLM client is bundled; judges are resolved from a `module:attribute` string (`judge/loader.py`) | §4 lists the dependencies and none is a model SDK, but §2 sanctions judge calls in two places. Users supply their own judge; `stub` is the offline default. |
| 8 | Added `goldens/base.py` (not in the §5 layout) | `Rejection` / `GoldenBatch` are shared by both `synthesize.py` and `importer.py`; putting them in either would make the other import it sideways. |
| 9 | Added `config.py`, `runner.py`, `plugins.py` (not in the §5 layout) | §5 names no home for `ragdx.yaml` parsing or for the assembly that turns a config into a report. Keeping them out of `cli.py` is what makes the whole pipeline testable without invoking the CLI. |
| 10 | `report.json` carries no timestamp | §4 requires two runs on the same input to be byte-identical, and Phase 2 diffs this file against a committed baseline. A timestamp would make every diff non-empty. The HTML footer carries the generation time instead. |
| 11 | Recall counts generation failures as retrieval successes | For those queries the gold chunk *was* retrieved. Counting them as retrieval misses would re-merge the two planes this milestone exists to separate. |
| 12 | Causes sharing a fix are merged into one recommendation | `vocabulary_mismatch` and `paraphrase_gap` are both fixed by turning on hybrid retrieval. Listing them separately would understate that single change, and §1's worked example reports one combined figure. |
| 13 | `metadata_filter` can be reached without the `filters_removed` ablation | When retrieval is replayed from a trace the ablation cannot re-run, but the gold document's metadata still settles it: if nothing covering the evidence span passes the filter, the retriever was never permitted to return it. Deterministic, so it is reported — at confidence 0.7, because the ranking is unproven. |
| 14 | Added `Replayed` protocol to `adapters/base.py` | Ablations must be able to tell a live index from a recording. Without it, `rank_cutoff` against a 5-deep trace would report "not recovered" and the classifier would rule out a cause it never tested. |
