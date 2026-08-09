# The planted-failure fixture

You cannot validate a diagnostician on inputs whose correct diagnosis you do not
already know. This directory is that set of inputs.

## What is here

| Path | What it is |
|---|---|
| `corpus/` | 21 documents of a fictional logistics knowledge base, with YAML front matter carrying `department` and `status` metadata |
| `goldens.yaml` | 51 queries, each with quoted evidence and the **planted** ground-truth failure |
| `expected/diagnoses.json` | The ground truth in `Diagnosis` terms — the yardstick for Milestone 5 |

## The production configuration under diagnosis

Defined in `tests/support.py`, because the fixture only means anything relative
to a specific configuration:

| Setting | Value |
|---|---|
| retriever | dense (`StubEmbedder`, cosine) |
| `k` | 5 |
| filters | `status == "current"` |
| chunking | 240 characters, 60 overlap → 213 chunks |
| rank-cutoff ablation depth | 20 (4 × production `k`) |
| alternate chunking | 960 characters, 480 overlap → 67 chunks |
| gold coverage threshold | 0.75 of the evidence span, in a single chunk |

The rank-cutoff depth is 20 rather than the library default of 100 because the
fixture index is only 213 chunks; at k=100 the ablation degenerates into
"return half the corpus", which is not a fix anybody can ship.

## What was planted, and how

| Cause | n | Mechanism |
|---|---|---|
| `hit` | 36 | Ordinary queries that work. These are the false-positive guard — a classifier that cries wolf fails here first. |
| `rank_cutoff` | 3 | Gold chunk ranks 6–15 with `k=5`: ranked, just too low. |
| `vocabulary_mismatch` | 2 | A jargon term (`rollover`, `deadhead`) occurring in exactly one chunk of `glossary.md`, surrounded in the query by words that are common everywhere else. Mean-pooled embeddings dilute the one rare token among the chunk's other thirty (dense ranks 79 and 55); BM25's IDF weighting makes it dominant (lexical ranks 3 and 1). |
| `chunk_boundary` | 2 | A table whose header row and the answering data row are 400+ characters apart, so no 240-character chunk can cover 75% of the evidence span. Under 960-character chunking a single chunk covers it entirely. |
| `metadata_filter` | 4 | The answer lives in `refund-processing.md` or `invoice-disputes.md`, both tagged `status: archived`, which the production filter excludes outright. |
| `embedding_blind_spot` | 2 | Query and gold chunk share no token and no synonym bridge, so dense ranks them near the bottom of the index (190 and 152 of 213) and BM25 scores them zero. Nothing recovers them. |
| `generation_failure` | 2 | Retrieval succeeds at rank 0; the recorded answer contradicts the retrieved evidence. |

## How the ground truth was established

This is the part that decides whether the Milestone 5 gate means anything.

1. **Authored from intent.** Each failure was deliberately built into the corpus
   before anything was measured. A table was reordered so the answering row
   would fall outside its header's chunk; two documents were tagged `archived`;
   a glossary was written to hold single-occurrence jargon.
2. **Confirmed by primitive measurement.** Every plant was then checked against
   facts that are arithmetic, not judgement: the rank of the gold chunk under
   each retrieval condition, the best span coverage achievable under each
   chunker, and whether the production filter excludes the document. Where a
   plant did not take, the *corpus or the query was changed* until it did.
3. **Never derived from the classifier.** No label in `expected/diagnoses.json`
   was produced by running `diagnose/`. The classifier's job is to reconstruct
   these labels from ablation results alone, including getting the precedence
   order right, and it can still fail at that.

`tests/test_fixture_corpus.py` re-runs step 2 on every test run, so a corpus
edit that quietly destroys a planted failure fails the build instead of silently
weakening the yardstick.

## Deliberately excluded

Two candidate cases were dropped rather than labelled. Queries naming the
identifiers `NW-4417` and `HS 8471.30` land at dense rank 23 and 41 and lexical
rank 6 and 10 — outside the rerank depth but also outside `k` on the lexical
plane, so no ablation recovers them and no single label is defensible. Ambiguous
cases do not belong in a yardstick. `unclassified` behaviour is tested directly
against the classifier instead.

## Offline by construction

`StubEmbedder` derives vectors by hashing tokens, so it needs no model and is
byte-identical across platforms and runs. `StubJudge` answers from a canned
table and **abstains** on any prompt it does not recognise, so a call site
wired up wrongly fails its test rather than passing on an invented answer.
