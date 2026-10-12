"""Offline evaluation harness for FAQ retrieval (Phase 1f of the FAQ retrieval overhaul).

Run from the repo root (PowerShell shown; use `export` on macOS/Linux):

    python tests/retrieval/eval_faq_retrieval.py validate        # check the case file against the guideline
    python tests/retrieval/eval_faq_retrieval.py splits          # give new cases a dev/test split (never moves old ones)

    $env:VOYAGE_API_KEY = "pa-..."
    python tests/retrieval/eval_faq_retrieval.py run             # embed (disk-cached), score, save raw, report on dev
    python tests/retrieval/eval_faq_retrieval.py run --reuse --split test
                                                                 # re-analyse the saved raw output: no API calls
    python tests/retrieval/eval_faq_retrieval.py probe --n 200   # Voyage latency and error rate, production client as is
    python tests/retrieval/eval_faq_retrieval.py run --embedder fake
                                                                 # plumbing check only; hashed trigrams, not a quality signal

How it works: every unique (embedder, model, dimensions, input type, text) is embedded once and cached on disk, and
cosine similarity is computed here in Python. After the first run, an experiment (a new threshold, a margin rule, a
different split) costs nothing. The raw top-5 index rows and the best score per FAQ are saved for every query to
artifacts/faq_retrieval_raw.json, and `run --reuse` rebuilds every number in the report from that file alone.

Discipline: tune on `dev`, report on `test`. The default `--threshold tune-dev` picks the threshold on dev; `--split test`
then reports it out of sample. Scores are the same cosine similarity pgvector reports as 1 - (embedding <=> query).

The embedding cache lives outside the repository (see --cache-dir). `artifacts/` is where the safety eval also writes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Arabic output on Windows consoles


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


ROOT = _find_root()
if (ROOT / "src").is_dir() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CASES_PATH = Path(__file__).with_name("faq_eval_cases.json")
RAW_PATH = ROOT / "artifacts" / "faq_retrieval_raw.json"
LATENCY_PATH = ROOT / "artifacts" / "faq_embed_latency.json"
CALIBRATION_PATH = ROOT / "src" / "customer_service" / "data" / "faq_calibration.json"
DEFAULT_CACHE_DIR = Path(os.environ.get("FAQ_EVAL_CACHE_DIR") or Path.home() / ".cache" / "customer-service-faq-eval")

NO_ANSWER_ID = "__no_answer__"  # reserved sink pseudo-FAQ (Phase 3d); never a real answer
SPLITS = ("dev", "test")
SPLIT_SEED = "faq-eval-v1"
POSITIVE_SLICES = ("en_natural", "en_messy", "msa", "egyptian", "arabizi", "code_switched")
NEGATIVE_LABELS = ("near_miss", "out_of_domain", "degenerate")
LABELS = ("positive", *NEGATIVE_LABELS)
STYLES = ("en", "msa", "egyptian", "arabizi", "mixed", "none")
REVIEW_STATUSES = ("draft", "approved", "edited", "rejected")
REVIEWED = ("approved", "edited")
REQUIRED_KEYS = ("id", "text", "slice", "style", "label", "faq_id", "topic", "near_faq", "review")
ARMS = ("answers", "title_answer_variants", "variants_sink")
SMALL_N = 30  # below this a per-slice figure is diagnostic only

# Plan section 4 (test split). Verdicts are printed as a point estimate and, separately, as "verified".
TARGET_RECALL = 0.90
TARGET_RECALL_PER_SLICE = 0.80
TARGET_WRONG_FAQ = 0.01
TARGET_NEAR_MISS_FA = 0.05

PERSONAL_DATA = re.compile(r"@|\d{6,}")


# ----------------------------------------------------------------------------------------------------------------
# Cases: loading, validation, splits
# ----------------------------------------------------------------------------------------------------------------


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_cases(cases: list[dict], path: Path = CASES_PATH) -> None:
    """One case per line keeps review diffs readable."""
    body = ",\n".join(" " + json.dumps(case, ensure_ascii=False) for case in cases)
    path.write_text("[\n" + body + "\n]\n", encoding="utf-8")


def corpus_faq_ids() -> list[str]:
    from customer_service.retrieval.corpus import load_corpus

    return sorted({document.id for document in load_corpus()})


def stratum(case: dict) -> tuple[str, str]:
    """The cells that must stay 50/50 between dev and test."""
    if case["label"] == "positive":
        return case["slice"], case["faq_id"]
    if case["label"] == "near_miss":
        return case["slice"], case["style"]
    return case["slice"], ""


def validate_cases(cases: list[dict], faq_ids: Iterable[str], *, require_splits: bool = False) -> list[str]:
    known = set(faq_ids)
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_texts: dict[str, str] = {}
    split_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for index, case in enumerate(cases):
        label = f"case #{index} ({case.get('id', '?')})"
        missing = [key for key in REQUIRED_KEYS if key not in case]
        if missing:
            errors.append(f"{label}: missing keys {missing}")
            continue
        if case["id"] in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(case["id"])
        text = case["text"]
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{label}: empty text")
            continue
        folded = text.strip().casefold()
        if folded in seen_texts:
            errors.append(f"{label}: same text as {seen_texts[folded]}")
        seen_texts[folded] = case["id"]
        if PERSONAL_DATA.search(text):
            errors.append(f"{label}: looks like it contains personal data (an @ or a 6+ digit number)")
        if case["label"] not in LABELS:
            errors.append(f"{label}: unknown label {case['label']!r}")
            continue
        if case["style"] not in STYLES:
            errors.append(f"{label}: unknown style {case['style']!r}")
        if case["review"].get("status") not in REVIEW_STATUSES:
            errors.append(f"{label}: unknown review status {case['review'].get('status')!r}")
        if case["label"] == "positive":
            if case["slice"] not in POSITIVE_SLICES:
                errors.append(f"{label}: positive case in slice {case['slice']!r}")
            if case["faq_id"] not in known:
                errors.append(f"{label}: faq_id {case['faq_id']!r} is not in the corpus")
            if case["topic"] != case["faq_id"]:
                errors.append(f"{label}: positive topic must equal faq_id")
        else:
            if case["slice"] != case["label"]:
                errors.append(f"{label}: negative case must use slice {case['label']!r}")
            if case["faq_id"] is not None:
                errors.append(f"{label}: negative case must have faq_id null")
        if case["near_faq"] is not None and case["near_faq"] not in known:
            errors.append(f"{label}: near_faq {case['near_faq']!r} is not in the corpus")
        split = case.get("split")
        if split is None:
            if require_splits:
                errors.append(f"{label}: no split (run the `splits` command)")
        elif split not in SPLITS:
            errors.append(f"{label}: unknown split {split!r}")
        else:
            split_counts[stratum(case)][split] += 1
    for key, counts in sorted(split_counts.items()):
        if abs(counts["dev"] - counts["test"]) > 1:
            errors.append(f"split imbalance in {key}: dev={counts['dev']} test={counts['test']}")
    return errors


def assign_splits(cases: list[dict]) -> int:
    """Fill in missing `split` values. Existing assignments never move. Returns how many were assigned."""
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for case in cases:
        if case.get("split") in SPLITS:
            counts[stratum(case)][case["split"]] += 1
    assigned = 0
    for case in sorted((c for c in cases if c.get("split") not in SPLITS), key=lambda c: c["id"]):
        cell = counts[stratum(case)]
        if cell["dev"] != cell["test"]:
            side = "dev" if cell["dev"] < cell["test"] else "test"
        else:  # a tie is broken by a stable hash so the order of ids carries no signal
            digest = hashlib.sha256(f"{SPLIT_SEED}:{case['id']}".encode("utf-8")).hexdigest()
            side = "dev" if int(digest, 16) % 2 == 0 else "test"
        case["split"] = side
        cell[side] += 1
        assigned += 1
    return assigned


def active_cases(cases: list[dict], *, reviewed_only: bool = False) -> list[dict]:
    keep = REVIEWED if reviewed_only else ("draft", *REVIEWED)
    return [case for case in cases if case["review"]["status"] in keep]


# ----------------------------------------------------------------------------------------------------------------
# Embeddings: provider, disk cache, fake
# ----------------------------------------------------------------------------------------------------------------


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], *, model: str, dims: int, input_type: str) -> list[list[float]]: ...


RATE_LIMIT_WAIT_SECONDS = 22.0  # one request per 22 s stays under the unpaid tier's 3 requests per minute


def is_rate_limit(error_name: str) -> bool:
    return "ratelimit" in error_name.casefold()


class VoyageEmbedder:
    """Sync Voyage client with backoff and optional pacing. Production keeps its own client.

    Voyage's unpaid tier allows 3 requests per minute. A retry is itself a request, so a short backoff keeps
    failing: rate-limit errors wait 22 s or more, and `rpm` spaces requests so the limit is not hit at all.
    """

    name = "voyage"

    def __init__(
        self,
        api_key: str,
        *,
        retries: int = 5,
        timeout: float = 60.0,
        rpm: float = 0.0,
        client=None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        if client is None:
            import voyageai

            client = voyageai.Client(api_key=api_key, max_retries=0, timeout=timeout)
        self._client = client
        self._retries = retries
        self._min_interval = 60.0 / rpm + 1.0 if rpm > 0 else 0.0  # +1 s margin: the limiter is not exact
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._hinted = False

    def _pace(self) -> None:
        if self._min_interval and self._last_request is not None:
            wait = self._last_request + self._min_interval - self._clock()
            if wait > 0:
                print(f"  pacing for the rate limit: waiting {wait:.0f}s", file=sys.stderr)
                self._sleep(wait)
        self._last_request = self._clock()

    def embed(self, texts: list[str], *, model: str, dims: int, input_type: str) -> list[list[float]]:
        for attempt in range(self._retries + 1):
            self._pace()
            try:
                result = self._client.embed(texts, model=model, input_type=input_type, output_dimension=dims)
                return [list(vector) for vector in result.embeddings]
            except Exception as exc:  # noqa: BLE001 - any failure is retried, then re-raised
                if attempt == self._retries:
                    raise
                limited = is_rate_limit(type(exc).__name__)
                if limited:
                    wait = min(65.0, RATE_LIMIT_WAIT_SECONDS * (attempt + 1)) + random.random()
                else:
                    wait = min(60.0, 2.0**attempt) + random.random()
                print(f"  embed failed ({type(exc).__name__}: {str(exc)[:100]}); retry {attempt + 1} in {wait:.0f}s", file=sys.stderr)
                if limited and not self._hinted:
                    self._hinted = True
                    print("  hint: on Voyage's unpaid tier (3 requests/minute) pass --rpm 3, or add a payment method in the Voyage dashboard", file=sys.stderr)
                self._sleep(wait)
        raise AssertionError("unreachable")


class FakeEmbedder:
    """Hashed character trigrams. Deterministic and offline. It checks plumbing and says nothing about quality."""

    name = "fake"

    def embed(self, texts: list[str], *, model: str, dims: int, input_type: str) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * dims
            padded = f" {text.casefold()} "
            for start in range(len(padded) - 2):
                digest = hashlib.blake2b(padded[start : start + 3].encode("utf-8"), digest_size=8).digest()
                vector[int.from_bytes(digest, "big") % dims] += 1.0
            vectors.append(vector)
        return vectors


class CachedEmbedder:
    """Wraps an embedder with batching and an append-only JSONL cache keyed by text hash.

    Input type is part of the key because Voyage embeds a document and a query differently, so the plan's
    "(model, text)" key would let a document vector answer a query lookup.
    """

    def __init__(self, inner: Embedder, directory: Path, *, batch_size: int = 128, delay: float = 0.0) -> None:
        self.name = inner.name
        self._inner = inner
        self._directory = directory
        self._batch_size = max(1, batch_size)
        self._delay = delay
        self._stores: dict[Path, dict[str, list[float]]] = {}

    def _path(self, model: str, dims: int, input_type: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{self.name}__{model}__{dims}__{input_type}")
        return self._directory / f"{safe}.jsonl"

    def _store(self, path: Path) -> dict[str, list[float]]:
        if path not in self._stores:
            store: dict[str, list[float]] = {}
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                        store[item["k"]] = item["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # a half-written last line from an interrupted run
            self._stores[path] = store
        return self._stores[path]

    def embed(self, texts: list[str], *, model: str, dims: int, input_type: str) -> list[list[float]]:
        path = self._path(model, dims, input_type)
        store = self._store(path)
        missing = list(dict.fromkeys(text for text in texts if sha(text) not in store))
        if missing:
            self._directory.mkdir(parents=True, exist_ok=True)
            requests = math.ceil(len(missing) / self._batch_size)
            print(f"  embedding {len(missing)} new {input_type} text(s) in {requests} request(s); {len(texts) - len(missing)} already cached", file=sys.stderr)
        for start in range(0, len(missing), self._batch_size):
            batch = missing[start : start + self._batch_size]
            vectors = self._inner.embed(batch, model=model, dims=dims, input_type=input_type)
            if len(vectors) != len(batch) or any(len(vector) != dims for vector in vectors):
                raise RuntimeError("Embedder returned an unexpected number of vectors or dimensions")
            with path.open("a", encoding="utf-8") as handle:
                for text, vector in zip(batch, vectors, strict=True):
                    store[sha(text)] = vector
                    handle.write(json.dumps({"k": sha(text), "v": vector}) + "\n")
            if self._delay and start + self._batch_size < len(missing):
                time.sleep(self._delay)
        return [store[sha(text)] for text in texts]


def unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


# ----------------------------------------------------------------------------------------------------------------
# Index and scoring (today's design: answers embedded, top-1)
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexRow:
    row: str
    faq_id: str
    kind: str
    text: str


def build_index(arm: str) -> list[IndexRow]:
    if arm not in ARMS:
        raise SystemExit(f"Unknown arm {arm!r}; available: {', '.join(ARMS)}")
    from customer_service.retrieval.corpus import load_corpus, load_sink, load_variants

    documents = load_corpus()
    if arm == "answers":
        return [IndexRow(f"{d.id}:{d.language}", d.id, "answer", d.searchable_text) for d in documents]
    titles = {(d.id, d.language): d.title for d in documents}
    answers = {(d.id, d.language): d.searchable_text for d in documents}
    variants = load_variants()
    rows: list[IndexRow] = []
    if arm == "title_answer_variants":
        for number, variant in enumerate(variants):
            language = "ar" if variant.dialect in {"msa", "egyptian", "arabizi"} else "en"
            key = (variant.faq_id, language)
            rows.append(IndexRow(f"variant:{number}:{variant.faq_id}", variant.faq_id, "variant", f"{titles[key]}\n{answers[key]}\nQuestion: {variant.text}"))
        return rows
    for number, variant in enumerate([*variants, *load_sink()]):
        rows.append(IndexRow(f"variant:{number}:{variant.faq_id}", variant.faq_id, "variant", variant.text))
    return rows


def index_sha(rows: Sequence[IndexRow]) -> str:
    return sha("\n".join(f"{row.row}\t{row.text}" for row in rows))


def score_one(query: Sequence[float], index: Sequence[IndexRow], index_vectors: Sequence[Sequence[float]]) -> dict:
    scored = sorted(
        ((sum(a * b for a, b in zip(query, vector)), row) for row, vector in zip(index, index_vectors)),
        key=lambda item: (-item[0], item[1].row),
    )
    faq_best: dict[str, float] = {}
    for score, row in scored:  # sorted high to low, so the first row seen for a FAQ is its best
        faq_best.setdefault(row.faq_id, round(score, 6))
    return {
        "top": [{"row": row.row, "faq_id": row.faq_id, "score": round(score, 6)} for score, row in scored[:5]],
        "faq_best": faq_best,
    }


def load_calibration_fixtures(path: Path = CALIBRATION_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_raw(
    cases: list[dict],
    fixtures: dict,
    index: list[IndexRow],
    embedder: Embedder,
    *,
    arm: str,
    model: str,
    dims: int,
) -> dict:
    queries = [case["text"] for case in cases]
    fixture_queries = [item["query"] for item in (*fixtures["positive"], *fixtures["negative"])]
    query_vectors = embedder.embed(list(dict.fromkeys([*queries, *fixture_queries])), model=model, dims=dims, input_type="query")
    by_text = {text: unit(vector) for text, vector in zip(dict.fromkeys([*queries, *fixture_queries]), query_vectors, strict=True)}
    index_vectors = [unit(v) for v in embedder.embed([row.text for row in index], model=model, dims=dims, input_type="document")]
    return {
        "meta": {
            "arm": arm,
            "embedder": embedder.name,
            "model": model,
            "dims": dims,
            "index": [{"row": r.row, "faq_id": r.faq_id, "kind": r.kind} for r in index],
            "index_sha": index_sha(index),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "calibration": {
            "positive": [
                {"query": item["query"], "expected": item["faq_id"], "text_sha": sha(item["query"]), **score_one(by_text[item["query"]], index, index_vectors)}
                for item in fixtures["positive"]
            ],
            "negative": [
                {"query": item["query"], "expected": None, "text_sha": sha(item["query"]), **score_one(by_text[item["query"]], index, index_vectors)}
                for item in fixtures["negative"]
            ],
        },
        "rows": [
            {"id": case["id"], "text_sha": sha(case["text"]), **score_one(by_text[case["text"]], index, index_vectors)}
            for case in cases
        ],
    }


# ----------------------------------------------------------------------------------------------------------------
# Decision rule, classification, summaries
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    threshold: float
    margin: float | None = None  # required gap between the best and second-best FAQ (arm A4); None = off


@dataclass(frozen=True)
class Scored:
    case: dict
    ranked: tuple[tuple[str, float], ...]  # (faq_id, best score), best first; the sink competes like any FAQ


@dataclass(frozen=True)
class Result:
    scored: Scored
    predicted: str | None
    outcome: str  # positives: correct | wrong_faq | false_reject. negatives: false_accept | correct_reject

    @property
    def case(self) -> dict:
        return self.scored.case

    @property
    def top_score(self) -> float:
        return self.scored.ranked[0][1]


class StaleRawError(RuntimeError):
    pass


def rank_faqs(faq_best: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted(faq_best.items(), key=lambda item: (-item[1], item[0])))


def to_scored(cases: Sequence[dict], raw: dict) -> list[Scored]:
    by_id = {row["id"]: row for row in raw["rows"]}
    missing = [case["id"] for case in cases if case["id"] not in by_id]
    if missing:
        raise StaleRawError(f"{len(missing)} cases have no raw scores (e.g. {missing[:3]}); re-run without --reuse")
    scored = []
    for case in cases:
        row = by_id[case["id"]]
        if row["text_sha"] != sha(case["text"]):
            raise StaleRawError(f"text of case {case['id']} changed since the raw output was saved; re-run without --reuse")
        scored.append(Scored(case, rank_faqs(row["faq_best"])))
    return scored


def decide(ranked: Sequence[tuple[str, float]], rule: Rule) -> str | None:
    """Accept the best FAQ, or return None for "no match"."""
    top_id, top_score = ranked[0]
    if top_id == NO_ANSWER_ID or top_score < rule.threshold:
        return None
    if rule.margin is not None and len(ranked) > 1 and top_score - ranked[1][1] < rule.margin:
        return None
    return top_id


def classify(scored: Scored, rule: Rule) -> Result:
    predicted = decide(scored.ranked, rule)
    case = scored.case
    if case["label"] == "positive":
        outcome = "false_reject" if predicted is None else "correct" if predicted == case["faq_id"] else "wrong_faq"
    else:
        outcome = "false_accept" if predicted is not None else "correct_reject"
    return Result(scored, predicted, outcome)


def evaluate(scored: Sequence[Scored], rule: Rule) -> list[Result]:
    return [classify(item, rule) for item in scored]


def summary(results: Sequence[Result]) -> dict:
    positives = [r for r in results if r.case["label"] == "positive"]
    out = {
        "n_pos": len(positives),
        "correct": sum(r.outcome == "correct" for r in positives),
        "wrong_faq": sum(r.outcome == "wrong_faq" for r in positives),
        "false_reject": sum(r.outcome == "false_reject" for r in positives),
    }
    for label in NEGATIVE_LABELS:
        group = [r for r in results if r.case["label"] == label]
        out[f"{label}_n"] = len(group)
        out[f"{label}_fa"] = sum(r.outcome == "false_accept" for r in group)
    return out


def rank_of_expected(scored: Scored) -> int:
    for position, (faq_id, _) in enumerate(scored.ranked, 1):
        if faq_id == scored.case["faq_id"]:
            return position
    return len(scored.ranked) + 1


def top1_and_mrr(scored: Sequence[Scored]) -> tuple[int, float]:
    positives = [s for s in scored if s.case["label"] == "positive"]
    if not positives:
        return 0, float("nan")
    ranks = [rank_of_expected(s) for s in positives]
    return sum(rank == 1 for rank in ranks), sum(1 / rank for rank in ranks) / len(ranks)


# ----------------------------------------------------------------------------------------------------------------
# Thresholds: today's calibration, tuning, sweeps
# ----------------------------------------------------------------------------------------------------------------


def today_threshold(calibration: dict) -> tuple[float | None, str]:
    """Reproduce retrieval/calibration.py offline from the saved fixture scores.

    Returns (threshold, note). The threshold is None where today's calibrate() would raise.
    """
    positives: list[float] = []
    for item in calibration["positive"]:
        ranked = rank_faqs(item["faq_best"])
        if ranked[0][0] != item["expected"]:
            return None, f"calibrate() would raise: positive did not retrieve {item['expected']}: {item['query']!r} (got {ranked[0][0]})"
        positives.append(ranked[0][1])
    negatives = [rank_faqs(item["faq_best"])[0][1] for item in calibration["negative"]]
    low, high = min(positives), max(negatives)
    if low <= high:
        return None, f"calibrate() would raise: scores overlap (lowest positive {low:.4f} <= highest negative {high:.4f})"
    return round((low + high) / 2, 6), f"lowest positive {low:.4f}, highest negative {high:.4f}"


def candidate_thresholds(scores: Iterable[float]) -> list[float]:
    """Cut points between adjacent observed scores, nudged to the midpoint of each gap as calibration does."""
    ordered = sorted(set(scores))
    if not ordered:
        return []
    candidates = [ordered[0]]  # accepts everything observed
    for low, high in zip(ordered, ordered[1:]):
        mid = round((low + high) / 2, 6)
        candidates.append(mid if low < mid <= high else (low + high) / 2)
    candidates.append(ordered[-1] + 1e-6)  # accepts nothing observed
    return candidates


@dataclass(frozen=True)
class Tuned:
    threshold: float
    recall: float  # recall at that threshold on the tuning split
    feasible: bool  # False when recall is under the floor: a content or model problem, not a threshold problem


def tune_threshold(
    scored: Sequence[Scored],
    *,
    margin: float | None,
    max_near_miss_fa: float,
    max_wrong_faq: float,
    min_recall: float = 0.80,
) -> Tuned:
    """Lowest threshold meeting the false-accept limits, which is also the one with the best recall.

    Accepting nothing always meets the limits, so the real test is recall at that point: under `min_recall` the
    result is flagged infeasible (plan Phase 6) instead of crashing as today's calibration does on any overlap.
    """
    positives = [s for s in scored if s.case["label"] == "positive"]
    near = [s for s in scored if s.case["label"] == "near_miss"]
    if not positives or not near:
        raise ValueError("tuning needs positives and near-miss cases in the split")
    for threshold in candidate_thresholds(s.ranked[0][1] for s in scored):  # ascending; accept-nothing comes last
        counts = summary(evaluate(scored, Rule(threshold, margin)))
        if counts["near_miss_fa"] / counts["near_miss_n"] <= max_near_miss_fa and counts["wrong_faq"] / counts["n_pos"] <= max_wrong_faq:
            recall = counts["correct"] / counts["n_pos"]
            return Tuned(threshold, recall, recall >= min_recall)
    raise AssertionError("unreachable: the last candidate accepts nothing")


# ----------------------------------------------------------------------------------------------------------------
# Statistics: exact bounds, bootstrap, percentiles
# ----------------------------------------------------------------------------------------------------------------


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact (Clopper-Pearson) upper confidence bound on a rate. With k=0 it is about 3/n."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if _binom_cdf(k, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact lower confidence bound on a rate."""
    if n <= 0:
        return float("nan")
    if k <= 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if 1 - _binom_cdf(k - 1, n, mid) < alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def bootstrap_ci(k: int, n: int, *, n_boot: int = 2000, seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for k successes in n cases. Seeded, so reruns print the same numbers."""
    if n <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    flags = [1] * k + [0] * (n - k)
    means = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def fmt_rate(k: int, n: int) -> str:
    """k/n with a 95% interval. A bootstrap collapses to [0, 0] at zero events, so edges use the exact bound."""
    if n == 0:
        return "n/a"
    body = f"{k}/{n} = {k / n:.1%}"
    if k == 0:
        return f"{body}  (95% upper bound <= {cp_upper(k, n):.1%})"
    if k == n:
        return f"{body}  (95% lower bound >= {cp_lower(k, n):.1%})"
    low, high = bootstrap_ci(k, n)
    return f"{body}  [95% CI {low:.1%} to {high:.1%}]"


# ----------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------


def verdict(point_ok: bool, verified: bool) -> str:
    return "VERIFIED" if verified else "met, not verifiable at this n" if point_ok else "NOT met"


def section_headline(results: Sequence[Result], scored: Sequence[Scored]) -> list[str]:
    counts = summary(results)
    top1, mrr = top1_and_mrr(scored)
    n_pos = counts["n_pos"]
    lines = ["", f"Headline (positives n={n_pos}; rates use 95% intervals, exact bounds at 0 or all)"]
    lines.append(f"  recall (accepted and correct)  {fmt_rate(counts['correct'], n_pos)}")
    lines.append(f"  wrong FAQ accepted             {fmt_rate(counts['wrong_faq'], n_pos)}")
    lines.append(f"  false reject                   {fmt_rate(counts['false_reject'], n_pos)}")
    for label in NEGATIVE_LABELS:
        lines.append(f"  {label.replace('_', '-') + ' false accept':<30} {fmt_rate(counts[f'{label}_fa'], counts[f'{label}_n'])}")
    if n_pos:
        lines.append(f"  threshold-free: top-1 {fmt_rate(top1, n_pos)}   MRR {mrr:.3f}")
    return lines


def section_targets(results: Sequence[Result], split: str) -> list[str]:
    counts = summary(results)
    n_pos, near_n = counts["n_pos"], counts["near_miss_n"]
    lines = ["", "Targets (plan section 4, defined on the test split)"]
    if split != "test":
        lines.append(f"  This run reports on '{split}', so these verdicts are diagnostic only.")
    if n_pos:
        recall = counts["correct"] / n_pos
        lines.append(
            f"  recall >= {TARGET_RECALL:.0%}            {recall:.1%}  "
            + verdict(recall >= TARGET_RECALL, cp_lower(counts["correct"], n_pos) >= TARGET_RECALL)
        )
        wrong = counts["wrong_faq"] / n_pos
        lines.append(
            f"  wrong FAQ <= {TARGET_WRONG_FAQ:.0%}            {wrong:.1%}  "
            + verdict(wrong <= TARGET_WRONG_FAQ, cp_upper(counts["wrong_faq"], n_pos) <= TARGET_WRONG_FAQ)
        )
    if near_n:
        near = counts["near_miss_fa"] / near_n
        lines.append(
            f"  near-miss FA <= {TARGET_NEAR_MISS_FA:.0%}         {near:.1%}  "
            + verdict(near <= TARGET_NEAR_MISS_FA, cp_upper(counts["near_miss_fa"], near_n) <= TARGET_NEAR_MISS_FA)
        )
    lines.append(f"  per-slice recall >= {TARGET_RECALL_PER_SLICE:.0%}: see the slice table (n is small; diagnostic only)")
    lines.append("  'VERIFIED' means the one-sided 95% bound meets the target, not just the point estimate.")
    return lines


def section_slices(results: Sequence[Result], scored: Sequence[Scored]) -> list[str]:
    lines = ["", "Positives by slice", f"  {'slice':<14} {'n':>3} {'recall':>7} {'correct':>8} {'wrong':>6} {'reject':>7} {'top-1':>6} {'MRR':>6}"]
    for name in POSITIVE_SLICES:
        group = [r for r in results if r.case["label"] == "positive" and r.case["slice"] == name]
        if not group:
            continue
        counts = summary(group)
        top1, mrr = top1_and_mrr([r.scored for r in group])
        flag = f"  diagnostic only (n<{SMALL_N})" if len(group) < SMALL_N else ""
        lines.append(
            f"  {name:<14} {len(group):>3} {counts['correct'] / len(group):>7.0%} {counts['correct']:>8} "
            f"{counts['wrong_faq']:>6} {counts['false_reject']:>7} {top1 / len(group):>6.0%} {mrr:>6.2f}{flag}"
        )
    by_faq: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        if r.case["label"] == "positive":
            by_faq[r.case["faq_id"]].append(r)
    if by_faq:
        lines.append("  by FAQ: " + ", ".join(f"{faq} {summary(group)['correct']}/{len(group)}" for faq, group in sorted(by_faq.items())))
    near = [r for r in results if r.case["label"] == "near_miss"]
    if near:
        lines.extend(["", "Near-misses accepted, by language style and by topic"])
        for style in STYLES:
            group = [r for r in near if r.case["style"] == style]
            if group:
                lines.append(f"  {style:<9} {sum(r.outcome == 'false_accept' for r in group)}/{len(group)} accepted")
        leaked = Counter(r.case["topic"] for r in near if r.outcome == "false_accept")
        lines.append("  topics with an accepted near-miss: " + (", ".join(f"{t} x{c}" for t, c in leaked.most_common()) or "none"))
    return lines


def section_scores(scored: Sequence[Scored]) -> list[str]:
    lines = ["", "Top-1 score by class (the number the threshold sees)", f"  {'class':<14} {'n':>3} {'min':>6} {'p25':>6} {'median':>7} {'p75':>6} {'max':>6}"]
    for label in LABELS:
        values = [s.ranked[0][1] for s in scored if s.case["label"] == label]
        if values:
            lines.append(
                f"  {label:<14} {len(values):>3} {min(values):>6.3f} {percentile(values, 25):>6.3f} "
                f"{statistics.median(values):>7.3f} {percentile(values, 75):>6.3f} {max(values):>6.3f}"
            )
    return lines


def section_worst(results: Sequence[Result], count: int) -> list[str]:
    lines: list[str] = []
    negatives = sorted((r for r in results if r.case["label"] != "positive"), key=lambda r: -r.top_score)[:count]
    positives = sorted((r for r in results if r.case["label"] == "positive"), key=lambda r: (r.outcome == "correct", r.top_score))[:count]
    if negatives:
        lines.extend(["", "Highest-scoring negatives (the false accepts to fear first)"])
        for r in negatives:
            mark = "ACCEPTED" if r.outcome == "false_accept" else "rejected"
            lines.append(f"  {r.top_score:.3f} {mark:<9} {r.scored.ranked[0][0]:<15} [{r.case['label']}/{r.case['topic']}] {r.case['id']}: {r.case['text'][:60]}")
    if positives:
        lines.extend(["", "Weakest positives (wrong FAQ and false rejects first)"])
        for r in positives:
            lines.append(f"  {r.top_score:.3f} {r.outcome:<12} got {r.scored.ranked[0][0]:<15} want {r.case['faq_id']:<15} {r.case['id']}: {r.case['text'][:50]}")
    return lines


def section_sweep(scored: Sequence[Scored], rule: Rule, tuned_note: str, *, max_rows: int = 24) -> list[str]:
    candidates = candidate_thresholds(s.ranked[0][1] for s in scored)
    if len(candidates) > max_rows:
        step = (len(candidates) - 1) / (max_rows - 1)
        candidates = [candidates[round(i * step)] for i in range(max_rows)]
    thresholds = sorted(set([*candidates, rule.threshold]))
    lines = [
        "",
        f"Threshold sweep on this split. {tuned_note}",
        f"  {'threshold':>9} {'recall':>7} {'wrong':>6} {'near-miss FA':>13} {'ood FA':>7} {'degenerate FA':>14}",
    ]
    for threshold in thresholds:
        counts = summary(evaluate(scored, Rule(threshold, rule.margin)))
        n_pos = counts["n_pos"] or 1
        cells = [
            f"{counts['near_miss_fa']}/{counts['near_miss_n']}",
            f"{counts['out_of_domain_fa']}/{counts['out_of_domain_n']}",
            f"{counts['degenerate_fa']}/{counts['degenerate_n']}",
        ]
        mark = "   <-- reported" if threshold == rule.threshold else ""
        lines.append(f"  {threshold:>9.4f} {counts['correct'] / n_pos:>7.0%} {counts['wrong_faq'] / n_pos:>6.0%} {cells[0]:>13} {cells[1]:>7} {cells[2]:>14}{mark}")
    return lines


# ----------------------------------------------------------------------------------------------------------------
# Latency probe (production embedding client, as is)
# ----------------------------------------------------------------------------------------------------------------


async def probe_embeddings(provider, texts: Sequence[str], n: int, *, delay: float = 0.0) -> list[dict]:
    """Time n single-query embeds through a production-shaped provider (async embed(texts, input_type=...))."""
    rows = []
    for index in range(n):
        started = time.perf_counter()
        error = None
        try:
            await provider.embed([texts[index % len(texts)]], input_type="query")
        except Exception as exc:  # noqa: BLE001 - the point is to count whatever fails
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
        rows.append({"seconds": time.perf_counter() - started, "error": error})
        if delay:
            await asyncio.sleep(delay)
    return rows


def latency_summary(rows: Sequence[dict]) -> list[str]:
    good = [row["seconds"] for row in rows if row["error"] is None]
    errors = Counter(row["error"].split(":")[0] for row in rows if row["error"])
    lines = [f"Calls: {len(rows)}   ok: {len(good)}   failed: {sum(errors.values())} {dict(errors) or ''}".rstrip()]
    if rows:
        lines.append(f"error rate: {fmt_rate(sum(errors.values()), len(rows))}")
    if good:
        lines.append(
            f"embed latency (s): first {good[0]:.3f}   p50 {percentile(good, 50):.3f}   "
            f"p95 {percentile(good, 95):.3f}   max {max(good):.3f}"
        )
    limited = sum(count for name, count in errors.items() if is_rate_limit(name))
    if limited:
        lines.append(
            f"WARNING: {limited} call(s) were rate limited by Voyage. These numbers describe your account's limit, "
            "not production behaviour. Lift the limit (add a payment method in the Voyage dashboard), then run the probe again."
        )
    return lines


# ----------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------


def _defaults() -> tuple[str, int]:
    try:
        from customer_service.config.settings import Settings

        fields = Settings.model_fields
        return fields["voyage_model"].default, fields["voyage_embedding_dimensions"].default
    except Exception:  # pragma: no cover - the script still works outside the installed package
        return "voyage-3.5-lite", 1024


def _count_table(cases: Sequence[dict]) -> list[str]:
    lines = ["", "Counts (plan target in brackets)"]
    per_faq = {"en_natural": 8, "en_messy": 6, "msa": 8, "egyptian": 8, "arabizi": 8, "code_switched": 4}
    faqs = sorted({c["faq_id"] for c in cases if c["label"] == "positive"})
    for name in POSITIVE_SLICES:
        by_faq = Counter(c["faq_id"] for c in cases if c["slice"] == name)
        lines.append(f"  {name:<14} " + "  ".join(f"{faq}={by_faq[faq]}" for faq in faqs) + f"   [{per_faq[name]} per FAQ]")
    plan = {"near_miss": 40, "out_of_domain": 12, "degenerate": 10}
    for label in NEGATIVE_LABELS:
        lines.append(f"  {label:<14} {sum(c['label'] == label for c in cases)}   [~{plan[label]}]")
    status = Counter(c["review"]["status"] for c in cases)
    lines.append("  review: " + ", ".join(f"{name}={status[name]}" for name in REVIEW_STATUSES))
    split = Counter(c.get("split", "unassigned") for c in cases)
    lines.append("  split: " + ", ".join(f"{name}={count}" for name, count in sorted(split.items())))
    return lines


def cmd_validate(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    errors = validate_cases(cases, corpus_faq_ids(), require_splits=args.require_splits)
    print("\n".join(_count_table(cases)))
    if errors:
        print(f"\n{len(errors)} problems:")
        print("\n".join(f"  - {error}" for error in errors))
        return 1
    print("\nCase file is valid.")
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    path = Path(args.cases)
    cases = load_cases(path)
    assigned = assign_splits(cases)
    if assigned:
        write_cases(cases, path)
    errors = validate_cases(cases, corpus_faq_ids(), require_splits=True)
    print(f"Assigned a split to {assigned} case(s); existing assignments were not touched.")
    print("\n".join(_count_table(cases)))
    if errors:
        print("\n".join(f"  - {error}" for error in errors))
        return 1
    return 0


def _build_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "fake":
        return FakeEmbedder()
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        sys.exit("Set VOYAGE_API_KEY first (keep it out of chat and out of git), or use --reuse / --embedder fake.")
    voyage = VoyageEmbedder(key, retries=args.retries, rpm=args.rpm)
    return CachedEmbedder(voyage, Path(args.cache_dir), batch_size=args.batch_size, delay=args.delay)


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    errors = validate_cases(cases, corpus_faq_ids(), require_splits=True)
    if errors:
        print("The case file has problems (run `validate`):", *errors[:5], sep="\n  - ")
        return 1
    active = active_cases(cases, reviewed_only=args.reviewed_only)
    if not active:
        print("No cases to evaluate (with --reviewed-only, none have been reviewed yet).")
        return 1
    raw_path = Path(args.raw) if args.raw else (RAW_PATH if args.arm == "answers" else ROOT / "artifacts" / f"faq_retrieval_raw_{args.arm}.json")
    if args.reuse:
        if not raw_path.exists():
            sys.exit(f"{raw_path} does not exist yet; run once without --reuse.")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        index = build_index(args.arm)
        print(f"Embedding {len(active)} queries and {len(index)} index rows with {args.model} ({args.dims} dims)...")
        raw = build_raw(active, load_calibration_fixtures(), index, _build_embedder(args), arm=args.arm, model=args.model, dims=args.dims)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Raw scores saved to {raw_path}")
    meta = raw["meta"]
    try:
        everything = to_scored(active, raw)
    except StaleRawError as exc:
        print(f"error: {exc}")
        return 1

    lines = [""]
    if meta["embedder"] == "fake":
        lines.append("FAKE EMBEDDER: hashed character trigrams. This checks plumbing only; do not read quality from these numbers.")
    drafts = sum(case["review"]["status"] == "draft" for case in active)
    if drafts:
        lines.append(f"DRAFT SET: {drafts} of {len(active)} cases are not yet reviewed by a native speaker. Treat every number as provisional.")
    try:
        if meta["index_sha"] != index_sha(build_index(meta["arm"])):
            lines.append("WARNING: the FAQ corpus changed since these raw scores were saved. Re-run without --reuse.")
    except SystemExit:
        pass
    lines.append(f"FAQ retrieval eval: arm={meta['arm']} embedder={meta['embedder']} model={meta['model']} dims={meta['dims']} (raw saved {meta['created_utc']})")

    today, today_note = today_threshold(raw["calibration"])
    lines.append("Today's calibration would give: " + (f"FAQ_MIN_SIMILARITY={today} ({today_note})" if today is not None else f"nothing; {today_note}"))

    dev = [s for s in everything if s.case["split"] == "dev"]
    in_split = everything if args.split == "all" else [s for s in everything if s.case["split"] == args.split]
    tuned_on = None
    if args.threshold == "today":
        if today is None:
            print("\n".join(lines))
            print("\nThe baseline with today's calibration cannot be produced on this data. Use --threshold tune-dev or a number.")
            return 2
        rule, how = Rule(today, args.margin), "today's calibration (fixture-based)"
    elif args.threshold == "tune-dev":
        tuned = tune_threshold(
            dev, margin=args.margin, max_near_miss_fa=args.max_near_miss_fa, max_wrong_faq=args.max_wrong_faq, min_recall=args.min_recall
        )
        rule = Rule(tuned.threshold, args.margin)
        tuned_on = "dev"
        how = (
            f"tuned on dev: lowest threshold with near-miss FA <= {args.max_near_miss_fa:.0%} and wrong FAQ <= {args.max_wrong_faq:.0%}"
            f" (dev recall there: {tuned.recall:.0%})"
        )
        if not tuned.feasible:
            how += f"\n  ** INFEASIBLE: dev recall {tuned.recall:.0%} is under the {args.min_recall:.0%} floor. Suspect the content or the model, not the threshold. **"
    else:
        rule, how = Rule(float(args.threshold), args.margin), "fixed by --threshold"
    margin_text = "off" if rule.margin is None else f"{rule.margin}"
    lines.append(f"Rule: accept the best FAQ if score >= {rule.threshold} (margin {margin_text}); {how}")
    in_sample = tuned_on is not None and args.split in (tuned_on, "all")
    counts_text = ", ".join(f"{sum(s.case['label'] == label for s in in_split)} {label}" for label in LABELS)
    lines.append(f"Reporting on split '{args.split}': {len(in_split)} cases ({counts_text})" + ("   IN-SAMPLE: the threshold was tuned on these cases" if in_sample else ""))
    if not in_split:
        print("\n".join(lines))
        print("No cases in this split.")
        return 1

    results = evaluate(in_split, rule)
    lines += section_headline(results, in_split)
    lines += section_targets(results, args.split)
    lines += section_slices(results, in_split)
    lines += section_scores(in_split)
    lines += section_worst(results, args.worst)
    if not args.no_sweep:
        lines += section_sweep(in_split, rule, "In-sample by construction: a best-possible threshold, not an expected one.")
    lines.append("")
    lines.append("Caveats: the cases are drafts until reviewed; per-slice figures are diagnostic; embed latency is measured by `probe`, SQL latency by the pgvector parity check.")
    print("\n".join(lines))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        sys.exit("Set VOYAGE_API_KEY first.")
    from customer_service.config.settings import Settings
    from customer_service.retrieval.embeddings import VoyageEmbeddingProvider

    provider = VoyageEmbeddingProvider(Settings(voyage_api_key=key, voyage_model=args.model, voyage_embedding_dimensions=args.dims))
    texts = [case["text"] for case in active_cases(load_cases(Path(args.cases)))]
    print(f"Timing {args.n} single-query embeds with the production client ({args.model}, {args.dims} dims, no retries)...")
    rows = asyncio.run(probe_embeddings(provider, texts, args.n, delay=args.delay))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": args.model, "dims": args.dims, "rows": rows}, indent=1), encoding="utf-8")
    print("\n".join(latency_summary(rows)))
    print(f"Raw timings saved to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    model, dims = _defaults()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(CASES_PATH), help="the labeled case file")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check the case file against the guideline")
    validate.add_argument("--require-splits", action="store_true")
    validate.set_defaults(func=cmd_validate)

    splits = commands.add_parser("splits", help="assign dev/test to cases that have no split yet")
    splits.set_defaults(func=cmd_splits)

    run = commands.add_parser("run", help="embed, score offline, save raw output, and report")
    run.add_argument("--reuse", action="store_true", help="analyse the saved raw output; make no API calls")
    run.add_argument("--embedder", choices=("voyage", "fake"), default="voyage")
    run.add_argument("--arm", choices=ARMS, default="answers")
    run.add_argument("--model", default=model)
    run.add_argument("--dims", type=int, default=dims)
    run.add_argument("--split", choices=(*SPLITS, "all"), default="dev", help="which split to report on (default dev)")
    run.add_argument("--threshold", default="tune-dev", help="'tune-dev', 'today', or a number (default tune-dev)")
    run.add_argument("--margin", type=float, default=None, help="require this gap to the second-best FAQ (arm A4)")
    run.add_argument("--max-near-miss-fa", type=float, default=TARGET_NEAR_MISS_FA)
    run.add_argument("--max-wrong-faq", type=float, default=TARGET_WRONG_FAQ)
    run.add_argument("--min-recall", type=float, default=0.80, help="recall floor for tune-dev; below it the result is flagged infeasible")
    run.add_argument("--reviewed-only", action="store_true", help="drop cases still marked draft")
    run.add_argument("--worst", type=int, default=8, help="how many worst offenders to list")
    run.add_argument("--no-sweep", action="store_true")
    run.add_argument("--raw", default=None, help="raw score output path (defaults by arm)")
    run.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    run.add_argument("--batch-size", type=int, default=128, help="texts per request; fewer requests help under a per-minute limit")
    run.add_argument("--rpm", type=float, default=0.0, help="requests per minute to stay under; Voyage's unpaid tier is 3")
    run.add_argument("--delay", type=float, default=0.0, help="extra seconds between batches")
    run.add_argument("--retries", type=int, default=5)
    run.set_defaults(func=cmd_run)

    probe = commands.add_parser("probe", help="time single-query embeds with the production client")
    probe.add_argument("--n", type=int, default=200)
    probe.add_argument("--model", default=model)
    probe.add_argument("--dims", type=int, default=dims)
    probe.add_argument("--delay", type=float, default=0.0)
    probe.add_argument("--out", default=str(LATENCY_PATH))
    probe.set_defaults(func=cmd_probe)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "threshold", None) not in (None, "today", "tune-dev"):
        try:
            float(args.threshold)
        except ValueError:
            sys.exit("--threshold must be 'tune-dev', 'today', or a number")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Embeddings already fetched are cached; run the same command again to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
