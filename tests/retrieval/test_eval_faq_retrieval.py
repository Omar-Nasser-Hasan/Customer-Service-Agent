"""Offline tests for the FAQ retrieval eval harness and its case file. No network, no Voyage, no Postgres."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

_PATH = Path(__file__).with_name("eval_faq_retrieval.py")
_SPEC = importlib.util.spec_from_file_location("eval_faq_retrieval", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
harness = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_faq_retrieval"] = harness  # dataclasses resolve their annotations through sys.modules
_SPEC.loader.exec_module(harness)

PER_FAQ_PER_SLICE = {"en_natural": 8, "en_messy": 6, "msa": 8, "egyptian": 8, "arabizi": 8, "code_switched": 4}


# --------------------------------------------------------------------------------------------------- the case file


def test_case_file_is_valid_and_covers_the_plan() -> None:
    cases = harness.load_cases()
    faq_ids = harness.corpus_faq_ids()
    assert harness.validate_cases(cases, faq_ids, require_splits=True) == []
    for faq_id in faq_ids:
        for name, minimum in PER_FAQ_PER_SLICE.items():
            have = sum(c["slice"] == name and c["faq_id"] == faq_id for c in cases)
            assert have >= minimum, f"{faq_id}/{name}: {have} < {minimum}"
    labels = Counter(c["label"] for c in cases)
    # Cross-topic hard pairs are labeled positive for the FAQ that actually answers them;
    # only deliberate content gaps remain near-miss negatives.
    assert labels["near_miss"] >= 40 and labels["out_of_domain"] >= 12 and labels["degenerate"] >= 10


def test_every_near_miss_topic_is_spread_over_the_four_language_styles() -> None:
    near = [c for c in harness.load_cases() if c["label"] == "near_miss"]
    by_topic: dict[str, set[str]] = {}
    for case in near:
        by_topic.setdefault(case["topic"], set()).add(case["style"])
    assert len(by_topic) >= 8
    assert all(
        {"en", "msa", "egyptian", "arabizi"} <= styles
        for topic, styles in by_topic.items()
        if topic != "hours_contact"
    )


def _case(case_id: str, label: str = "positive", faq_id: str | None = "shipping-times", **extra) -> dict:
    slice_name = "msa" if label == "positive" else label
    base = {
        "id": case_id, "text": f"text {case_id}", "slice": slice_name, "style": "msa" if label == "positive" else "en",
        "label": label, "faq_id": faq_id if label == "positive" else None,
        "topic": faq_id if label == "positive" else "t", "near_faq": None,
        "review": {"status": "draft", "reviewer": None, "note": None},
    }
    return {**base, **extra}


def test_validation_reports_each_kind_of_mistake() -> None:
    faq_ids = ["shipping-times", "return-window"]
    cases = [
        _case("a"),
        _case("a"),  # duplicate id
        _case("b", text="text a"),  # duplicate text (same as the first case)
        _case("c", faq_id="unknown-faq"),
        _case("d", text="call me on 01012345678"),
        _case("e", label="bogus"),
        _case("f", label="near_miss", slice="positive"),
        _case("g", split="train"),
        _case("h", review={"status": "maybe"}),
    ]
    text = "\n".join(harness.validate_cases(cases, faq_ids))
    for expected in ("duplicate id", "same text as", "not in the corpus", "personal data", "unknown label", "negative case must use slice", "unknown split", "unknown review status"):
        assert expected in text, expected


def test_validation_flags_a_split_imbalance_and_missing_splits() -> None:
    cases = [_case(str(i), split="dev") for i in range(4)]
    assert any("split imbalance" in e for e in harness.validate_cases(cases, ["shipping-times"]))
    assert any("no split" in e for e in harness.validate_cases([_case("x")], ["shipping-times"], require_splits=True))


def test_split_assignment_is_balanced_deterministic_and_never_moves_old_cases() -> None:
    def build(count: int) -> list[dict]:
        return [_case(f"c{i:02d}") for i in range(count)]

    first = build(10)
    assert harness.assign_splits(first) == 10
    assert Counter(c["split"] for c in first) == {"dev": 5, "test": 5}
    again = build(10)
    harness.assign_splits(again)
    assert [c["split"] for c in first] == [c["split"] for c in again]

    grown = build(10) + [_case("c10"), _case("c11")]
    for old, kept in zip(grown, first):
        old["split"] = kept["split"]
    assert harness.assign_splits(grown) == 2
    assert [c["split"] for c in grown[:10]] == [c["split"] for c in first]
    assert Counter(c["split"] for c in grown) == {"dev": 6, "test": 6}
    assert harness.assign_splits(grown) == 0


def test_active_cases_drops_rejected_and_optionally_drafts() -> None:
    cases = [_case("a"), _case("b", review={"status": "approved"}), _case("c", review={"status": "edited"}), _case("d", review={"status": "rejected"})]
    assert [c["id"] for c in harness.active_cases(cases)] == ["a", "b", "c"]
    assert [c["id"] for c in harness.active_cases(cases, reviewed_only=True)] == ["b", "c"]


# ------------------------------------------------------------------------------------------- decisions and metrics


def _scored(case_id: str, label: str, faq_id: str | None, scores: dict[str, float], **extra) -> "harness.Scored":
    return harness.Scored(_case(case_id, label, faq_id or "shipping-times", **extra), harness.rank_faqs(scores))


def test_decide_applies_threshold_margin_and_sink() -> None:
    ranked = harness.rank_faqs({"a": 0.80, "b": 0.70, "c": 0.20})
    assert harness.decide(ranked, harness.Rule(0.75)) == "a"
    assert harness.decide(ranked, harness.Rule(0.85)) is None
    assert harness.decide(ranked, harness.Rule(0.75, margin=0.05)) == "a"
    assert harness.decide(ranked, harness.Rule(0.75, margin=0.15)) is None  # too close to the runner-up
    sink_wins = harness.rank_faqs({"a": 0.60, harness.NO_ANSWER_ID: 0.90})
    assert harness.decide(sink_wins, harness.Rule(0.10)) is None
    assert harness.decide(harness.rank_faqs({"a": 0.9}), harness.Rule(0.5, margin=0.5)) == "a"  # nothing to compare against


def test_classification_and_summary_partition_positives_and_count_false_accepts() -> None:
    items = [
        _scored("p1", "positive", "shipping-times", {"shipping-times": 0.9, "return-window": 0.1}),
        _scored("p2", "positive", "shipping-times", {"shipping-times": 0.4, "return-window": 0.1}),  # below threshold
        _scored("p3", "positive", "return-window", {"shipping-times": 0.8, "return-window": 0.3}),  # wrong FAQ
        _scored("n1", "near_miss", None, {"shipping-times": 0.7, "return-window": 0.1}),  # false accept
        _scored("n2", "near_miss", None, {"shipping-times": 0.2, "return-window": 0.1}),
        _scored("o1", "out_of_domain", None, {"shipping-times": 0.1, "return-window": 0.1}),
        _scored("d1", "degenerate", None, {"shipping-times": 0.6, "return-window": 0.2}),  # false accept
    ]
    results = harness.evaluate(items, harness.Rule(0.5))
    assert [r.outcome for r in results] == ["correct", "false_reject", "wrong_faq", "false_accept", "correct_reject", "correct_reject", "false_accept"]
    counts = harness.summary(results)
    assert (counts["n_pos"], counts["correct"], counts["wrong_faq"], counts["false_reject"]) == (3, 1, 1, 1)
    assert (counts["near_miss_n"], counts["near_miss_fa"]) == (2, 1)
    assert (counts["out_of_domain_n"], counts["out_of_domain_fa"]) == (1, 0)
    assert (counts["degenerate_n"], counts["degenerate_fa"]) == (1, 1)


def test_top1_and_mrr_do_not_depend_on_a_threshold() -> None:
    items = [
        _scored("p1", "positive", "shipping-times", {"shipping-times": 0.2, "return-window": 0.1, "payment-methods": 0.0}),
        _scored("p2", "positive", "shipping-times", {"shipping-times": 0.2, "return-window": 0.3, "payment-methods": 0.0}),
        _scored("p3", "positive", "shipping-times", {"shipping-times": 0.0, "return-window": 0.3, "payment-methods": 0.2}),
    ]
    top1, mrr = harness.top1_and_mrr(items)
    assert top1 == 1
    assert mrr == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


# ----------------------------------------------------------------------------------------------- tuning, calibration


def _tuning_set(near_scores: list[float], positive_scores: list[float]) -> list["harness.Scored"]:
    items = [_scored(f"p{i}", "positive", "shipping-times", {"shipping-times": s, "return-window": 0.0}) for i, s in enumerate(positive_scores)]
    items += [_scored(f"n{i}", "near_miss", None, {"shipping-times": s, "return-window": 0.0}) for i, s in enumerate(near_scores)]
    return items


def test_tuning_picks_the_midpoint_of_the_lowest_feasible_gap() -> None:
    tuned = harness.tune_threshold(_tuning_set([0.5, 0.4, 0.3], [0.9, 0.8, 0.7, 0.6]), margin=None, max_near_miss_fa=0.0, max_wrong_faq=0.0)
    assert tuned.threshold == pytest.approx(0.55)
    assert tuned.feasible and tuned.recall == 1.0


def test_tuning_allows_the_stated_false_accept_rate_to_buy_recall() -> None:
    near = [0.65] + [0.1] * 19  # 1 of 20 near-misses sits above the weakest positive
    tuned = harness.tune_threshold(_tuning_set(near, [0.9, 0.8, 0.7, 0.6]), margin=None, max_near_miss_fa=0.05, max_wrong_faq=0.0)
    assert tuned.recall == 1.0 and tuned.threshold < 0.6


def test_tuning_flags_overlap_as_infeasible_instead_of_crashing() -> None:
    tuned = harness.tune_threshold(_tuning_set([0.95, 0.9], [0.9, 0.8, 0.7, 0.6]), margin=None, max_near_miss_fa=0.0, max_wrong_faq=0.0, min_recall=0.8)
    assert not tuned.feasible and tuned.recall < 0.8


def test_tuning_needs_positives_and_near_misses() -> None:
    with pytest.raises(ValueError):
        harness.tune_threshold(_tuning_set([], [0.9]), margin=None, max_near_miss_fa=0.05, max_wrong_faq=0.01)


def test_candidate_thresholds_are_gap_midpoints_between_observed_scores() -> None:
    cuts = harness.candidate_thresholds([0.2, 0.4, 0.4, 0.8])
    assert cuts[0] == 0.2 and cuts[1:3] == [0.3, 0.6] and cuts[-1] > 0.8
    assert harness.candidate_thresholds([]) == []


def _calibration(positive: list[tuple[str, float]], negative: list[float]) -> dict:
    return {
        "positive": [{"query": f"p{i}", "expected": faq, "faq_best": {faq: score, "other": 0.01}} for i, (faq, score) in enumerate(positive)],
        "negative": [{"query": f"n{i}", "expected": None, "faq_best": {"other": score}} for i, score in enumerate(negative)],
    }


async def _production_calibrate(calibration: dict):
    """Run the real retrieval.calibration.calibrate() on the same scores, so the offline replica cannot drift."""
    from customer_service.infrastructure.faqs import FaqEntry
    from customer_service.retrieval.calibration import calibrate

    ranked = {}
    fixtures = {"positive": [], "negative": []}
    for key in ("positive", "negative"):
        for item in calibration[key]:
            top_faq, top_score = harness.rank_faqs(item["faq_best"])[0]
            ranked[item["query"]] = (FaqEntry(id=top_faq, title="t", answer="a"), top_score)
            fixtures[key].append({"query": item["query"], **({"faq_id": item["expected"]} if key == "positive" else {})})

    class Repository:
        async def best_similarities(self, queries):
            return [ranked[q] for q in queries]

    return await calibrate(Repository(), fixtures=fixtures)  # type: ignore[arg-type]


async def test_offline_replica_of_todays_calibration_matches_production_code() -> None:
    good = _calibration([("shipping-times", 0.8), ("return-window", 0.7)], [0.3, 0.2])
    threshold, _ = harness.today_threshold(good)
    assert threshold == (await _production_calibrate(good)).threshold == 0.5

    wrong = _calibration([("shipping-times", 0.8)], [0.3])
    wrong["positive"][0]["faq_best"] = {"return-window": 0.9, "shipping-times": 0.1}
    assert harness.today_threshold(wrong)[0] is None
    with pytest.raises(RuntimeError, match="did not retrieve"):
        await _production_calibrate(wrong)

    overlap = _calibration([("shipping-times", 0.4)], [0.4])
    threshold, note = harness.today_threshold(overlap)
    assert threshold is None and "overlap" in note
    with pytest.raises(RuntimeError, match="overlap"):
        await _production_calibrate(overlap)


# ------------------------------------------------------------------------------------------------------ statistics


def test_exact_bounds_match_the_rule_of_three_and_are_consistent() -> None:
    for n in (30, 60, 300):
        assert harness.cp_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)
        assert abs(harness.cp_upper(0, n) - 3 / n) < 0.01
    assert harness.cp_upper(0, 30) == pytest.approx(0.095, abs=1e-3)  # why 5% is not verifiable on 30 near-misses
    for k, n in ((1, 20), (5, 63), (60, 63)):
        assert harness._binom_cdf(k, n, harness.cp_upper(k, n)) == pytest.approx(0.05, abs=1e-9)
        assert 1 - harness._binom_cdf(k - 1, n, harness.cp_lower(k, n)) == pytest.approx(0.05, abs=1e-9)
        assert harness.cp_lower(k, n) < k / n < harness.cp_upper(k, n)
    assert harness.cp_upper(5, 5) == 1.0 and harness.cp_lower(0, 5) == 0.0
    assert math.isnan(harness.cp_upper(0, 0))


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    low, high = harness.bootstrap_ci(58, 63)
    assert (low, high) == harness.bootstrap_ci(58, 63)
    assert low < 58 / 63 < high <= 1.0
    assert harness.fmt_rate(0, 30).endswith("(95% upper bound <= 9.5%)")
    assert "lower bound" in harness.fmt_rate(10, 10) and "95% CI" in harness.fmt_rate(58, 63)
    assert harness.fmt_rate(0, 0) == "n/a"


def test_percentile_interpolates() -> None:
    assert harness.percentile([1, 2, 3, 4], 50) == 2.5
    assert harness.percentile([5], 95) == 5
    assert math.isnan(harness.percentile([], 50))


# --------------------------------------------------------------------------------------------- embedding and cache


class CountingEmbedder:
    name = "counting"

    def __init__(self, dims: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self._dims = dims

    def embed(self, texts, *, model, dims, input_type):
        self.calls.append(list(texts))
        width = self._dims or dims
        return [[float(len(t) % 7), float(i), 1.0, *([0.0] * (width - 3))] for i, t in enumerate(texts)]


def test_cache_embeds_each_text_once_and_survives_a_restart(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cached = harness.CachedEmbedder(inner, tmp_path, batch_size=2)
    first = cached.embed(["a", "bb", "a", "ccc"], model="m", dims=4, input_type="query")
    assert [text for call in inner.calls for text in call] == ["a", "bb", "ccc"]  # deduplicated
    assert [len(call) for call in inner.calls] == [2, 1]  # batched
    assert cached.embed(["ccc", "a"], model="m", dims=4, input_type="query") == [first[3], first[0]]
    assert len(inner.calls) == 2

    restarted_inner = CountingEmbedder()
    restarted = harness.CachedEmbedder(restarted_inner, tmp_path)
    assert restarted.embed(["a", "bb"], model="m", dims=4, input_type="query") == [first[0], first[1]]
    assert restarted_inner.calls == []


def test_cache_key_includes_model_dimensions_and_input_type(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cached = harness.CachedEmbedder(inner, tmp_path)
    cached.embed(["a"], model="m", dims=4, input_type="query")
    cached.embed(["a"], model="m", dims=4, input_type="document")  # a document vector must not answer a query lookup
    cached.embed(["a"], model="other", dims=4, input_type="query")
    cached.embed(["a"], model="m", dims=8, input_type="query")
    assert len(inner.calls) == 4


def test_cache_ignores_a_half_written_last_line_and_rejects_bad_vectors(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    harness.CachedEmbedder(inner, tmp_path).embed(["a"], model="m", dims=4, input_type="query")
    file = next(tmp_path.glob("*.jsonl"))
    file.write_text(file.read_text(encoding="utf-8") + '{"k": "abc", "v": [0.1', encoding="utf-8")
    reread = CountingEmbedder()
    assert harness.CachedEmbedder(reread, tmp_path).embed(["a"], model="m", dims=4, input_type="query")
    assert reread.calls == []
    with pytest.raises(RuntimeError, match="unexpected"):
        harness.CachedEmbedder(CountingEmbedder(dims=5), tmp_path / "bad").embed(["x"], model="m", dims=4, input_type="query")


def test_fake_embedder_is_deterministic_and_similar_texts_score_higher() -> None:
    fake = harness.FakeEmbedder()
    a, b, c = (harness.unit(v) for v in fake.embed(["how long is shipping", "how long is shipping?", "pay by card"], model="m", dims=128, input_type="query"))
    assert a == harness.unit(fake.embed(["how long is shipping"], model="m", dims=128, input_type="query")[0])
    assert sum(x * y for x, y in zip(a, b)) > sum(x * y for x, y in zip(a, c))


# ------------------------------------------------------------------------ end to end with the fake embedder and files


def _fake_run_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["run", "--embedder", "fake", "--dims", "64", "--raw", str(tmp_path / "raw.json"), "--split", "test", *extra]


def test_raw_output_reproduces_the_in_memory_numbers(tmp_path: Path) -> None:
    cases = harness.active_cases(harness.load_cases())
    index = harness.build_index("answers")
    raw = harness.build_raw(cases, harness.load_calibration_fixtures(), index, harness.FakeEmbedder(), arm="answers", model="m", dims=64)
    reloaded = json.loads(json.dumps(raw, ensure_ascii=False))
    rule = harness.Rule(0.3)
    assert harness.summary(harness.evaluate(harness.to_scored(cases, raw), rule)) == harness.summary(harness.evaluate(harness.to_scored(cases, reloaded), rule))
    assert len(raw["rows"]) == len(cases) and all(len(row["top"]) == 5 for row in raw["rows"])
    assert all(set(row["faq_best"]) == set(harness.corpus_faq_ids()) for row in raw["rows"])


def test_reuse_rebuilds_an_identical_report_from_the_saved_raw_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert harness.main(_fake_run_args(tmp_path)) == 0
    fresh = capsys.readouterr().out
    assert harness.main(_fake_run_args(tmp_path, "--reuse")) == 0
    reused = capsys.readouterr().out
    assert fresh.split("FAKE EMBEDDER", 1)[1] == reused.split("FAKE EMBEDDER", 1)[1]
    assert "DRAFT SET" in reused and "Highest-scoring negatives" in reused and "Threshold sweep" in reused


def test_reuse_refuses_stale_raw_output(tmp_path: Path) -> None:
    cases = harness.active_cases(harness.load_cases())
    raw = harness.build_raw(cases, harness.load_calibration_fixtures(), harness.build_index("answers"), harness.FakeEmbedder(), arm="answers", model="m", dims=64)
    edited = [dict(case) for case in cases]
    edited[0]["text"] += " (edited)"
    with pytest.raises(harness.StaleRawError, match="changed"):
        harness.to_scored(edited, raw)
    with pytest.raises(harness.StaleRawError, match="no raw scores"):
        harness.to_scored([*cases, {**cases[0], "id": "brand-new"}], raw)


def test_run_reports_a_rejected_case_file_and_a_bad_threshold(tmp_path: Path) -> None:
    broken = tmp_path / "cases.json"
    harness.write_cases([_case("x", split="dev"), _case("x", split="test")], broken)
    assert harness.main(["--cases", str(broken), "run", "--embedder", "fake", "--dims", "64", "--raw", str(tmp_path / "r.json")]) == 1
    with pytest.raises(SystemExit):
        harness.main(["run", "--threshold", "sometimes"])


def test_validate_and_splits_commands_work_on_a_scratch_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scratch = tmp_path / "cases.json"
    harness.write_cases([c for c in harness.load_cases() if c["label"] == "positive"][:12], scratch)
    stripped = harness.load_cases(scratch)
    for case in stripped:
        case.pop("split", None)
    harness.write_cases(stripped, scratch)
    assert harness.main(["--cases", str(scratch), "validate", "--require-splits"]) == 1  # no splits yet
    assert harness.main(["--cases", str(scratch), "splits"]) == 0
    assert harness.main(["--cases", str(scratch), "validate", "--require-splits"]) == 0
    assert "Assigned a split to 12" in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------------- latency probe


class FlakyProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts, *, input_type):
        self.calls += 1
        if self.calls % 4 == 0:
            raise TimeoutError("simulated timeout")
        return [[0.0] for _ in texts]


async def test_latency_probe_counts_errors_and_summarises_percentiles() -> None:
    rows = await harness.probe_embeddings(FlakyProvider(), ["q1", "q2", "q3"], 12)
    assert len(rows) == 12 and sum(row["error"] is not None for row in rows) == 3
    lines = "\n".join(harness.latency_summary(rows))
    assert "ok: 9" in lines and "failed: 3" in lines and "TimeoutError" in lines and "p95" in lines
    assert "25.0%" in lines
    assert "WARNING" not in lines  # a timeout is not a rate limit


def test_latency_summary_warns_when_calls_were_rate_limited() -> None:
    rows = [{"seconds": 0.2, "error": None}, {"seconds": 0.1, "error": "RateLimitError: reduced rate limits of 3 RPM"}]
    assert "WARNING: 1 call(s) were rate limited" in "\n".join(harness.latency_summary(rows))


# ------------------------------------------------------------------------------- Voyage rate limits (fake client)


class RateLimitError(Exception):
    """Same class name as voyageai.error.RateLimitError, which is how the harness recognises it."""


class FakeVoyageClient:
    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.failures = list(failures or [])
        self.calls = 0

    def embed(self, texts, model, input_type, output_dimension):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return type("Result", (), {"embeddings": [[0.0] * output_dimension for _ in texts]})()


class FakeClock:
    """Sleeping advances time instantly, so the tests never actually wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def _voyage(client: FakeVoyageClient, clock: FakeClock, **options) -> "harness.VoyageEmbedder":
    return harness.VoyageEmbedder("unused", client=client, sleep=clock.sleep, clock=clock, **options)


def _embed_once(embedder: "harness.VoyageEmbedder") -> list[list[float]]:
    return embedder.embed(["q"], model="m", dims=3, input_type="query")


def test_rate_limit_errors_back_off_for_at_least_the_per_minute_window() -> None:
    clock, client = FakeClock(), FakeVoyageClient([RateLimitError("3 RPM"), RateLimitError("3 RPM")])
    assert _embed_once(_voyage(client, clock)) == [[0.0, 0.0, 0.0]]
    assert client.calls == 3 and len(clock.sleeps) == 2
    assert 22 <= clock.sleeps[0] < 23 and 44 <= clock.sleeps[1] < 45  # the old 2 s and 3 s waits kept hitting the limit


def test_other_errors_still_use_the_short_backoff() -> None:
    clock, client = FakeClock(), FakeVoyageClient([ConnectionError("reset")])
    _embed_once(_voyage(client, clock))
    assert len(clock.sleeps) == 1 and 1 <= clock.sleeps[0] < 2


def test_retries_are_bounded_and_the_last_error_is_raised() -> None:
    clock, client = FakeClock(), FakeVoyageClient([RateLimitError("x")] * 5)
    with pytest.raises(RateLimitError):
        _embed_once(_voyage(client, clock, retries=2))
    assert client.calls == 3 and len(clock.sleeps) == 2


def test_rpm_pacing_spaces_requests_under_the_limit() -> None:
    clock, client = FakeClock(), FakeVoyageClient()
    embedder = _voyage(client, clock, rpm=3)
    for _ in range(3):
        _embed_once(embedder)
    assert clock.sleeps == [pytest.approx(21.0), pytest.approx(21.0)]  # 60/3 plus a 1 s margin; none before the first call
    unpaced = FakeClock()
    for _ in range(3):
        _embed_once(_voyage(FakeVoyageClient(), unpaced))
    assert unpaced.sleeps == []


def test_cli_defaults_favor_few_requests_and_accept_a_rate() -> None:
    assert harness.build_parser().parse_args(["run"]).batch_size == 128
    assert harness.build_parser().parse_args(["run"]).rpm == 0.0
    assert harness.build_parser().parse_args(["run", "--rpm", "3"]).rpm == 3.0


def test_default_batching_sends_the_whole_eval_set_in_a_handful_of_requests(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    texts = [f"query number {i}" for i in range(216)]  # 208 cases plus the calibration fixtures
    harness.CachedEmbedder(inner, tmp_path).embed(texts, model="m", dims=4, input_type="query")
    assert len(inner.calls) == 2


def test_ctrl_c_exits_cleanly_and_says_how_to_resume(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(harness, "cmd_run", interrupted)
    assert harness.main(["run"]) == 130
    assert "run the same command again to resume" in capsys.readouterr().err
