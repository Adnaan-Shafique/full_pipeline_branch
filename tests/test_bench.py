"""The benchmark harness: what may enter a latency figure, and what may not.

Most of this suite is about REFUSAL. A benchmark's failure mode is not crashing
- it is producing a confident number that is wrong, which nobody catches
because it looks like every other number. So the assertions below are mostly
"this must not be counted", "this must be flagged", "this must abort".

Runs with nothing installed: every scenario is driven by a fake caller with
known timings, so no GPU, no cv2, no requests.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


from bench import harness, quality, scenarios                          # noqa: E402
from bench.metrics import (ERROR, MOCK, OK, SATURATED, TIMEOUT,        # noqa: E402
                           LatencyStats, Sample, ScenarioResult, percentile)


def sample(ms, outcome=OK, **kw):
    return Sample(wall_ms=ms, outcome=outcome, **kw)


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def result_of(samples, wall_s=1.0, scenario="batch", model="m"):
    r = ScenarioResult(scenario=scenario, model=model, wall_s=wall_s)
    r.samples = list(samples)
    return r


print("\na failed request is not a fast request")
# The failure that makes a saturated server look quick: a 4 ms connection
# refusal entering the latency distribution as a 4 ms response.
r = result_of([sample(100), sample(300),
               sample(4, SATURATED), sample(2, ERROR), sample(9, TIMEOUT)])
check("only successful requests are timed", r.latency().n == 2, str(r.latency().n))
check("and the fast failures do not drag the minimum down",
      r.latency().min_ms == 100, str(r.latency().min_ms))
check("every request is still counted", r.total == 5)
check("the outcomes are broken out", r.counts()[SATURATED] == 1
      and r.counts()[ERROR] == 1 and r.counts()[TIMEOUT] == 1)
check("success rate reflects reality", abs(r.success_rate - 0.4) < 1e-9)
for outcome, fragment in ((SATURATED, "503"), (TIMEOUT, "timed out"), (ERROR, "errored")):
    check(f"{outcome} is called out in the warnings",
          any(fragment in w for w in r.warnings()), str(r.warnings()))

print("\na mock is not a measurement")
m = result_of([sample(1, MOCK), sample(2, MOCK)])
check("mocks never enter the latency figures", m.latency().n == 0)
check("but are visible as their own timing", m.harness_latency().n == 2)
check("and the result says loudly that no model looked",
      any("no model looked" in w for w in m.warnings()), str(m.warnings()))
check("a result with nothing successful says so",
      any("NOTHING succeeded" in w for w in m.warnings()))
check("the serialised form names the mock timing so it cannot be quoted",
      "harness_only_timing_NOT_latency" in m.to_dict())
check("a real run carries no such key value",
      result_of([sample(10)]).to_dict()["harness_only_timing_NOT_latency"] is None)

print("\npercentiles are nearest-rank, and say when they mean nothing")
check("p50 of 1..100", percentile(range(1, 101), 50) == 50)
check("p95 of 1..100", percentile(range(1, 101), 95) == 95)
check("p99 of 1..100", percentile(range(1, 101), 99) == 99)
check("p100 is the max", percentile(range(1, 101), 100) == 100)
check("p0 is the min", percentile(range(1, 101), 0) == 1)
check("an empty set yields None, not zero", percentile([], 95) is None)
# Interpolation would invent a value between two real samples and read as
# precision that is not there.
check("the value returned actually occurred in the data",
      percentile([10, 20, 30], 95) in (10, 20, 30))
small = LatencyStats.from_values([1, 2, 3])
check("three samples carry a caveat", "not meaningful" in small.caveat, small.caveat)
mid = LatencyStats.from_values(list(range(50)))
check("fifty samples still caveat p99", "indicative" in mid.caveat, mid.caveat)
big = LatencyStats.from_values(list(range(500)))
check("five hundred samples need no caveat", big.caveat == "", big.caveat)
check("no samples is stated, not silently zero",
      LatencyStats.from_values([]).n == 0
      and "no successful requests" in LatencyStats.from_values([]).caveat)

print("\nclient time, server time, and the gap between them")
s = sample(500, server_ms=430)
check("transport cost is the difference", s.transport_ms == 70)
check("a server that reported nothing yields None, not zero",
      sample(500).transport_ms is None)
# Negative would mean the server claims to have taken longer than the client
# waited - clock skew, not a measurement.
check("a nonsensical negative gap is clamped to zero",
      sample(400, server_ms=450).transport_ms == 0.0)

print("\none scenario, one model")
mixed = result_of([sample(10, model="qwen3-vl"), sample(12, model="molmo-72b")])
check("two models answering inside one scenario is flagged",
      any("more than one model" in w for w in mixed.warnings()), str(mixed.warnings()))

print("\nthe arrival patterns do what they claim")
calls = {"n": 0, "peak": 0, "inflight": 0}
import threading  # noqa: E402
import time  # noqa: E402

_lock = threading.Lock()


def fake_call(item, delay=0.01):
    with _lock:
        calls["n"] += 1
        calls["inflight"] += 1
        calls["peak"] = max(calls["peak"], calls["inflight"])
    time.sleep(delay)
    with _lock:
        calls["inflight"] -= 1
    return sample(delay * 1000, model="m", stem=getattr(item, "stem", str(item)),
                  question_id="q", answer="yes", parse_tier="json")


ITEMS = [harness.WorkItem(stem=f"p{i}", question_id="q") for i in range(3)]

calls.update(n=0, peak=0, inflight=0)
b = scenarios.run_batch(fake_call, ITEMS, model="m", repeats=2)
check("batch sends items x repeats", b.total == 6, str(b.total))
check("batch is strictly sequential", calls["peak"] == 1, str(calls["peak"]))
check("batch records its configuration", b.config["repeats"] == 2)

calls.update(n=0, peak=0, inflight=0)
p = scenarios.run_parallel(fake_call, ITEMS, model="m", concurrency=3, requests=9)
check("parallel sends the requested count", p.total == 9, str(p.total))
check("parallel actually overlaps", calls["peak"] > 1, str(calls["peak"]))
check("parallel respects its concurrency cap", calls["peak"] <= 3, str(calls["peak"]))
check("throughput is answered requests over wall time",
      p.throughput_rps is not None and p.throughput_rps > 0)

calls.update(n=0, peak=0, inflight=0)
c = scenarios.run_continuous(fake_call, ITEMS, model="m", rate_rps=20,
                             duration_s=0.5, max_inflight=4)
check("continuous runs for about its duration", 0.4 <= c.wall_s <= 1.5, str(c.wall_s))
check("continuous sent something", c.total > 0, str(c.total))
check("continuous records schedule lag", c.max_schedule_lag_ms >= 0)
# Open-loop is the only pattern that can show a queue, and it is only honest
# while the generator keeps up.
lagged = result_of([sample(10, schedule_lag_ms=900)])
check("a generator that fell behind invalidates its own rate",
      any("behind its own schedule" in w for w in lagged.warnings()),
      str(lagged.warnings()))

print("\nthe ramp stops at the knee instead of flogging a dead server")
def failing_above(limit):
    state = {"inflight": 0}
    lk = threading.Lock()

    def call(item):
        with lk:
            state["inflight"] += 1
            over = state["inflight"] > limit
        try:
            time.sleep(0.005)
            if over:
                return sample(5, SATURATED, model="m", error="HTTP 503")
            return sample(10, model="m", stem="p", question_id="q",
                          answer="yes", parse_tier="json")
        finally:
            with lk:
                state["inflight"] -= 1

    return call


levels = scenarios.run_ramp(failing_above(2), ITEMS, model="m",
                            levels=(1, 2, 4, 8), requests_per_level=8)
check("the ramp stopped early rather than running every level",
      len(levels) < 4, f"ran {len(levels)} levels")
check("and the level that stopped it says why",
      any(n.startswith("STOPPING THE RAMP") for n in levels[-1].notes),
      str(levels[-1].notes))
sat = scenarios.saturation_point(levels)
check("the saturation point names the highest healthy concurrency",
      sat["highest_healthy_concurrency"] in (1, 2), str(sat))
check("and reports the reason it stopped", "STOPPING" in sat["reason_it_stopped"])
check("every level is reported, not just the last", len(sat["levels"]) == len(levels))

clean = scenarios.run_ramp(fake_call, ITEMS, model="m", levels=(1, 2),
                           requests_per_level=4)
check("a ramp that finds no limit says so rather than inventing one",
      "did not find a limit" in scenarios.saturation_point(clean)["reason_it_stopped"])

print("\nthe model guard - the failure that voids a whole run")


class FakeClient:
    def __init__(self, registry, error=None):
        self.registry = registry
        self._error = error

    def refresh_registry(self):
        return list(self.registry), self._error


ok_registry = {"qwen3-vl": {"modality": "vision", "max_images": 4},
               "mistral": {"modality": "text"}}
check("a registered vision model passes",
      harness.verify_model(FakeClient(ok_registry), "qwen3-vl").ok)
missing = harness.verify_model(FakeClient(ok_registry), "molmo-72b")
check("a model the server does not list is refused", not missing.ok)
check("and the refusal lists what IS available",
      "mistral" in missing.detail and "qwen3-vl" in missing.detail, missing.detail)
text_only = harness.verify_model(FakeClient(ok_registry), "mistral")
check("a text-only model is refused before any image is sent", not text_only.ok)
check("and the reason names the modality", "vision" in text_only.detail, text_only.detail)
unreachable = harness.verify_model(FakeClient({}, error="connection refused"), "x")
check("an unreadable registry is refused, not assumed empty", not unreachable.ok)
check("and says the registry could not be read",
      "registry could not be read" in unreachable.detail, unreachable.detail)

# The one that matters most: the server lists the model, but is SERVING another.
wrong = harness.confirm_served_model(sample(10, model="qwen3-vl"), "molmo-72b")
check("a server answering as a different model aborts the run", wrong is not None)
check("and the message names the real cause - a forgotten model switch",
      "never switched" in wrong, str(wrong))
check("the matching case passes",
      harness.confirm_served_model(sample(10, model="qwen3-vl"), "qwen3-vl") is None)
check("a server that named no model does not trigger a false alarm",
      harness.confirm_served_model(sample(10, model=""), "qwen3-vl") is None)

print("\ncold model load is priced, not hidden")
seq = iter([sample(9000, model="m"), sample(300, model="m"), sample(280, model="m")])
warm = harness.warmup(lambda item: next(seq), ITEMS[0], rounds=3)
check("the cold call is reported on its own", warm["cold_ms"] == 9000)
check("the warm baseline is the best of the rest", warm["warm_best_ms"] == 280)
check("the apparent load time is the gap", warm["apparent_load_ms"] == 8720)
check("and it is explained, not left as a bare number",
      "excluded from every latency distribution" in warm["note"])
# A model that was already resident has no load to report, and must not be
# given a fabricated one.
seq2 = iter([sample(300, model="m"), sample(310, model="m")])
warm2 = harness.warmup(lambda item: next(seq2), ITEMS[0], rounds=2)
check("an already-warm model reports no load time rather than a negative one",
      warm2["apparent_load_ms"] is None, str(warm2["apparent_load_ms"]))
seq3 = iter([sample(5, ERROR), sample(5, ERROR)])
warm3 = harness.warmup(lambda item: next(seq3), ITEMS[0], rounds=2)
check("a warm-up that never succeeded reports no cold time",
      warm3["cold_ms"] is None and warm3["cold_outcome"] == ERROR)

print("\nerror classification - 503 is load, not breakage")
check("a 503 is its own outcome",
      harness.classify_error(RuntimeError("HTTP 503 Service Unavailable"))[0] == SATURATED)
check("a timeout is its own outcome",
      harness.classify_error(TimeoutError("Read timed out"))[0] == TIMEOUT)
check("anything else is an error",
      harness.classify_error(ConnectionError("refused"))[0] == ERROR)

print("\ncontract compliance - the metric a tolerant parser hides")
from pipeline.stage3_vlm import PARSE_TIERS  # noqa: E402

check("the bench's tier list matches the parser's, so they cannot drift",
      set(quality.TIER_ORDER) == set(PARSE_TIERS),
      f"{quality.TIER_ORDER} vs {PARSE_TIERS}")

perfect = quality.summarise_answers(
    [sample(10, answer="yes", parse_tier="json") for _ in range(10)], model="a")
check("a model that always emits JSON is 100% compliant",
      perfect.contract_compliance == 1.0)
check("and is called usable", "usable" in perfect.verdict, perfect.verdict)

# The failure this metric exists for: the parser rescues prose, so the answers
# look fine while the model is one prompt edit from producing nothing.
proseish = quality.summarise_answers(
    [sample(10, answer="yes", parse_tier="prose") for _ in range(8)]
    + [sample(10, answer="no", parse_tier="json") for _ in range(2)], model="b")
check("a model rescued by the parser's tolerance is flagged fragile",
      "fragile" in proseish.verdict, proseish.verdict)
check("and the number behind it is visible", proseish.contract_compliance == 0.2)

broken = quality.summarise_answers(
    [sample(10, answer="unknown", parse_tier="give_up") for _ in range(5)]
    + [sample(10, answer="yes", parse_tier="json") for _ in range(5)], model="c")
check("a model whose replies cannot be parsed is called UNUSABLE",
      "UNUSABLE" in broken.verdict, broken.verdict)
check("unparseable is measured separately from unknown",
      broken.unparseable == 0.5 and broken.unknown_rate == 0.5)

vague = quality.summarise_answers(
    [sample(10, answer="unknown", parse_tier="json") for _ in range(9)]
    + [sample(10, answer="yes", parse_tier="json")], model="d")
check("a compliant model that decides nothing is still called out",
      "not deciding anything" in vague.verdict, vague.verdict)
# A request that never landed says nothing about answer quality either way.
check("failed requests are not counted as answers",
      quality.summarise_answers([sample(10, SATURATED)]).total == 0)

print("\nagreement, with no ground truth to appeal to")
by_model = {
    "a": {"p1 :: q :: m": "yes", "p2 :: q :: m": "no", "p3 :: q :: m": "yes"},
    "b": {"p1 :: q :: m": "yes", "p2 :: q :: m": "yes", "p3 :: q :: m": "yes"},
    "c": {"p1 :: q :: m": "yes", "p2 :: q :: m": "no"},
}
matrix = quality.agreement_matrix(by_model)
check("only items EVERY model answered are compared", matrix["compared"] == 2,
      str(matrix["compared"]))
check("unanimity is counted over that shared set", matrix["unanimous"] == 1)
check("and splits too", matrix["split"] == 1)
pair_ab = next(p for p in matrix["pairs"] if p["a"] == "a" and p["b"] == "b")
check("a pair is scored over what BOTH answered, not the union",
      pair_ab["compared"] == 3 and abs(pair_ab["agreement"] - 2 / 3) < 1e-9,
      str(pair_ab))
splits = quality.disagreements(by_model)
check("the disagreements are listed for eyeballing", len(splits) == 1)
check("and name every model's answer", set(splits[0]["answers"]) == {"a", "b", "c"})
check("one model alone yields no comparison", quality.disagreements({"a": {}}) == [])

print("\nkeys are safe to print into a markdown table")
key = quality.answer_key("site_01", "hazard_warning", "classic")
check("the key separator is not a pipe", "|" not in key, key)
check("a pipe in a key would split a table cell silently - so none is produced",
      "|" not in "".join(quality.answers_by_key(
          [sample(10, answer="yes", stem="s", question_id="q")])))

print("\nper-stage breakdown")


class FakeStage:
    def __init__(self, ms):
        self.elapsed_ms = ms


class FakeVLM:
    def __init__(self, s):
        self.elapsed_s = s


class FakeRecord:
    def __init__(self, q, d, o, v):
        self.quality, self.detection, self.ocr, self.vlm = q, d, o, v


records = [FakeRecord(FakeStage(2000), FakeStage(3000), FakeStage(300), FakeVLM(0.4))
           for _ in range(3)]
stages = quality.per_stage_breakdown(records)
check("every timed stage appears",
      set(stages["_measured_stages"]) == {"quality", "detection", "ocr", "vlm"},
      str(stages["_measured_stages"]))
check("the vlm's seconds are converted to milliseconds",
      abs(stages["vlm"]["p50_ms"] - 400) < 1e-6, str(stages["vlm"]["p50_ms"]))
# A stage reporting 0.0 was never measured. Listing it as instant would send
# someone optimising the wrong thing.
unmeasured = quality.per_stage_breakdown(
    [FakeRecord(FakeStage(0), FakeStage(3000), None, None)])
check("a stage that was not measured is absent, not reported as instant",
      "quality" not in unmeasured["_measured_stages"],
      str(unmeasured["_measured_stages"]))

print("\nwill it fit? the arithmetic that decides what can be benchmarked")
from bench import capacity  # noqa: E402

# The finding this module exists for. 72B x 2 bytes is 144 GB against a 141 GB
# card - Molmo-72B cannot load on one H200 in bf16 at ANY gpu_memory_utilization.
molmo = capacity.fit_on_gpu("molmo-72b", 72, dtype="bf16", gpu_memory_utilization=0.95)
check("molmo-72b does not fit on one H200 in bf16", not molmo.fits, molmo.reason)
check("and the reason names the weights, not the budget",
      "at any gpu_memory_utilization" in molmo.reason, molmo.reason)
# FP8 halves the weights and is the only way to honour "one GPU" for it.
molmo_fp8 = capacity.fit_on_gpu("molmo-72b", 72, dtype="fp8",
                                gpu_memory_utilization=0.90)
check("at fp8 it fits with real KV cache left",
      molmo_fp8.usable and molmo_fp8.kv_gb > 40, molmo_fp8.reason)
check("smallest_working_dtype picks fp8, not something more aggressive",
      capacity.smallest_working_dtype("molmo-72b", 72,
                                      gpu_memory_utilization=0.90) == "fp8")
# Sharding is the other way out, and the module should say so.
check("TP=2 makes bf16 fit",
      capacity.fit_on_gpu("molmo-72b", 72, dtype="bf16", tensor_parallel=2,
                          gpu_memory_utilization=0.85).usable)

# The three that do fit on one card in bf16.
for name, params in (("pixtral-12b", 12), ("qwen3-vl", 30), ("internvl-38b", 38)):
    fit = capacity.fit_on_gpu(name, params, dtype="bf16", gpu_memory_utilization=0.85)
    check(f"{name} fits on one H200 in bf16", fit.usable, fit.reason)

# internvl at TP=1 needs more than its shipped 0.40: 38B bf16 is 76 GB and
# 0.40 of 141 is 56. This is the edit the runbook calls out, pinned.
shipped = capacity.fit_on_gpu("internvl-38b", 38, dtype="bf16",
                              gpu_memory_utilization=0.40, tensor_parallel=1)
check("internvl at its shipped 0.40 does NOT fit on one card", not shipped.fits,
      shipped.reason)
check("and the reason points at the budget, not the card",
      "raise gpu_memory_utilization" in shipped.reason, shipped.reason)

# Anything already resident comes out of the budget - mistral's ~21 GB is the
# difference between internvl loading and not.
with_mistral = capacity.fit_on_gpu("internvl-38b", 38, dtype="bf16",
                                   gpu_memory_utilization=0.85,
                                   already_resident_gb=21.0)
without = capacity.fit_on_gpu("internvl-38b", 38, dtype="bf16",
                              gpu_memory_utilization=0.85)
check("a resident model reduces the KV headroom by its own size",
      abs((without.kv_gb - with_mistral.kv_gb) - 21.0) < 1e-6,
      f"{without.kv_gb} vs {with_mistral.kv_gb}")

# A model that technically loads but has no cache left is flagged, because its
# concurrency numbers would describe the cache rather than the model.
tight = capacity.fit_on_gpu("tight", 60, dtype="bf16", gpu_memory_utilization=0.88)
check("a model with almost no KV cache loads but is not usable",
      tight.fits and not tight.usable, tight.reason)
check("and says the concurrency figures would be about the cache",
      "describes the cache" in tight.reason, tight.reason)
check("an unknown dtype is rejected rather than guessed",
      _raises(lambda: capacity.fit_on_gpu("x", 1, dtype="float4")))

print("\nthe CLI refuses what it should")
cli = (ROOT / "tools" / "run_bench.py").read_text()
check("saturating scenarios need explicit confirmation", "--yes" in cli
      and "deliberately saturates" in cli)
check("the model guard runs before any measurement",
      cli.index("verify_model") < cli.index("run_batch"))
check("a failed guard aborts rather than writing a plausible file",
      "Aborting: benchmarking a model the server is not serving" in cli)
check("images are encoded once, outside the measurement",
      "prepare_items" in cli and "encoding photographs once" in cli)
check("a mock run is labelled in the report it writes",
      'report["warnings"].append(' in cli and "MOCK RUN" in cli)
# Behaviour, not a grep: the warning text is built across two source lines, so
# a substring check on the file passed or failed on formatting rather than on
# what the tool does.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("bench_report", ROOT / "tools" / "bench_report.py")
_report = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_report)

_mock_run = {"model": "a", "is_mock": True, "model_check": {"ok": True}}
_bad_guard = {"model": "b", "is_mock": False,
              "model_check": {"ok": False, "detail": "wrong model"}}
_good = {"model": "c", "is_mock": False, "model_check": {"ok": True}}
_ranked = _report.rankable({"a": _mock_run, "b": _bad_guard, "c": _good})
check("the report excludes mock runs from the ranking", "a" not in _ranked)
check("and guard-failed runs too", "b" not in _ranked)
check("while keeping the sound ones", "c" in _ranked)

_warn = _report.comparability_warnings({"a": _mock_run, "b": _bad_guard})
check("a mock run is called out by the report", any("MOCK RUN" in w for w in _warn))
check("a failed guard voids its file", any("void" in w for w in _warn), str(_warn))

# Models measured over different photographs or settings cannot be tabulated
# beside each other, and a table does not show that unless it is said.
_diff = _report.comparability_warnings({
    "x": {"model": "x", "question": "hazard_warning", "photographs": ["a.jpg"],
          "sampling": {"temperature": 0.0}, "transport": "direct"},
    "y": {"model": "y", "question": "gps_antenna", "photographs": ["b.jpg"],
          "sampling": {"temperature": 0.7}, "transport": "proxy"}})
for _what in ("question", "sampling settings", "photograph set", "transport"):
    check(f"a differing {_what} is reported as not comparable",
          any(_what in w and "comparable" in w for w in _diff), str(_diff))
check("identical settings raise no comparability warning",
      not _report.comparability_warnings({
          "x": {"model": "x", "question": "q", "photographs": ["a"],
                "sampling": {}, "transport": "direct"},
          "y": {"model": "y", "question": "q", "photographs": ["a"],
                "sampling": {}, "transport": "direct"}}))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
