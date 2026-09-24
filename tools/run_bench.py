#!/usr/bin/env python3
"""Benchmark ONE model. Run once per model, switching the server in between.

    python tools/run_bench.py --model qwen3-vl --photos <folder>
    # switch the GPU server to the next model, then:
    python tools/run_bench.py --model pixtral-12b --photos <folder>
    ...
    python tools/bench_report.py bench_runs/<run_id>     # merge into a comparison

The server holds one model at a time, so this tool deliberately benchmarks one
model per invocation and writes one file per model. It refuses to start if the
server is not actually serving the model you named - see --model below, and the
guard in backend/bench/harness.py, which exists because a forgotten model switch
produces a complete, plausible, entirely wrong result file.

WHAT IT MEASURES

  latency scenarios   batch / continuous / parallel / ramp, over ONE question,
                      calling the model directly. This is the load picture.
  --modes             the full three-mode pipeline over the same photographs,
                      for per-stage timings and per-mode answers. This is the
                      "is it fast enough per photograph" and "what are its
                      answers worth" picture.

BEFORE YOU RUN IT AGAINST A SHARED GPU

  The parallel and ramp scenarios deliberately saturate the server. Anyone
  demoing on it at the time will see 503s. --scenario ramp requires --yes for
  that reason, and concurrency above 8 requires it too.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect(target: Path, limit=None):
    if target.is_file():
        paths = [target]
    else:
        paths = sorted(p for p in target.rglob("*")
                       if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit else paths


def mock_caller(latency_ms: float = 40.0):
    """A caller that answers instantly, for exercising the harness with no GPU.

    Every sample it produces is tagged MOCK, so metrics.ScenarioResult.warnings()
    refuses to let the numbers be read as latency. That tagging is the point:
    a mock run is for proving the harness works, never for a result.
    """
    import random

    from bench.metrics import MOCK, Sample

    def call(item):
        delay = max(0.0, random.gauss(latency_ms, latency_ms * 0.25)) / 1000.0
        time.sleep(delay)
        return Sample(wall_ms=delay * 1000, outcome=MOCK, server_ms=delay * 900,
                      model="mock", question_id=item.question_id, stem=item.stem,
                      answer="unknown", parse_tier="json", response_chars=120)

    return call


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="the model to benchmark. The run aborts unless the "
                         "server's registry lists it AND a real call comes back "
                         "attributed to it")
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--question", default="hazard_warning")
    ap.add_argument("--limit", type=int, default=8,
                    help="photographs to use (default 8)")
    ap.add_argument("--scenario", default="batch,parallel",
                    help="comma-separated: batch, continuous, parallel, ramp, all")
    ap.add_argument("--modes", action="store_true",
                    help="also run the full three-mode pipeline for per-stage "
                         "timings and per-mode answers")
    ap.add_argument("--repeats", type=int, default=2, help="batch passes")
    ap.add_argument("--rate", type=float, default=0.5, help="continuous: rps")
    ap.add_argument("--duration", type=float, default=60.0, help="continuous: seconds")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel: workers")
    ap.add_argument("--requests", type=int, default=20, help="parallel: total requests")
    ap.add_argument("--ramp-levels", default="1,2,4,8")
    ap.add_argument("--transport", choices=("direct", "proxy"), default=None)
    ap.add_argument("--gpu-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-request timeout in seconds. Raise it for a cold "
                         "load of a large model")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "bench_runs")
    ap.add_argument("--run-id", default=None,
                    help="share one id across every model so bench_report.py "
                         "can merge them (default: today's date)")
    ap.add_argument("--mock", action="store_true",
                    help="no GPU: exercise the harness itself. Every sample is "
                         "labelled a mock and is NOT a latency measurement")
    ap.add_argument("--yes", action="store_true",
                    help="confirm scenarios that deliberately saturate a shared GPU")
    args = ap.parse_args()

    from bench import harness, quality, scenarios
    from bench.metrics import OK
    from pipeline.config import default_config
    from pipeline.questions import get_question
    from pipeline.stage3_vlm import VLMClient

    wanted = ([s for s in scenarios.SCENARIOS if s != scenarios.RAMP]
              if args.scenario == "all"
              else [s.strip() for s in args.scenario.split(",") if s.strip()])
    unknown = [s for s in wanted if s not in scenarios.SCENARIOS]
    if unknown:
        print(f"unknown scenario(s) {unknown}; known: {list(scenarios.SCENARIOS)}")
        return 2

    saturating = [s for s in wanted if s in (scenarios.RAMP,)]
    if (saturating or args.concurrency > scenarios.DEFAULT_MAX_CONCURRENCY) \
            and not args.yes and not args.mock:
        print("This run deliberately saturates the GPU server, which will make\n"
              "anyone else's requests fail with 503 while it lasts.\n"
              "Re-run with --yes once you know nobody is demoing on it.")
        return 2

    overrides = {"vlm_model": args.model}
    if args.transport:
        overrides["vlm_transport"] = args.transport
    if args.gpu_url:
        overrides["gpu_url"] = args.gpu_url
    if args.api_key:
        overrides["vlm_api_key"] = args.api_key
    if args.timeout:
        overrides["request_timeout_s"] = args.timeout
    cfg = default_config(**overrides)
    question = get_question(args.question)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model      : {args.model}")
    print(f"question   : {question.id}")
    print(f"transport  : {cfg.vlm_transport} -> {cfg.gpu_url}")
    print(f"scenarios  : {', '.join(wanted)}{'  + three-mode pipeline' if args.modes else ''}")
    print(f"writing to : {out_dir}\n")

    paths = collect(Path(args.photos), args.limit)
    if not paths:
        print(f"no images under {args.photos}")
        return 1

    report = {
        "run_id": run_id, "model": args.model, "question": question.id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "transport": cfg.vlm_transport, "gpu_url": cfg.gpu_url,
        "photographs": [p.name for p in paths],
        "is_mock": bool(args.mock),
        "sampling": {"max_new_tokens": cfg.max_new_tokens,
                     "temperature": cfg.temperature, "top_p": cfg.top_p,
                     "top_k": cfg.top_k},
        "scenarios": {}, "warnings": [],
    }
    if args.mock:
        report["warnings"].append(
            "MOCK RUN - no model was called. These timings measure the harness "
            "and must never be quoted as model latency.")

    # ── Prepare once ─────────────────────────────────────────────────────────
    print("encoding photographs once (re-encoding per request would put the "
          "JPEG encoder in the measurement)...")
    items, failed = harness.prepare_items(paths, question.id, cfg)
    if failed:
        report["warnings"].append(f"{len(failed)} photographs could not be read: "
                                  f"{'; '.join(failed[:3])}")
        print(f"  {len(failed)} unreadable, skipped")
    if not items:
        print("no usable photographs")
        return 1
    print(f"  {len(items)} ready\n")

    # ── The guard ────────────────────────────────────────────────────────────
    if args.mock:
        call = mock_caller()
        report["model_check"] = {"ok": True, "detail": "mock run - no server asked"}
    else:
        client = VLMClient(cfg)
        check = harness.verify_model(client, args.model)
        report["model_check"] = {"ok": check.ok, "detail": check.detail,
                                 "registry": sorted(check.registry)}
        print(f"model check: {check.detail}")

        # What the server was actually running, recorded WITH the numbers. A
        # latency figure without the configuration that produced it is an
        # anecdote: a model given twice the KV cache queues later, which looks
        # exactly like being faster.
        snapshot = harness.serving_snapshot(client)
        report["serving"] = {
            "config": harness.config_of(snapshot, args.model),
            "resident_models": harness.resident_models(snapshot),
            "health": snapshot.get("health"),
            "metrics_before": snapshot.get("metrics"),
        }
        serving = report["serving"]["config"]
        resident = report["serving"]["resident_models"]
        if serving:
            print(f"  serving: TP={serving.get('tensor_parallel_size')} "
                  f"gpu_mem={serving.get('gpu_memory_utilization')} "
                  f"max_concurrent={serving.get('max_concurrent')} "
                  f"max_model_len={serving.get('max_model_len')}")
        if resident:
            print(f"  resident on the box: {', '.join(str(r) for r in resident)}")
            # On a one-model-at-a-time server, a model that is not resident is
            # about to be lazy-loaded inside the warm-up - which is fine and is
            # what the cold-load figure measures. A DIFFERENT one being resident
            # is the thing worth flagging before minutes of GPU time are spent.
            others = [r for r in resident if r not in (args.model, "mistral")]
            if others:
                note = (f"another VLM is resident ({', '.join(others)}) - loading "
                        f"{args.model!r} will evict it, and the first request "
                        f"will pay that load")
                report["warnings"].append(note)
                print(f"  ! {note}")
        if not check.ok:
            print("\nAborting: benchmarking a model the server is not serving "
                  "produces a result file that looks fine and is wrong.")
            json.dump(report, open(out_dir / f"{args.model}.json", "w"), indent=2)
            return 1
        call = harness.make_caller(client, question, cfg)

    # ── Warm up, and price the cold load ─────────────────────────────────────
    print("warming up (the first call may be a cold model load)...")
    warm = harness.warmup(call, items[0], rounds=2)
    cold_sample = warm.pop("_cold_sample")
    report["warmup"] = warm
    if warm.get("apparent_load_ms"):
        print(f"  cold {warm['cold_ms']:.0f} ms, warm {warm['warm_best_ms']:.0f} ms "
              f"-> about {warm['apparent_load_ms']:.0f} ms of model load, "
              f"reported separately and excluded from every latency figure")
    elif warm["cold_outcome"] != OK:
        print(f"  warm-up did not succeed ({warm['cold_outcome']}) - continuing, "
              f"but expect the scenarios to fail the same way")

    if not args.mock:
        refusal = harness.confirm_served_model(cold_sample, args.model)
        if refusal:
            print("\n" + refusal)
            report["warnings"].append(refusal)
            report["model_check"]["ok"] = False
            json.dump(report, open(out_dir / f"{args.model}.json", "w"), indent=2)
            return 1
        print(f"  server answered as {cold_sample.model or '(unnamed)'}\n")

    def progress(i, total, name):
        if total:
            print(f"\r  {name}: {i}/{total}", end="", flush=True)
        else:
            print(f"\r  {name}: {i} sent", end="", flush=True)

    # ── Scenarios ────────────────────────────────────────────────────────────
    for name in wanted:
        print(f"\n{name}:")
        if name == scenarios.BATCH:
            result = scenarios.run_batch(call, items, model=args.model,
                                         repeats=args.repeats, progress=progress)
            report["scenarios"][name] = result.to_dict()
        elif name == scenarios.CONTINUOUS:
            result = scenarios.run_continuous(
                call, items, model=args.model, rate_rps=args.rate,
                duration_s=args.duration, progress=progress)
            report["scenarios"][name] = result.to_dict()
        elif name == scenarios.PARALLEL:
            result = scenarios.run_parallel(
                call, items, model=args.model, concurrency=args.concurrency,
                requests=args.requests, progress=progress)
            report["scenarios"][name] = result.to_dict()
        elif name == scenarios.RAMP:
            levels = tuple(int(x) for x in args.ramp_levels.split(",") if x.strip())
            results = scenarios.run_ramp(call, items, model=args.model,
                                         levels=levels, progress=progress)
            report["scenarios"][name] = {
                "levels": [r.to_dict(include_samples=False) for r in results],
                "saturation": scenarios.saturation_point(results),
            }
            result = results[-1] if results else None
        print()
        if name != scenarios.RAMP and result is not None:
            stats = result.latency()
            if stats.n:
                print(f"  {result.ok}/{result.total} ok   "
                      f"p50 {stats.p50_ms:.0f} ms   p95 {stats.p95_ms:.0f} ms   "
                      f"max {stats.max_ms:.0f} ms   "
                      f"{result.throughput_rps:.2f} rps")
            elif args.mock:
                # Named at every turn so a mock number cannot escape into a
                # report as a latency figure.
                harness_stats = result.harness_latency()
                print(f"  harness check only: {result.total} mock requests, "
                      f"p50 {harness_stats.p50_ms:.0f} ms, "
                      f"{result.throughput_rps if result.throughput_rps else 0:.2f} rps "
                      f"- NOT a latency measurement")
            for w in result.warnings():
                print(f"  ! {w}")

    # ── Answer quality, from every sample we collected ───────────────────────
    all_samples = []
    for name in wanted:
        block = report["scenarios"].get(name) or {}
        if name == scenarios.RAMP:
            continue
        all_samples.extend(block.get("samples", []))

    class _S:                      # the dicts back into something attribute-ish
        def __init__(self, row):
            self.__dict__.update(row)

    samples = [_S(r) for r in all_samples]
    aq = quality.summarise_answers(samples, model=args.model)
    report["answer_quality"] = {
        "total": aq.total, "tiers": aq.tiers, "answers": aq.answers,
        "contract_compliance": aq.contract_compliance,
        "unparseable": aq.unparseable, "unknown_rate": aq.unknown_rate,
        "verdict": aq.verdict,
    }
    report["answers_by_key"] = quality.answers_by_key(samples)
    print(f"\nanswer quality: {aq.verdict}")

    # ── The full pipeline, all three modes ───────────────────────────────────
    if args.modes:
        print("\nthree-mode pipeline over the same photographs...")
        from pipeline.modes import MODE_ORDER, run_all_modes

        cfg.run_id = f"bench_{run_id}_{args.model}"
        try:
            results = run_all_modes(paths, question.id, cfg)
            report["modes"] = {}
            for mode in MODE_ORDER:
                records = results.get(mode, [])
                report["modes"][mode] = {
                    "photographs": len(records),
                    "answers": {},
                    "per_stage_ms": quality.per_stage_breakdown(records),
                }
                counts = {}
                for r in records:
                    key = r.vlm.answer if r.vlm else "stopped"
                    counts[key] = counts.get(key, 0) + 1
                report["modes"][mode]["answers"] = counts
                print(f"  {mode:9} {counts}")
        except Exception as exc:
            report["warnings"].append(f"the three-mode run failed: {exc}")
            print(f"  failed: {exc}")

    # A second snapshot: the server's own counters over the run, and the VRAM
    # it ended up at. /metrics is cumulative, so before-and-after is what makes
    # it readable, and the VRAM reading is the only way to see whether the
    # model fitted comfortably or scraped in.
    if not args.mock:
        after = harness.serving_snapshot(client)
        report["serving"]["metrics_after"] = after.get("metrics")
        report["serving"]["health_after"] = after.get("health")

    dest = out_dir / f"{args.model}.json"
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {dest}")
    print(f"\nNow switch the server to the next model and run this again with "
          f"--run-id {run_id}.\nWhen every model is done: "
          f"python tools/bench_report.py {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
