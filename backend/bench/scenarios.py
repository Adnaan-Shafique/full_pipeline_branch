"""The three ways requests arrive, and the ramp that finds where it breaks.

There is no token streaming anywhere in this pipeline - `/infer` returns a
complete answer - so "streaming" here means how requests ARRIVE, not how
tokens leave. Three patterns, and they answer different questions:

    batch       N requests back to back, one at a time. "How long does a
                folder of photographs take?" No concurrency, so this is the
                cleanest per-request latency you will get.

    continuous  requests offered at a fixed rate for a fixed duration,
                OPEN-LOOP: the next one is sent when the clock says so, not
                when the previous one finished. "Can it keep up with a field
                crew uploading steadily?" This is the only pattern that can
                show a queue forming.

    parallel    K requests in flight at once, closed-loop: each worker sends
                the next as soon as its previous returns. "What does it do
                with K users?" Reports throughput as well as latency, because
                latency alone makes every concurrent system look worse.

    ramp        parallel at K = 1, 2, 4, 8 … until latency or errors cross a
                threshold. Finds the knee rather than guessing it.

── Open-loop vs closed-loop, and why it matters ────────────────────────────
Closed-loop load (parallel) cannot overload a server: if the server slows, the
client sends less, and the system quietly finds an equilibrium that hides the
problem. Open-loop load (continuous) keeps offering work regardless, so a queue
builds and the latency you measure includes the waiting - which is what a real
user experiences. Benchmarks that only do closed-loop routinely report a system
as healthy right up until it collapses.

The catch is that an open-loop generator can itself fall behind. Every sample
records `schedule_lag_ms`, and `ScenarioResult` warns when it grows, because at
that point the harness is no longer offering the rate it claims.

`call` is always a function taking one work item and returning a Sample. It is
injected rather than imported so these functions can be tested against a fake
with known timings, and so nothing here needs a GPU.
"""
from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from .metrics import SATURATED, Sample, ScenarioResult

BATCH = "batch"
CONTINUOUS = "continuous"
PARALLEL = "parallel"
RAMP = "ramp"
SCENARIOS = (BATCH, CONTINUOUS, PARALLEL, RAMP)

# Past this many requests in flight we are no longer characterising the server,
# we are just queueing on it - and on a SHARED GPU we are also ruining someone
# else's afternoon. The CLI requires an explicit flag to go higher.
DEFAULT_MAX_CONCURRENCY = 8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cycle(items):
    """Endless supply of work items, in order, repeating."""
    return itertools.cycle(list(items))


def run_batch(call: Callable, items: Iterable, model: str = "",
              repeats: int = 1, progress: Optional[Callable] = None) -> ScenarioResult:
    """Every item, `repeats` times, strictly one at a time."""
    work = list(items) * max(1, repeats)
    result = ScenarioResult(scenario=BATCH, model=model, started_at=_now(),
                            config={"items": len(items), "repeats": repeats,
                                    "requests": len(work), "concurrency": 1})
    t0 = time.perf_counter()
    for index, item in enumerate(work, start=1):
        if progress:
            progress(index, len(work), BATCH)
        result.samples.append(call(item))
    result.wall_s = time.perf_counter() - t0
    return result


def run_continuous(call: Callable, items: Iterable, model: str = "",
                   rate_rps: float = 0.5, duration_s: float = 60.0,
                   max_inflight: int = DEFAULT_MAX_CONCURRENCY,
                   progress: Optional[Callable] = None) -> ScenarioResult:
    """Offer requests at `rate_rps` for `duration_s`, open-loop.

    Requests go out on the clock, not on completion, so if the server is slower
    than the offered rate they overlap and a queue forms - which is the entire
    point of this scenario.

    `max_inflight` is a safety valve, not part of the measurement: without it an
    overloaded server turns this into an unbounded fork bomb against a shared
    GPU. When it engages the scenario says so in its notes, because at that
    moment the offered rate stopped being the configured one.
    """
    supply = _cycle(items)
    result = ScenarioResult(
        scenario=CONTINUOUS, model=model, started_at=_now(),
        config={"rate_rps": rate_rps, "duration_s": duration_s,
                "max_inflight": max_inflight})
    interval = 1.0 / rate_rps if rate_rps > 0 else 0.0
    lock = threading.Lock()
    inflight = 0
    throttled = 0

    def one(item, scheduled_at):
        nonlocal inflight
        lag = max(0.0, (time.perf_counter() - scheduled_at) * 1000)
        sample = call(item)
        sample.schedule_lag_ms = lag
        with lock:
            result.samples.append(sample)
            inflight -= 1

    t0 = time.perf_counter()
    sent = 0
    with ThreadPoolExecutor(max_workers=max(1, max_inflight)) as pool:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= duration_s:
                break
            scheduled_at = t0 + sent * interval
            wait = scheduled_at - time.perf_counter()
            if wait > 0:
                time.sleep(min(wait, duration_s - elapsed))
            with lock:
                if inflight >= max_inflight:
                    throttled += 1
                    # Skip this slot rather than queueing it locally: a local
                    # queue would measure this harness's backlog, not the
                    # server's.
                    sent += 1
                    continue
                inflight += 1
            pool.submit(one, next(supply), scheduled_at)
            sent += 1
            if progress:
                progress(sent, 0, CONTINUOUS)
    result.wall_s = time.perf_counter() - t0
    if throttled:
        result.notes.append(
            f"{throttled} scheduled requests were SKIPPED because {max_inflight} "
            f"were already in flight - the offered rate was therefore below the "
            f"configured {rate_rps} rps, and the server is the reason")
    return result


def run_parallel(call: Callable, items: Iterable, model: str = "",
                 concurrency: int = 4, requests: int = 20,
                 progress: Optional[Callable] = None) -> ScenarioResult:
    """`concurrency` workers, closed-loop, until `requests` have been sent."""
    supply = _cycle(items)
    result = ScenarioResult(
        scenario=PARALLEL, model=model, started_at=_now(),
        config={"concurrency": concurrency, "requests": requests})
    lock = threading.Lock()
    work = [next(supply) for _ in range(requests)]
    done = 0

    def one(item):
        nonlocal done
        sample = call(item)
        with lock:
            result.samples.append(sample)
            done += 1
            if progress:
                progress(done, requests, PARALLEL)
        return sample

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(one, work))
    result.wall_s = time.perf_counter() - t0
    return result


def run_ramp(call: Callable, items: Iterable, model: str = "",
             levels=(1, 2, 4, 8), requests_per_level: int = 12,
             p95_ceiling_ms: Optional[float] = None,
             error_ceiling: float = 0.10,
             progress: Optional[Callable] = None) -> list:
    """Parallel at each level until something gives. Returns one
    ScenarioResult per level actually run.

    Stops early on the first level that breaches a ceiling, and records WHY in
    that level's notes. Continuing past the knee tells you nothing you did not
    already know and costs the shared GPU real time.

    `p95_ceiling_ms` defaults to four times the p95 measured at the lowest
    level - a relative ceiling, because an absolute one would have to be
    guessed per model and molmo-72b is not qwen.
    """
    out = []
    baseline_p95 = None
    for level in levels:
        result = run_parallel(call, items, model=model, concurrency=level,
                              requests=requests_per_level, progress=progress)
        result.config["ramp_level"] = level
        stats = result.latency()
        if baseline_p95 is None and stats.p95_ms:
            baseline_p95 = stats.p95_ms
        ceiling = p95_ceiling_ms or (baseline_p95 * 4 if baseline_p95 else None)

        counts = result.counts()
        failed = counts[SATURATED] + counts["timeout"] + counts["error"]
        error_rate = failed / result.total if result.total else 0.0
        out.append(result)

        if error_rate > error_ceiling:
            result.notes.append(
                f"STOPPING THE RAMP: {error_rate:.0%} of requests failed at "
                f"concurrency {level} (ceiling {error_ceiling:.0%}). This is the "
                f"saturation point - the server refused work rather than slowing "
                f"down, which is what a 503 means here")
            break
        if ceiling and stats.p95_ms and stats.p95_ms > ceiling:
            result.notes.append(
                f"STOPPING THE RAMP: p95 reached {stats.p95_ms:.0f} ms at "
                f"concurrency {level}, past the {ceiling:.0f} ms ceiling. Beyond "
                f"this the queue, not the model, is what you are measuring")
            break
    return out


def saturation_point(ramp_results) -> dict:
    """Read a ramp: the highest concurrency that still behaved, and what ended
    it. This is the number the "how much load does the demo survive" question
    actually wants."""
    best, reason = None, "no level breached a ceiling - the ramp did not find a limit"
    for result in ramp_results:
        level = result.config.get("ramp_level")
        stopped = [n for n in result.notes if n.startswith("STOPPING THE RAMP")]
        if stopped:
            reason = stopped[0]
            break
        if result.ok:
            best = level
    return {
        "highest_healthy_concurrency": best,
        "reason_it_stopped": reason,
        "levels": [
            {"concurrency": r.config.get("ramp_level"),
             "ok": r.ok, "total": r.total,
             "p50_ms": r.latency().p50_ms, "p95_ms": r.latency().p95_ms,
             "throughput_rps": r.throughput_rps,
             "counts": r.counts()}
            for r in ramp_results],
    }
