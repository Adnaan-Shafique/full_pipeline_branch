"""Latency samples and the statistics drawn from them.

A benchmark is a machine for producing confident numbers, which is exactly what
makes it dangerous. Most of this module is about refusing to produce one.

Three rules it enforces:

  · **A failed request is not a fast request.** An error, a 503 or a timeout
    must never enter the latency distribution. Counting a 40 ms connection
    refusal as a 40 ms response makes a saturated server look quick. Outcomes
    are counted separately from timings, and `LatencyStats` is built only from
    samples that actually returned an answer.

  · **A mock is not a measurement.** A run with no model behind it returns in
    microseconds. Those samples are tagged and the report refuses to present
    them as latency.

  · **Percentiles, not means.** The mean of a bimodal distribution - which is
    what you get the moment a queue forms - describes nothing that happened.
    p95 and max are the numbers that decide whether a demo stalls.

Pure stdlib, no cv2, no requests: importable and assertable anywhere.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Optional

# Sample.outcome values.
OK = "ok"                    # a real answer from a real model
MOCK = "mock"                # the mock path - no model looked
SATURATED = "saturated"      # HTTP 503: every concurrency slot busy
TIMEOUT = "timeout"          # no response inside request_timeout_s
ERROR = "error"              # anything else: connection refused, 4xx, 5xx
OUTCOMES = (OK, MOCK, SATURATED, TIMEOUT, ERROR)

# Only these contribute to latency statistics.
TIMED_OUTCOMES = (OK,)


@dataclass
class Sample:
    """One request, from the client's point of view.

    `wall_ms` is what the caller experienced: encode-free, but including
    network, queueing on the server, inference, and the response coming back.
    `server_ms` is what the server said it spent. The GAP between them is the
    transport and queueing cost, and it is the number that answers "is the
    proxy expensive" and "is there a queue forming" - so both are kept, never
    just one.
    """

    wall_ms: float
    outcome: str = OK
    server_ms: Optional[float] = None
    model: str = ""              # what the SERVER said answered, not what we asked
    scenario: str = ""
    mode: str = ""
    question_id: str = ""
    stem: str = ""
    answer: str = ""
    parse_tier: str = ""         # see stage3_vlm.PARSE_TIERS
    response_chars: int = 0
    error: str = ""
    # ── Straight from the server, when it offers them ────────────────────────
    # Time spent waiting for one of the GPU server's max_concurrent semaphore
    # slots. Rising queue_wait with flat server time is a queue forming, which
    # is invisible in wall-clock alone - the single most useful saturation
    # signal here.
    #
    # WHERE IT COMES FROM MATTERS. gpu_api_server_v7 measures the wait around
    # the semaphore itself and llm_proxy_v4 passes that figure through.
    # llm_proxy_v3 could only approximate it as proxy_elapsed - elapsed, which
    # is queueing PLUS network PLUS proxy - and on a fast, unloaded server that
    # approximation is almost entirely transport, so reading it as queue wait
    # says the server is saturated when nothing is queueing at all. v4 sets
    # queue_wait_is_approximate so the two can never be tabulated as one.
    queue_wait_ms: Optional[float] = None
    queue_wait_approx: bool = False
    proxy_ms: Optional[float] = None
    # v4 only: what is left of the round trip once the server's own accounting
    # and the queue wait are subtracted - network, JSON encoding of a
    # multi-megabyte data URI, and the proxy. "The proxy is expensive" and "the
    # GPU is busy" are opposite findings, and this is what tells them apart.
    transport_overhead_ms: Optional[float] = None
    # Output length, because a model that writes longer answers takes longer.
    # Comparing latency across models without this compares VERBOSITY, not
    # speed - tokens_per_sec is the figure that survives that.
    new_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None

    @property
    def tokens_per_sec(self) -> Optional[float]:
        if not self.new_tokens or not self.server_ms:
            return None
        return self.new_tokens / (self.server_ms / 1000.0)
    # Open-loop scenarios only: how late this request STARTED against its
    # scheduled arrival time. A rising value means the load generator itself is
    # falling behind, which invalidates the arrival rate it claims to be
    # producing - see scenarios.continuous().
    schedule_lag_ms: float = 0.0

    @property
    def transport_ms(self) -> Optional[float]:
        """Client wall-clock minus the server's own elapsed time."""
        if self.server_ms is None:
            return None
        return max(0.0, self.wall_ms - self.server_ms)

    def to_row(self) -> dict:
        row = asdict(self)
        row["transport_ms"] = self.transport_ms
        row["tokens_per_sec"] = self.tokens_per_sec
        return row


def percentile(values, q: float) -> Optional[float]:
    """Nearest-rank percentile of `values` at q in [0, 100].

    Nearest-rank rather than interpolated on purpose: an interpolated p99 over
    12 samples invents a number between two real ones and reads as precision
    that is not there. This returns a value that actually occurred.
    """
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if q <= 0:
        return data[0]
    if q >= 100:
        return data[-1]
    rank = math.ceil(q / 100.0 * len(data))
    return data[min(len(data), max(1, rank)) - 1]


@dataclass
class LatencyStats:
    """Summary of the timed samples. `n` is how many actually counted."""

    n: int = 0
    mean_ms: Optional[float] = None
    stdev_ms: Optional[float] = None
    min_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    max_ms: Optional[float] = None
    # How far each percentile can be trusted. With 12 samples a "p99" is just
    # the maximum wearing a label, and saying so beats printing it bare.
    caveat: str = ""

    @classmethod
    def from_values(cls, values) -> "LatencyStats":
        data = [v for v in values if v is not None]
        if not data:
            return cls(n=0, caveat="no successful requests to measure")
        stats = cls(
            n=len(data),
            mean_ms=statistics.fmean(data),
            stdev_ms=statistics.pstdev(data) if len(data) > 1 else 0.0,
            min_ms=min(data), max_ms=max(data),
            p50_ms=percentile(data, 50), p90_ms=percentile(data, 90),
            p95_ms=percentile(data, 95), p99_ms=percentile(data, 99),
        )
        # A percentile needs roughly 1/(1-q) samples before it means anything.
        if len(data) < 100:
            stats.caveat = (f"only {len(data)} samples - p99 is effectively the "
                            f"maximum; treat p95 and above as indicative")
        if len(data) < 20:
            stats.caveat = (f"only {len(data)} samples - percentiles above p50 "
                            f"are not meaningful")
        return stats


@dataclass
class ScenarioResult:
    """Everything one scenario produced, ready to serialise."""

    scenario: str
    model: str
    samples: list = field(default_factory=list)
    started_at: str = ""
    wall_s: float = 0.0
    config: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # ── Outcomes ─────────────────────────────────────────────────────────────
    def counts(self) -> dict:
        out = {o: 0 for o in OUTCOMES}
        for s in self.samples:
            out[s.outcome] = out.get(s.outcome, 0) + 1
        return out

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def ok(self) -> int:
        return sum(1 for s in self.samples if s.outcome in TIMED_OUTCOMES)

    @property
    def success_rate(self) -> Optional[float]:
        return (self.ok / self.total) if self.total else None

    # ── Latency ──────────────────────────────────────────────────────────────
    def latency(self) -> LatencyStats:
        """Client-side wall-clock over the requests that actually answered."""
        return LatencyStats.from_values(
            [s.wall_ms for s in self.samples if s.outcome in TIMED_OUTCOMES])

    def harness_latency(self) -> LatencyStats:
        """Timings of the MOCK samples, for proving the scenario machinery runs.

        Kept apart from latency() and named so it cannot be mistaken for one: a
        mock returns in microseconds and describes this harness, never a model.
        It exists so a --mock run can show that batch, continuous and parallel
        actually do what they claim without a GPU in the room.
        """
        return LatencyStats.from_values(
            [s.wall_ms for s in self.samples if s.outcome == MOCK])

    def server_latency(self) -> LatencyStats:
        return LatencyStats.from_values(
            [s.server_ms for s in self.samples if s.outcome in TIMED_OUTCOMES])

    def transport_latency(self) -> LatencyStats:
        return LatencyStats.from_values(
            [s.transport_ms for s in self.samples if s.outcome in TIMED_OUTCOMES])

    def queue_wait(self) -> LatencyStats:
        """Time spent waiting for a concurrency slot, as the proxy reports it.

        This is the measurement that separates "the model is slow" from "the
        model is busy". With max_concurrent=2 on the VLMs, a third caller waits
        here rather than being refused, so a rising queue wait against a flat
        server time is the saturation signal - the 503 only arrives once the
        wait exceeds the server's QUEUE_TIMEOUT_S.

        Against llm_proxy_v3 these values are an approximation that also
        contains network and proxy time; see `Sample.queue_wait_ms` and the
        warning in `warnings()`.
        """
        return LatencyStats.from_values(
            [s.queue_wait_ms for s in self.samples if s.outcome in TIMED_OUTCOMES])

    @property
    def queue_wait_is_approximate(self) -> bool:
        """True when ANY timed sample's queue wait was inferred rather than
        measured. Any, not all: one approximated value in a distribution is
        enough to make its percentiles mean something different."""
        return any(s.queue_wait_approx for s in self.samples
                   if s.outcome in TIMED_OUTCOMES and s.queue_wait_ms is not None)

    def output_tokens(self) -> LatencyStats:
        """Reusing the stats shape for token counts: a model that writes twice
        as much is not twice as slow, and this is what tells them apart."""
        return LatencyStats.from_values(
            [float(s.new_tokens) for s in self.samples
             if s.outcome in TIMED_OUTCOMES and s.new_tokens])

    def transport_overhead(self) -> LatencyStats:
        """Round trip minus the server's own time minus the queue wait.

        Only llm_proxy_v4 reports it; everywhere else this is empty, which is
        correct - the quantity cannot be separated out without a measured
        queue wait to subtract.
        """
        return LatencyStats.from_values(
            [s.transport_overhead_ms for s in self.samples
             if s.outcome in TIMED_OUTCOMES])

    def tokens_per_sec(self) -> LatencyStats:
        return LatencyStats.from_values(
            [s.tokens_per_sec for s in self.samples if s.outcome in TIMED_OUTCOMES])

    @property
    def throughput_rps(self) -> Optional[float]:
        """Answered requests per second of wall time. The number that actually
        matters under concurrency - a per-request latency that doubles while
        throughput also doubles is a win, not a regression."""
        if not self.wall_s:
            return None
        return self.ok / self.wall_s

    @property
    def max_schedule_lag_ms(self) -> float:
        """Worst arrival-time slip. Open-loop scenarios are only honest while
        this stays small; past that the generator is describing a load it did
        not actually produce."""
        return max((s.schedule_lag_ms for s in self.samples), default=0.0)

    # ── Honesty checks ───────────────────────────────────────────────────────
    def warnings(self) -> list:
        """Everything about this result that should stop someone quoting it."""
        out = list(self.notes)
        counts = self.counts()
        if counts[MOCK]:
            out.append(
                f"{counts[MOCK]} of {self.total} requests were MOCKS - no model "
                f"looked at them. These timings measure this harness, not a "
                f"model, and must not be reported as latency")
        if counts[SATURATED]:
            out.append(
                f"{counts[SATURATED]} requests came back 503 (server saturated). "
                f"They are excluded from the latency figures, which therefore "
                f"describe only the requests that got through")
        if counts[TIMEOUT]:
            out.append(f"{counts[TIMEOUT]} requests timed out and are excluded "
                       f"from the latency figures")
        if counts[ERROR]:
            out.append(f"{counts[ERROR]} requests errored and are excluded from "
                       f"the latency figures")
        if self.ok == 0 and self.total:
            out.append("NOTHING succeeded - there are no latency figures here at all")
        if self.max_schedule_lag_ms > 250:
            out.append(
                f"the load generator fell up to {self.max_schedule_lag_ms:.0f} ms "
                f"behind its own schedule, so the arrival rate it reports was not "
                f"the rate actually offered")
        queue = self.queue_wait()
        server = self.server_latency()
        if queue.n and self.queue_wait_is_approximate:
            out.append(
                "queue wait here is APPROXIMATED as proxy time minus server "
                "time, which also contains network and proxy overhead - it is "
                "an upper bound on queueing, not a measurement of it. Run "
                "against gpu_api_server_v7 through llm_proxy_v4 for the real "
                "figure")
        if queue.n and queue.p95_ms and server.p50_ms and queue.p95_ms > server.p50_ms:
            out.append(
                f"p95 queue wait ({queue.p95_ms:.0f} ms) exceeds the median time "
                f"the model itself spent ({server.p50_ms:.0f} ms) - most of the "
                f"latency here is WAITING FOR A SLOT, not inference. Compare "
                f"against max_concurrent for this model before reading these "
                f"numbers as model speed"
                + (" (and note the queue figure is approximated here)"
                   if self.queue_wait_is_approximate else ""))
        tokens = self.output_tokens()
        if tokens.n and tokens.stdev_ms and tokens.mean_ms and \
                tokens.stdev_ms > tokens.mean_ms * 0.5:
            out.append(
                f"output length varies widely ({tokens.mean_ms:.0f} tokens mean, "
                f"{tokens.stdev_ms:.0f} stdev) - some of this latency spread is "
                f"the model choosing to write more, not to think longer")
        served = {s.model for s in self.samples if s.model}
        if len(served) > 1:
            out.append(f"more than one model answered during this scenario "
                       f"({sorted(served)}) - the result mixes models and cannot "
                       f"be attributed to one")
        return out

    def to_dict(self, include_samples: bool = True) -> dict:
        out = {
            "scenario": self.scenario, "model": self.model,
            "started_at": self.started_at, "wall_s": round(self.wall_s, 3),
            "config": self.config,
            "total": self.total, "ok": self.ok,
            "success_rate": self.success_rate,
            "counts": self.counts(),
            "throughput_rps": self.throughput_rps,
            "latency_client_ms": asdict(self.latency()),
            "latency_server_ms": asdict(self.server_latency()),
            "latency_transport_ms": asdict(self.transport_latency()),
            "queue_wait_ms": asdict(self.queue_wait()),
            "queue_wait_is_approximate": self.queue_wait_is_approximate,
            "transport_overhead_ms": asdict(self.transport_overhead()),
            "output_tokens": asdict(self.output_tokens()),
            "tokens_per_sec": asdict(self.tokens_per_sec()),
            # Present only on mock runs, and named so nobody quotes it.
            "harness_only_timing_NOT_latency": (
                asdict(self.harness_latency()) if self.counts()[MOCK] else None),
            "max_schedule_lag_ms": round(self.max_schedule_lag_ms, 1),
            "warnings": self.warnings(),
        }
        if include_samples:
            out["samples"] = [s.to_row() for s in self.samples]
        return out
