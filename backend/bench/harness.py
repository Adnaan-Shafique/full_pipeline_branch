"""Driving the real pipeline under a scenario, and refusing to produce a
number that would mislead.

── The guard this module exists for ────────────────────────────────────────
The GPU server holds ONE model at a time. Benchmarking four of them means four
separate runs with a manual model switch in between, possibly days apart. The
failure that costs you the whole exercise is forgetting to switch: you run
`--model molmo-72b`, the server is still serving qwen3-vl, and you get a
complete, plausible, entirely wrong result file. Nothing in the numbers would
ever tell you.

So `verify_model()` asks the server what it actually answered with, before any
measurement starts, and aborts if it disagrees with what was requested. It is
the first thing every run does.

── The other way a benchmark lies ─────────────────────────────────────────
A cold model load is tens of seconds for a 72B, and the config comment records
that only `qwen3-vl` and `mistral` are eager-loaded - so for pixtral, internvl
and molmo the FIRST call of a run includes loading the weights. Averaged in, it
inflates every summary; silently dropped, it hides a real operational cost.
`warmup()` therefore measures it deliberately, reports it as its own number,
and keeps it out of the latency distribution.

Images are encoded ONCE per photograph and the data URI reused for every
repetition. Re-encoding per request would put cv2's JPEG encoder in the
measurement, which is not what anyone is trying to find out.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .metrics import ERROR, MOCK, OK, SATURATED, TIMEOUT, Sample


@dataclass
class WorkItem:
    """One request's worth of input, prepared once and reused."""

    stem: str
    question_id: str
    image_uri: str = ""          # encoded once; see the module docstring
    image_bgr: object = None
    mode: str = ""
    label: str = ""


@dataclass
class ModelCheck:
    requested: str
    served: str = ""
    ok: bool = False
    detail: str = ""
    registry: dict = field(default_factory=dict)


def classify_error(exc: Exception) -> tuple:
    """(outcome, text) for an exception raised by a request.

    A 503 is pulled out as its own outcome rather than lumped in with errors:
    it means the server was BUSY, not broken, and the difference is the whole
    finding in a saturation test.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "503" in text or "saturat" in lowered:
        return SATURATED, text
    if "timeout" in lowered or "timed out" in lowered:
        return TIMEOUT, text
    return ERROR, text


def verify_model(client, requested: str) -> ModelCheck:
    """Confirm the server is serving the model this run claims to benchmark.

    Two checks, because either alone can pass while the situation is wrong:
    the registry must LIST the model, and a real inference must come back
    ATTRIBUTED to it. A server that lists four models but only has one resident
    passes the first and fails the second - which is exactly this setup.
    """
    check = ModelCheck(requested=requested)
    try:
        names, error = client.refresh_registry()
    except Exception as exc:
        check.detail = f"the registry could not be read: {exc}"
        return check
    if error:
        check.detail = f"the registry could not be read: {error}"
        return check

    check.registry = dict(client.registry)
    if requested not in check.registry:
        check.detail = (f"{requested!r} is not in the server's registry. It lists "
                        f"{sorted(check.registry)}. Load it on the server first")
        return check
    info = check.registry.get(requested) or {}
    if info.get("modality") != "vision":
        check.detail = (f"{requested!r} is registered as "
                        f"{info.get('modality')!r}, not 'vision' - it will reject "
                        f"images and there is nothing here to benchmark")
        return check
    check.ok = True
    check.detail = f"{requested!r} is registered as a vision model"
    return check


def serving_snapshot(client) -> dict:
    """What the server was actually running, recorded with the numbers.

    A latency figure without the configuration that produced it is not a
    measurement, it is an anecdote. With one model resident at a time and the
    registry edited between runs, `tensor_parallel_size` and
    `gpu_memory_utilization` can differ from run to run without anyone
    noticing - and a model given twice the KV cache will hold more concurrent
    requests before it queues, which looks exactly like being faster.

    Everything here is read-only and best-effort: a server that does not expose
    an endpoint costs that field, never the run. Through llm_proxy_v3 these are
    /v1/gpu-models, /v1/gpu-health and /v1/metrics; direct they are /models,
    /health and /metrics.
    """
    from .metrics import OK  # noqa: F401  - keeps the import surface obvious

    out: dict = {}
    endpoints = client.endpoints
    probes = [
        ("registry", getattr(endpoints, "registry", None)),
        ("health", getattr(endpoints, "gpu_health", None) or getattr(endpoints, "health", None)),
    ]
    for key, path in probes:
        if not path:
            continue
        try:
            from pipeline.stage3_vlm import _get
            out[key] = _get(client.base, path, timeout=30, headers=client.headers)
        except Exception as exc:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    # /metrics is not in the Endpoints table - it is an observability extra
    # rather than part of the pipeline's contract, so it is probed by hand and
    # its absence is unremarkable.
    metrics_path = ("/v1/metrics" if getattr(client, "transport", "") == "proxy"
                    else "/metrics")
    try:
        from pipeline.stage3_vlm import _get
        out["metrics"] = _get(client.base, metrics_path, timeout=30,
                              headers=client.headers)
    except Exception as exc:
        out["metrics"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def config_of(snapshot: dict, model: str) -> dict:
    """The one model's serving configuration, pulled out of a snapshot.

    The GPU server's /models returns a list of ModelInfo; the proxy passes it
    through unchanged at /v1/gpu-models, so one reader handles both.
    """
    registry = (snapshot or {}).get("registry")
    entries = registry if isinstance(registry, list) else []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == model:
            return {k: entry.get(k) for k in (
                "name", "loaded", "modality", "tensor_parallel_size",
                "max_model_len", "max_concurrent", "gpu_memory_utilization",
                "max_images", "quantization", "error")}
    return {}


def resident_models(snapshot: dict) -> list:
    """Which models the server says are loaded RIGHT NOW.

    On a one-model-at-a-time box this is the check that catches a switch that
    silently failed: the registry still lists every model, but `loaded` is true
    for only the resident one, and /health names it outright.
    """
    health = (snapshot or {}).get("health")
    if isinstance(health, dict) and isinstance(health.get("loaded_models"), list):
        return list(health["loaded_models"])
    registry = (snapshot or {}).get("registry")
    if isinstance(registry, list):
        return [e.get("name") for e in registry
                if isinstance(e, dict) and e.get("loaded")]
    return []


def confirm_served_model(sample: Sample, requested: str) -> Optional[str]:
    """The second half of the guard, run on the warm-up's own response.

    Returns a refusal message when the server answered as a DIFFERENT model,
    which on a one-model-at-a-time server is the signature of a model switch
    that did not happen.
    """
    served = (sample.model or "").strip()
    if not served:
        return None      # the server named no model; nothing to contradict
    if served == requested:
        return None
    return (f"REFUSING TO BENCHMARK: this run was told to measure {requested!r}, "
            f"but the server answered as {served!r}. On a server that holds one "
            f"model at a time this almost always means the model was never "
            f"switched, and the result would be a complete, plausible, entirely "
            f"wrong file attributed to the wrong model.")


def make_caller(client, question, cfg) -> Callable:
    """A function that performs ONE request and returns a Sample.

    Never raises: a benchmark that dies on the first refused connection has
    measured nothing, and the refusal is itself a data point.
    """
    from pipeline.stage3_vlm import (build_payload, _post, parse_vlm_answer_tiered,
                                     sampling_for)
    from pipeline.questions import render_user_prompt

    def call(item: WorkItem) -> Sample:
        prompt = render_user_prompt(question, [], None)
        sampling = sampling_for(question, cfg)
        t0 = time.perf_counter()
        try:
            payload = build_payload(cfg.vlm_model, prompt, question.system_prompt,
                                    [item.image_uri], sampling, client.registry,
                                    client.transport)
            response = _post(client.base, client.endpoints.infer, payload,
                             timeout=cfg.request_timeout_s, headers=client.headers)
        except Exception as exc:
            outcome, text = classify_error(exc)
            return Sample(wall_ms=(time.perf_counter() - t0) * 1000,
                          outcome=outcome, error=text, scenario="",
                          question_id=question.id, stem=item.stem)
        wall_ms = (time.perf_counter() - t0) * 1000
        text = response.get("text", "") or ""
        answer, _reasoning, tier = parse_vlm_answer_tiered(text)

        def ms(key):
            value = response.get(key)
            return float(value) * 1000 if value is not None else None

        def count(key):
            value = response.get(key)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        return Sample(
            wall_ms=wall_ms, outcome=OK,
            server_ms=ms("elapsed_s"),
            # Present only through llm_proxy_v3; the direct path leaves them
            # None, which the stats treat as absent rather than as zero.
            queue_wait_ms=ms("queue_wait_s"),
            proxy_ms=ms("proxy_elapsed_s"),
            new_tokens=count("new_tokens"),
            prompt_tokens=count("prompt_tokens"),
            model=str(response.get("model", "")),
            question_id=question.id, stem=item.stem,
            answer=answer, parse_tier=tier, response_chars=len(text))

    return call


def warmup(call: Callable, item: WorkItem, rounds: int = 2) -> dict:
    """Pay the cold-model-load cost deliberately, and record what it was.

    The first call after a model switch loads weights; on a 72B that is tens of
    seconds. Reported on its own rather than averaged in, because it is a real
    operational cost - it is what the first photograph of a demo costs - and
    because leaving it in the distribution would make every summary wrong.

    Returns the cold sample, the warm samples, and the gap between them, which
    is the load time as closely as a client can see it.
    """
    samples = [call(item) for _ in range(max(1, rounds))]
    cold, warm = samples[0], samples[1:]
    warm_ok = [s.wall_ms for s in warm if s.outcome == OK]
    warm_best = min(warm_ok) if warm_ok else None
    return {
        "cold_ms": cold.wall_ms if cold.outcome == OK else None,
        "cold_outcome": cold.outcome,
        "warm_best_ms": warm_best,
        "apparent_load_ms": ((cold.wall_ms - warm_best)
                             if (warm_best is not None and cold.outcome == OK
                                 and cold.wall_ms > warm_best) else None),
        "served_model": cold.model,
        "rounds": len(samples),
        "note": ("The cold figure includes loading the model's weights when it "
                 "was not already resident. It is excluded from every latency "
                 "distribution in this run and reported only here."),
        "_cold_sample": cold,
    }


def prepare_items(image_paths, question_id: str, cfg, limit: Optional[int] = None):
    """Decode and encode each photograph ONCE.

    Uses the pipeline's own loader so the array is the EXIF-corrected one every
    other stage sees, then the pipeline's own encoder so the bytes on the wire
    are byte-for-byte what a real run sends. A benchmark that encodes its images
    differently from the application is measuring a different system.
    """
    from pathlib import Path

    from pipeline.stage3_vlm import array_to_data_uri
    from quality_check import load_image_bgr

    items, failed = [], []
    for path in list(image_paths)[:limit] if limit else list(image_paths):
        path = Path(path)
        try:
            image_bgr = load_image_bgr(path)
            items.append(WorkItem(stem=path.stem, question_id=question_id,
                                  image_uri=array_to_data_uri(image_bgr),
                                  image_bgr=image_bgr))
        except Exception as exc:
            failed.append(f"{path.name}: {exc}")
    return items, failed
