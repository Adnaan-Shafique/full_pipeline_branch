"""Benchmarking the pipeline: latency under load, and what each VLM is worth.

Built around one hard constraint: **the GPU server holds one model at a time.**
So a four-model comparison is four separate runs with a manual switch between
them, and this package is shaped accordingly - each run writes its own file,
and a separate report step merges whatever files exist.

    tools/run_bench.py      one model, one run, one file
    tools/bench_report.py   merge the files into a comparison

    metrics.py    samples, percentiles, and the rules about what may enter them
    scenarios.py  batch / continuous / parallel arrival patterns, and the ramp
    harness.py    driving the real client, the model guard, cold-load warmup
    quality.py    contract compliance, decisiveness, cross-model agreement

`BENCHMARKS.md` is the plan and the how-to. Read it before running anything
against a shared GPU.
"""
from __future__ import annotations
