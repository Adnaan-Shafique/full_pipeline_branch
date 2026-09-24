#!/usr/bin/env python3
"""Merge the per-model benchmark files into one comparison.

    python tools/bench_report.py bench_runs/20260924

Reads every <model>.json in the folder - however many models have been run so
far - and writes report.md and comparison.csv beside them.

The sections are ordered by the decisions they serve:

    1. Which model to ship      compliance first, then speed. A model that
                                cannot honour the JSON contract is not a
                                candidate however fast it is, so that column
                                comes before the milliseconds.
    2. How much load it takes   the saturation point from the ramp.
    3. Per photograph           where the seconds go across the stages.
    4. Where models disagree    the shortlist to look at by eye, which is the
                                real product of a comparison with no labels.

It refuses to rank models on numbers that cannot carry a ranking: mock runs,
runs whose model guard failed, and models measured under different settings
are called out rather than tabulated beside each other.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def fmt(value, suffix="", digits=0):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def pct(value):
    return "—" if value is None else f"{value:.0%}"


def load(folder: Path):
    reports = {}
    for path in sorted(folder.glob("*.json")):
        if path.name in ("report.json",):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"  skipping {path.name}: {exc}")
            continue
        if "model" in data:
            reports[data["model"]] = data
    return reports


def comparability_warnings(reports: dict) -> list:
    """Everything that makes a side-by-side table misleading.

    Models benchmarked over different photographs, different questions or
    different sampling settings are not comparable, and a table does not show
    that unless it is said out loud.
    """
    out = []
    for model, r in sorted(reports.items()):
        if r.get("is_mock"):
            out.append(f"{model}: MOCK RUN - no model was called. Its timings "
                       f"measure the harness and are excluded from the ranking.")
        if not (r.get("model_check") or {}).get("ok", True):
            out.append(f"{model}: the model guard FAILED "
                       f"({(r.get('model_check') or {}).get('detail', '')}). "
                       f"Treat this file as void.")
        for w in r.get("warnings", []):
            out.append(f"{model}: {w}")

    def distinct(key):
        return {model: json.dumps(r.get(key), sort_keys=True, default=str)
                for model, r in reports.items()}

    for key, label in (("question", "question"), ("sampling", "sampling settings"),
                       ("photographs", "photograph set"),
                       ("transport", "transport")):
        values = distinct(key)
        if len(set(values.values())) > 1:
            out.append(f"models were run with DIFFERENT {label} - they are not "
                       f"directly comparable: "
                       + ", ".join(f"{m}" for m in sorted(values)))
    return out


def rankable(reports: dict) -> dict:
    """Only the runs whose numbers may be compared."""
    return {m: r for m, r in reports.items()
            if not r.get("is_mock") and (r.get("model_check") or {}).get("ok", True)}


def scenario_stats(report: dict, scenario: str) -> dict:
    block = (report.get("scenarios") or {}).get(scenario) or {}
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from bench import quality

    folder = args.folder
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 1
    reports = load(folder)
    if not reports:
        print(f"no model result files in {folder}")
        return 1

    out_dir = args.out or folder
    lines = [f"# Benchmark — {folder.name}", ""]
    lines.append(f"{len(reports)} model(s): {', '.join(sorted(reports))}")
    lines.append("")

    problems = comparability_warnings(reports)
    if problems:
        lines += ["## Read this first", ""]
        lines += [f"- {p}" for p in problems]
        lines.append("")

    ranked = rankable(reports)
    if not ranked:
        lines += ["No file in this folder carries numbers that may be compared.",
                  ""]

    # ── 1. Which model to ship ───────────────────────────────────────────────
    lines += ["## 1. Which model to ship", "",
              "Compliance before speed: a model that cannot honour the JSON "
              "contract is not a candidate however fast it is. `unparseable` is "
              "the share of replies that silently became `unknown`.", "",
              "| model | verdict | contract | unparseable | unknown | batch p50 | "
              "batch p95 | cold load |", "|---|---|---|---|---|---|---|---|"]
    for model, r in sorted(ranked.items()):
        aq = r.get("answer_quality") or {}
        batch = scenario_stats(r, "batch").get("latency_client_ms") or {}
        warm = r.get("warmup") or {}
        lines.append(
            f"| `{model}` | {aq.get('verdict', '—')} "
            f"| {pct(aq.get('contract_compliance'))} "
            f"| {pct(aq.get('unparseable'))} "
            f"| {pct(aq.get('unknown_rate'))} "
            f"| {fmt(batch.get('p50_ms'), ' ms')} "
            f"| {fmt(batch.get('p95_ms'), ' ms')} "
            f"| {fmt(warm.get('apparent_load_ms'), ' ms')} |")
    lines.append("")
    lines.append("Cold load is what the FIRST photograph of a demo costs after a "
                 "model switch. It is excluded from every latency figure above.")
    lines.append("")

    # ── 2. Load ──────────────────────────────────────────────────────────────
    lines += ["## 2. How much load it survives", "",
              "| model | parallel p50 | parallel p95 | throughput | 503s | "
              "highest healthy concurrency |", "|---|---|---|---|---|---|"]
    for model, r in sorted(ranked.items()):
        par = scenario_stats(r, "parallel")
        lat = par.get("latency_client_ms") or {}
        counts = par.get("counts") or {}
        ramp = scenario_stats(r, "ramp").get("saturation") or {}
        lines.append(
            f"| `{model}` | {fmt(lat.get('p50_ms'), ' ms')} "
            f"| {fmt(lat.get('p95_ms'), ' ms')} "
            f"| {fmt(par.get('throughput_rps'), ' rps', 2)} "
            f"| {counts.get('saturated', 0)} "
            f"| {fmt(ramp.get('highest_healthy_concurrency'))} |")
    lines.append("")
    for model, r in sorted(ranked.items()):
        ramp = scenario_stats(r, "ramp").get("saturation") or {}
        if ramp.get("reason_it_stopped"):
            lines.append(f"- `{model}`: {ramp['reason_it_stopped']}")
    lines.append("")

    # ── 3. Per photograph ────────────────────────────────────────────────────
    lines += ["## 3. Where the seconds go, per photograph", "",
              "Only present for runs made with `--modes`. A stage with no "
              "figure was not measured, which is not the same as instant.", ""]
    any_modes = False
    for model, r in sorted(ranked.items()):
        modes = r.get("modes") or {}
        if not modes:
            continue
        any_modes = True
        lines.append(f"### `{model}`")
        lines.append("")
        lines.append("| mode | photos | answers | stages measured | total p50/photo |")
        lines.append("|---|---|---|---|---|")
        for mode, block in modes.items():
            stages = block.get("per_stage_ms") or {}
            total = stages.get("_total_per_photo_ms") or {}
            lines.append(
                f"| {mode} | {block.get('photographs', 0)} "
                f"| {block.get('answers', {})} "
                f"| {', '.join(stages.get('_measured_stages', [])) or '—'} "
                f"| {fmt(total.get('p50_ms'), ' ms')} |")
        lines.append("")
    if not any_modes:
        lines += ["No run in this folder used `--modes`.", ""]

    # ── 4. Disagreement ──────────────────────────────────────────────────────
    by_model = {m: r.get("answers_by_key") or {} for m, r in ranked.items()}
    by_model = {m: v for m, v in by_model.items() if v}
    lines += ["## 4. Where the models disagree", ""]
    if len(by_model) < 2:
        lines += ["Fewer than two comparable models have answers - nothing to "
                  "compare yet. Run the rest and re-run this report.", ""]
    else:
        matrix = quality.agreement_matrix(by_model)
        lines.append(f"Compared over {matrix['compared']} items answered by every "
                     f"model: **{matrix['unanimous']} unanimous**, "
                     f"**{matrix['split']} split**.")
        lines.append("")
        lines.append("| pair | compared | agreement |")
        lines.append("|---|---|---|")
        for pair in matrix["pairs"]:
            lines.append(f"| `{pair['a']}` vs `{pair['b']}` | {pair['compared']} "
                         f"| {pct(pair['agreement'])} |")
        lines.append("")
        splits = quality.disagreements(by_model)
        if splits:
            lines.append("With no ground truth, agreement is weak evidence of "
                         "correctness but disagreement is strong evidence that "
                         "someone is wrong. These are the ones to look at:")
            lines.append("")
            lines.append("| item | " + " | ".join(f"`{m}`" for m in sorted(by_model)) + " |")
            lines.append("|---" * (len(by_model) + 1) + "|")
            for row in splits:
                answers = row["answers"]
                lines.append(f"| {row['key']} | "
                             + " | ".join(answers.get(m, "—") for m in sorted(by_model))
                             + " |")
            lines.append("")

    report_md = "\n".join(lines)
    (out_dir / "report.md").write_text(report_md)

    with open(out_dir / "comparison.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "is_mock", "guard_ok", "verdict",
                         "contract_compliance", "unparseable", "unknown_rate",
                         "batch_p50_ms", "batch_p95_ms", "parallel_p50_ms",
                         "parallel_p95_ms", "throughput_rps", "saturated_503",
                         "highest_healthy_concurrency", "cold_load_ms"])
        for model, r in sorted(reports.items()):
            aq = r.get("answer_quality") or {}
            batch = scenario_stats(r, "batch").get("latency_client_ms") or {}
            par = scenario_stats(r, "parallel")
            plat = par.get("latency_client_ms") or {}
            ramp = scenario_stats(r, "ramp").get("saturation") or {}
            writer.writerow([
                model, r.get("is_mock", False),
                (r.get("model_check") or {}).get("ok", True),
                aq.get("verdict", ""), aq.get("contract_compliance"),
                aq.get("unparseable"), aq.get("unknown_rate"),
                batch.get("p50_ms"), batch.get("p95_ms"),
                plat.get("p50_ms"), plat.get("p95_ms"),
                par.get("throughput_rps"),
                (par.get("counts") or {}).get("saturated", 0),
                ramp.get("highest_healthy_concurrency"),
                (r.get("warmup") or {}).get("apparent_load_ms")])

    print(report_md)
    print(f"\nwrote {out_dir / 'report.md'} and {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
