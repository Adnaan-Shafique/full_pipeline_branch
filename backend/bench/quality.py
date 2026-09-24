"""What a model's answers are worth, with no ground truth to check them against.

There are no labels for these photographs, so nothing here reports accuracy -
and it is worth being blunt about what that means: **none of these numbers say
a model is right.** They say whether it is usable, self-consistent and
comparable, which is a different question and, for choosing what to ship, very
often the decisive one.

Three measures, in descending order of how much they should influence a choice:

1. **Contract compliance.** Every question's prompt ends with a JSON contract.
   `parse_vlm_answer` is tolerant in four descending tiers, so a model that
   never emits valid JSON still produces answers - and looks fine. It is not
   fine: it is one prompt edit away from producing nothing at all, and the
   tolerance is a safety net, not a licence. A model answering mostly at
   TIER_PROSE should lose to one answering at TIER_JSON even if it is faster.

2. **Decisiveness.** How often a model says "unknown". High is not
   automatically bad - refusing to guess from an unusable photograph is
   correct - but a model that is unknown on everything has told you nothing,
   and one that is never unknown is probably guessing.

3. **Agreement.** With no labels, two models agreeing is weak evidence of
   correctness and two models disagreeing is strong evidence that at least one
   is wrong. The photographs where models split are the ones worth looking at
   by eye, which is the real output of this: a shortlist, not a score.

Mode 3 adds one more: whether the model can produce usable coordinates at all.
A model that cannot ground is a model that cannot do mode 3, whatever its
latency.

Pure stdlib.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Tiers, ordered best to worst. Imported by name rather than from stage3_vlm so
# this module stays importable with nothing installed; test_bench pins them
# against the real constants so they cannot drift apart silently.
TIER_ORDER = ("json", "keys", "prose", "give_up", "empty")
COMPLIANT_TIERS = ("json",)


@dataclass
class AnswerQuality:
    """One model's answers for one mode, summarised."""

    model: str
    mode: str = ""
    total: int = 0
    tiers: dict = field(default_factory=dict)
    answers: dict = field(default_factory=dict)
    grounding_attempted: int = 0
    grounding_usable: int = 0

    @property
    def contract_compliance(self) -> Optional[float]:
        """Share of replies that honoured the JSON contract outright."""
        if not self.total:
            return None
        return sum(self.tiers.get(t, 0) for t in COMPLIANT_TIERS) / self.total

    @property
    def unparseable(self) -> Optional[float]:
        """Share where the parser gave up entirely. These arrive as "unknown"
        and are indistinguishable from a real "unknown" in the results - which
        is the whole reason the tier is recorded."""
        if not self.total:
            return None
        return (self.tiers.get("give_up", 0) + self.tiers.get("empty", 0)) / self.total

    @property
    def unknown_rate(self) -> Optional[float]:
        if not self.total:
            return None
        return self.answers.get("unknown", 0) / self.total

    @property
    def grounding_rate(self) -> Optional[float]:
        """Of the replies asked for coordinates, how many gave usable ones."""
        if not self.grounding_attempted:
            return None
        return self.grounding_usable / self.grounding_attempted

    @property
    def verdict(self) -> str:
        """One line a human can act on. Deliberately conservative: it says when
        a model is unusable, and otherwise declines to rank."""
        if not self.total:
            return "no answers to judge"
        if self.unparseable and self.unparseable > 0.10:
            return (f"UNUSABLE as configured - {self.unparseable:.0%} of replies "
                    f"could not be parsed at all and silently became 'unknown'")
        if self.contract_compliance is not None and self.contract_compliance < 0.80:
            return (f"fragile - only {self.contract_compliance:.0%} honoured the "
                    f"JSON contract; the rest were rescued by the parser's "
                    f"tolerance and would break if the prompt changed")
        if self.unknown_rate is not None and self.unknown_rate > 0.60:
            return (f"answers {self.unknown_rate:.0%} 'unknown' - it is running, "
                    f"but it is not deciding anything")
        return "usable - honours the contract and commits to answers"


def summarise_answers(samples, model: str = "", mode: str = "") -> AnswerQuality:
    q = AnswerQuality(model=model, mode=mode)
    tiers, answers = Counter(), Counter()
    for s in samples:
        if s.outcome != "ok":
            continue         # a request that never answered says nothing here
        q.total += 1
        if s.parse_tier:
            tiers[s.parse_tier] += 1
        if s.answer:
            answers[s.answer] += 1
    q.tiers, q.answers = dict(tiers), dict(answers)
    return q


def agreement_matrix(by_model: dict) -> dict:
    """Pairwise agreement between models over the keys they both answered.

    `by_model` is {model: {key: answer}} where key identifies one
    (photograph, question, mode). Only keys BOTH models answered are compared -
    scoring a model against photographs it never saw would reward whoever ran
    the smaller set.
    """
    models = sorted(by_model)
    out = {"models": models, "pairs": [], "unanimous": 0, "split": 0, "compared": 0}

    keys_per_model = {m: set(by_model[m]) for m in models}
    shared = set.intersection(*keys_per_model.values()) if models else set()
    out["compared"] = len(shared)

    for i, a in enumerate(models):
        for b in models[i + 1:]:
            both = keys_per_model[a] & keys_per_model[b]
            same = sum(1 for k in both if by_model[a][k] == by_model[b][k])
            out["pairs"].append({
                "a": a, "b": b, "compared": len(both),
                "agreed": same,
                "agreement": (same / len(both)) if both else None})

    for key in shared:
        values = {by_model[m][key] for m in models}
        if len(values) == 1:
            out["unanimous"] += 1
        else:
            out["split"] += 1
    return out


def disagreements(by_model: dict, limit: int = 25) -> list:
    """The specific items the models split on - the shortlist to eyeball.

    This, not any percentage, is the useful output of a label-free comparison.
    """
    models = sorted(by_model)
    if len(models) < 2:
        return []
    shared = set.intersection(*(set(by_model[m]) for m in models))
    rows = []
    for key in sorted(shared):
        answers = {m: by_model[m][key] for m in models}
        if len(set(answers.values())) > 1:
            rows.append({"key": key, "answers": answers})
    return rows[:limit]


# The key separator is NOT a pipe. Keys are printed verbatim into a markdown
# table in the report, and a pipe inside a cell silently splits it into more
# columns than the header has - which renders as a mangled table rather than an
# error, so it would have shipped.
KEY_SEP = " :: "


def answer_key(stem: str, question_id: str, mode: str = "") -> str:
    return KEY_SEP.join((stem, question_id, mode or "single"))


def answers_by_key(samples) -> dict:
    """{"stem :: question :: mode": answer} for the successful samples.

    One entry per (photograph, question, mode), which is the unit two models
    can be compared on. A sample that did not answer contributes nothing rather
    than a blank - a model should not be scored on requests that never landed.
    """
    out = {}
    for s in samples:
        if s.outcome != "ok" or not s.answer:
            continue
        out[answer_key(s.stem, s.question_id, getattr(s, "mode", ""))] = s.answer
    return out


def per_stage_breakdown(records) -> dict:
    """Where the seconds go for one photograph, across the classical stages.

    Answers "is the pipeline fast enough per photograph" by naming the stage to
    attack. Reads the timings the stages now record; a stage that reports 0.0
    was not measured and is listed as such rather than as instant.
    """
    buckets = defaultdict(list)
    for record in records:
        q = getattr(record, "quality", None)
        d = getattr(record, "detection", None)
        o = getattr(record, "ocr", None)
        v = getattr(record, "vlm", None)
        if q is not None and getattr(q, "elapsed_ms", 0):
            buckets["quality"].append(q.elapsed_ms)
        if d is not None and getattr(d, "elapsed_ms", 0):
            buckets["detection"].append(d.elapsed_ms)
        if o is not None and getattr(o, "elapsed_ms", 0):
            buckets["ocr"].append(o.elapsed_ms)
        if v is not None and getattr(v, "elapsed_s", 0):
            buckets["vlm"].append(v.elapsed_s * 1000)

    from .metrics import LatencyStats

    out = {stage: LatencyStats.from_values(values).__dict__
           for stage, values in buckets.items()}
    totals = [sum(v[i] for v in buckets.values() if i < len(v))
              for i in range(max((len(v) for v in buckets.values()), default=0))]
    out["_measured_stages"] = sorted(buckets)
    out["_photographs"] = max((len(v) for v in buckets.values()), default=0)
    if totals:
        out["_total_per_photo_ms"] = LatencyStats.from_values(totals).__dict__
    return out
