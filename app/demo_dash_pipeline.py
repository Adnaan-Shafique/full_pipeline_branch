#!/usr/bin/env python3
"""Field Ops integrated demo - DOMAIN, QUESTION, ONE PHOTOGRAPH, THREE MODES.

    python app/demo_dash_pipeline.py    ->  http://<host>:7873

Pick a domain, pick a question from that domain, drop a photograph, and see all
three modes answer it side by side:

    1  Quality gate -> Detector -> OCR -> Model
    2  (Quality OR Detector) -> OCR -> Model
    3  Everything by the model

Stage 2b, OCR, runs in modes 1 and 2 for the questions whose YAML sets
`ocr.enabled: true`, and NEVER in mode 3. That asymmetry is the point of this
screen: on a device reading, mode 3's answer is the model reading a
seven-segment display unaided, next to two modes that had PP-OCRv6's
transcription handed to them. Where they disagree, the OCR panel and the
model's reasoning say which one moved.

Siblings, all runnable at once on their own ports:
    app/demo_dash.py         7870  annotation detector, FROZEN (see FROZEN.md)
    app/demo_dash_yolox.py   7871  single-mode YOLOX
    app/demo_dash_modes.py   7872  three modes over a FOLDER of photographs
    app/demo_dash_pipeline.py 7873  this one

7872 is not superseded: it is the batch screen, and running twenty photographs
to find the two that disagree is a different job from examining one. This screen
is the per-photograph one, and it is the one that grew the domain selector and
the OCR panel.

The card renderers are IMPORTED from demo_dash and demo_dash_modes rather than
copied. A real detection and a human annotation must look identical on screen or
the audience learns to read the styling instead of the provenance banner, and a
copy would drift.

Questions, domains and detector classes all come from config/ via the registry -
see app/pipeline/registry.py. Nothing about a question is hard-coded here, which
is why the domain dropdown and the per-question panel below can be written once
and stay correct as questions are added.
"""
from __future__ import annotations

import base64
import os
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from dash import Dash, Input, Output, State, dcc, html, no_update  # noqa: E402

from pipeline.config import (SEND_FULL, SEND_FULL_CROP,  # noqa: E402
                             TRANSPORT_DIRECT, TRANSPORT_PROXY, VLM_MODE_LIVE,
                             VLM_MODE_MOCK, default_config)
from pipeline.modes import (DEFAULT_OVERLAY, MODE_ORDER,  # noqa: E402
                            MODE_VLM_ONLY, MODES, OVERLAY_BOX,
                            OVERLAY_BOX_LABEL, OVERLAY_CHOICES, OVERLAY_OFF,
                            agreement, compare_rows, run_all_modes)
from pipeline.orchestrator import (IMAGE_EXTENSIONS, new_run_id,  # noqa: E402
                                   sort_for_display)
from pipeline.prompts import default_store  # noqa: E402
from pipeline import questions as pq  # noqa: E402
from pipeline import stage2b_ocr  # noqa: E402
from pipeline.questions import (LEG_LABELS, LEGS, build_ocr_block,  # noqa: E402
                                default_leg_system, get_question,
                                render_leg_user, render_user_prompt,
                                select_relevant)
from pipeline.schemas import STOPPED_QUALITY  # noqa: E402

# Shared renderers. vlm_column is reused verbatim; quality_column and
# detection_column come from demo_dash_modes, which already handles mode 3's
# two special cases (a quality verdict with no numeric score, and presence
# reported in words rather than as boxes).
from demo_dash import FONTS, image_or_placeholder, tile, vlm_column  # noqa: E402
from demo_dash_modes import (claimed_box_lines, detection_column,  # noqa: E402
                             quality_column)

UPLOAD_ACCEPT = ",".join(["image/*"] + sorted(IMAGE_EXTENSIONS))


# ─────────────────────────────── Config ──────────────────────────────────────

def _cfg_from_controls(gpu_url, vlm_model, vlm_mode, send_mode, threshold,
                       transport, api_key, flags, checkpoint):
    flags = flags or []
    return default_config(
        # An explicit checkpoint turns the trained detector on. Without one the
        # annotation-sidecar backend runs and labels itself a stub - which is
        # the honest state for every Infra question today, none of which has
        # trained weights. yolox_class_names is NOT passed: it comes from
        # config/classes.yaml through default_config().
        use_model=bool((checkpoint or "").strip()),
        yolox_checkpoint=(checkpoint or "").strip() or None,
        gpu_url=(gpu_url or "").strip(),
        vlm_transport=(transport or TRANSPORT_DIRECT),
        vlm_api_key=(api_key or "").strip(),
        vlm_model=(vlm_model or "qwen3-vl").strip(),
        vlm_mode=vlm_mode, vlm_send_mode=send_mode,
        quality_threshold=float(threshold) if threshold is not None else 65.0,
        ignore_resolution="ignore_res" in flags,
        run_downstream_on_fail="run_on_fail" in flags,
        # Off by ABSENCE, not by default: the checkbox ships ticked, so leaving
        # the controls alone gets grounding. Unticking it returns mode 3 to
        # presence-in-words, which is what to do if the model's boxes turn out
        # to be noise on a given photo set.
        vlm_grounding="no_grounding" not in flags,
    )


def _domain_options():
    return [{"label": label, "value": did} for label, did in pq.REGISTRY.domain_choices()]


def _question_options(domain_id):
    return [{"label": q.label, "value": q.id}
            for q in pq.REGISTRY.questions_in(domain_id)]


def _first_domain():
    options = _domain_options()
    return options[0]["value"] if options else None


def _first_question(domain_id):
    options = _question_options(domain_id)
    return options[0]["value"] if options else None


# ─────────────────────────────── Controls ────────────────────────────────────

def controls():
    cfg = default_config()
    domain = _first_domain()
    return html.Div([
        html.Div([
            html.Div("1 · Domain", className="card-title"),
            dcc.Dropdown(id="domain", options=_domain_options(), value=domain,
                         clearable=False, className="opt"),
            html.Div(id="domain-blurb", className="field-help"),

            html.Div("2 · Question", className="card-title",
                     style={"marginTop": "18px"}),
            dcc.Dropdown(id="question", options=_question_options(domain),
                         value=_first_question(domain), clearable=False,
                         className="opt"),
            html.Div(id="question-detail"),

            html.Div([
                html.Button("Reload config/", id="reload-config",
                            className="btn btn-ghost"),
            ], className="btn-row", style={"marginTop": "14px"}),
            html.Div("Re-reads config/domains.yaml, classes.yaml and "
                     "questions/*.yaml without restarting. Add a question, press "
                     "this, and it is in the dropdown.", className="field-help"),
            html.Div(id="reload-status", className="field-help"),
        ], className="card"),

        html.Div([
            html.Div("3 · Photograph", className="card-title"),
            dcc.Upload(id="uploads", accept=UPLOAD_ACCEPT, multiple=True,
                       className="dropzone", children=html.Div([
                           html.Div("Drop a photograph here"),
                           html.Div("or click to choose · several are fine, each "
                                    "gets its own three-mode comparison",
                                    className="field-help")])),
            html.Div(id="upload-note", className="dz-list"),
            html.Div([html.Button("Clear", id="clear-uploads",
                                  className="btn btn-ghost")], className="btn-row"),
        ], className="card"),

        html.Div([
            html.Div("Model", className="card-title"),
            html.Div([html.Label("Server URL", className="field"),
                      dcc.Input(id="gpu-url", value=cfg.gpu_url, type="text",
                                className="opt mono")]),
            html.Div([html.Label("Transport", className="field"),
                      dcc.RadioItems(
                          id="transport", className="opt",
                          options=[{"label": " direct", "value": TRANSPORT_DIRECT},
                                   {"label": " proxy", "value": TRANSPORT_PROXY}],
                          value=cfg.vlm_transport)]),
            html.Div(id="transport-help", className="field-help"),
            html.Div([html.Label("API key (proxy only)", className="field"),
                      dcc.Input(id="api-key", value=cfg.vlm_api_key, type="password",
                                className="opt mono")]),
            html.Div([html.Label("Model", className="field"),
                      dcc.Input(id="vlm-model", value=cfg.vlm_model, type="text",
                                className="opt mono")]),
            html.Div([html.Label("Mode", className="field"),
                      dcc.RadioItems(
                          id="vlm-mode", className="opt",
                          options=[{"label": " live", "value": VLM_MODE_LIVE},
                                   {"label": " mock", "value": VLM_MODE_MOCK}],
                          value=cfg.vlm_mode)]),
            html.Div([html.Label("Send", className="field"),
                      dcc.RadioItems(
                          id="send-mode", className="opt",
                          options=[{"label": " full image", "value": SEND_FULL},
                                   {"label": " full + crop", "value": SEND_FULL_CROP}],
                          value=cfg.vlm_send_mode)]),
        ], className="card"),

        html.Div([
            html.Div("Detector & quality", className="card-title"),
            html.Div([html.Label("YOLOX checkpoint (blank = annotation sidecars)",
                                 className="field"),
                      dcc.Input(id="checkpoint", value="", type="text",
                                placeholder="models/best_ckpt.pth",
                                className="opt mono")]),
            html.Div("Only two classes have trained weights. Every Infra question "
                     "runs on annotation sidecars and says so on the card.",
                     className="field-help"),
            html.Div([html.Label("Quality threshold", className="field"),
                      dcc.Input(id="threshold", value=65, type="number",
                                min=0, max=100, className="opt")]),
            dcc.Checklist(id="flags", className="opt", value=[], options=[
                {"label": " ignore the resolution floor", "value": "ignore_res"},
                {"label": " run downstream legs on a quality FAIL",
                 "value": "run_on_fail"},
                {"label": " mode 3: presence in words only, no claimed boxes",
                 "value": "no_grounding"}]),
            html.Div("Mode 3 asks the model where things are and draws what "
                     "comes back, dashed, scored against the detector's own box "
                     "where one exists. Tick the last option to turn that off.",
                     className="field-help"),
        ], className="card"),

        html.Div([
            html.Button("Run all three modes", id="run", className="btn btn-primary"),
            html.Button("Check host", id="check", className="btn btn-ghost"),
        ], className="btn-row"),
    ])


# ─────────────────────────────── The OCR panel ───────────────────────────────

def ocr_column(record, question):
    """Stage 2b's panel. There are four different things it has to say, and
    conflating any two of them misleads:

      · the question does not use OCR          -> a muted "not part of this
                                                   question", no alarm
      · mode 3                                 -> "the model read it itself",
                                                   which is a property of the
                                                   mode, not a failure
      · OCR could not run                      -> a warning naming the missing
                                                   piece, actionable before the
                                                   next photograph
      · OCR ran and read nothing               -> a normal result for a
                                                   photograph with no legible
                                                   text in it
    """
    o = record.ocr
    header = html.H4("2b · Text read (OCR)")

    if o is None:
        return html.Div([header, html.Div("Not run — an earlier stage stopped "
                                          "this image.", className="rc-none")],
                        className="rc-col")

    if o.error:
        return html.Div([
            header,
            html.Div(html.Span(o.headline, className="pill pill-err"),
                     className="rc-line"),
            html.Div([html.B("The OCR stage could not run. "),
                      "The question was still answered, from the image alone — "
                      "the model simply received no OCR text."],
                     className="banner banner-warn"),
        ], className="rc-col")

    if not o.ran:
        # Skipped. The note distinguishes "this question has no OCR stage" from
        # "mode 3 deliberately runs none", and both are written by whoever
        # skipped it rather than guessed at here.
        note = o.note or "not part of this question"
        children = [header, html.Div(note, className="rc-none")]
        if question.ocr.enabled:
            children.append(html.Div(
                "Modes 1 and 2 ran PP-OCRv6 on this question. Compare their "
                "answer with this one to see how well the model reads the text "
                "unaided.", className="rc-muted"))
        return html.Div(children, className="rc-col")

    children = [header]
    if o.annotated_path:
        children.append(image_or_placeholder(
            o.annotated_path, "The OCR overlay could not be rendered"))
    children.append(html.Div(html.Span(o.headline, className="pill pill-stub"),
                             className="rc-line"))

    if o.lines:
        children.append(html.Ul(
            [html.Li([html.Span(f'"{ln.text}"', className="mono"),
                      html.Span(f" — {ln.text_confidence:.2f}"
                                + (f" · from {ln.from_label}" if ln.from_label else ""),
                                className="rc-muted")])
             for ln in o.lines], className="rc-ul"))
    else:
        children.append(html.Div("No legible text.", className="rc-muted"))

    if o.numeric:
        # Deliberately NOT styled as a yes/no chip. The rule cannot see a range
        # multiplier, tell a set-point from a measurement, or tell °F from °C,
        # and a verdict-shaped badge here would be read as the answer. The
        # matched token is shown so an operator can see WHICH number it took -
        # which is what makes "SET 22 ACT 38" visible rather than silent.
        kind = "banner-info" if o.numeric["passes"] else "banner-warn"
        children.append(html.Div([
            html.B("Threshold check (evidence, not the answer): "),
            o.numeric["sentence"],
            html.Span(f'  Matched the text "{o.numeric["matched"]}".',
                      className="rc-muted"),
        ], className=f"banner {kind}"))
    elif question.ocr.numeric is not None:
        children.append(html.Div(
            f"No {question.ocr.numeric.label.lower()} reading could be matched "
            f"in the text above.", className="rc-muted"))

    if o.note:
        children.append(html.Div(o.note, className="rc-muted"))
    children.append(html.Div(
        "Advisory only. The text above is given to the model behind a hedge "
        "telling it to prefer the image.", className="rc-muted"))
    return html.Div(children, className="rc-col")


# ─────────────────────────────── Result rendering ────────────────────────────

def evidence_column(record, question, mode_id):
    """The answer leg's claimed evidence region — "what did you actually look
    at to decide?"

    Distinct from the presence box on purpose. "Where the subject is" and "what
    I based my answer on" are different claims, and on the OCR questions the
    difference is the interesting one: mode 3 pointing at the display it read
    can be put straight next to the box PP-OCRv6 read from in modes 1 and 2.
    """
    boxes = getattr(record.vlm, "evidence_boxes", []) if record.vlm else []
    note = record.extra.get("answer_grounding_note", "")
    if not boxes and not note:
        return None
    children = [html.H4("3b · What the model looked at")]
    if boxes:
        children.extend(claimed_box_lines(boxes))
        children.append(html.Div(
            "Drawn dashed on the photograph in panel 2, alongside the presence "
            "claim.", className="rc-muted"))
    if note:
        children.append(html.Div(note, className="banner banner-warn"))
    return html.Div(children, className="rc-col")


def mode_column(record, question, mode_id):
    """One mode's whole verdict, as a vertical stack - so three of them sit
    side by side and the row reads across."""
    answer = record.vlm.answer if record.vlm else None
    chip = {"yes": "chip-yes", "no": "chip-no"}.get(answer, "chip-unknown")
    head = [
        html.Div(MODES[mode_id]["short"], className="rc-name"),
        html.Div(html.Span((answer or "stopped").upper(), className=f"chip {chip}",
                           title=question.answer_semantics)),
    ]
    if record.vlm and record.vlm.is_mock:
        head.append(html.Div("MOCK — no model looked", className="banner banner-warn"))
    body = [html.Div(record.vlm.reasoning if record.vlm else
                     "Stopped before the model was asked.", className="rc-reason")]
    if record.stopped_at == STOPPED_QUALITY:
        body.append(html.Div("Stopped at the quality gate.",
                             className="banner banner-stop"))
    gate = record.extra.get("gate")
    if gate:
        body.append(html.Div(f"Gate: {gate}", className="banner banner-info"))
    claimed = ((getattr(record.detection, "claimed_boxes", []) if record.detection else [])
               + (getattr(record.vlm, "evidence_boxes", []) if record.vlm else []))
    if claimed:
        scored = [c for c in claimed if c.iou is not None]
        summary = (f"best IoU {max(c.iou for c in scored):.2f} vs detector"
                   if scored else "not comparable — no detector box")
        body.append(html.Div(
            f"Model pointed at {len(claimed)} region"
            f"{'s' if len(claimed) != 1 else ''} — {summary}",
            className="rc-muted"))
    if record.ocr is not None and record.ocr.ran:
        body.append(html.Div(f"OCR read: {record.ocr.text or '—'}",
                             className="rc-muted mono"))
    elif mode_id == MODE_VLM_ONLY and question.ocr.enabled:
        body.append(html.Div("No OCR — the model read the image itself.",
                             className="rc-muted"))
    return html.Div(head + body, className="rc-col")


def side_by_side(stem, per_mode, question):
    """The headline block: one photograph, three modes across."""
    first = per_mode[MODE_ORDER[0]]
    answers = {m: (per_mode[m].vlm.answer if per_mode[m].vlm else None)
               for m in MODE_ORDER}
    distinct = {a for a in answers.values() if a}
    if len(distinct) <= 1:
        verdict = html.Div("All three modes agree.", className="banner banner-info")
    else:
        verdict = html.Div(
            [html.B("The three modes disagree — "),
             ", ".join(f"{MODES[m]['short']}: {(answers[m] or 'stopped').upper()}"
                       for m in MODE_ORDER),
             ". This is the photograph worth talking about."],
            className="banner banner-warn")
    return html.Div([
        html.Div([html.Span(first.filename, className="rc-name"),
                  html.Span(stem, className="rc-sub")], className="rc-head"),
        verdict,
        html.Div([mode_column(per_mode[m], question, m) for m in MODE_ORDER],
                 className="rc-cols"),
    ], className="rc")


def detail_card(record, question, mode_id, overlay=DEFAULT_OVERLAY):
    """One mode's four stages in full: quality, detection, OCR, answer."""
    children = [html.Div([html.Span(record.filename, className="rc-name"),
                          html.Span(MODES[mode_id]["label"], className="rc-sub")],
                         className="rc-head")]
    if record.stopped_at == STOPPED_QUALITY:
        children.append(html.Div(
            "Stopped at the quality gate — the downstream legs did not run.",
            className="banner banner-stop"))
    gate = record.extra.get("gate")
    if gate:
        children.append(html.Div(f"Gate: {gate}", className="banner banner-info"))
    panels = [quality_column(record), detection_column(record, overlay),
              ocr_column(record, question), vlm_column(record, question)]
    evidence = evidence_column(record, question, mode_id)
    if evidence is not None:
        panels.append(evidence)
    children.append(html.Div(panels, className="rc-cols"))
    return html.Div(children, className="rc")


def _by_stem(results):
    """{stem: {mode: record}} in the order the photographs were run."""
    order, out = [], {}
    for mode in MODE_ORDER:
        for record in results.get(mode, []):
            if record.stem not in out:
                out[record.stem] = {}
                order.append(record.stem)
            out[record.stem][mode] = record
    return [(stem, out[stem]) for stem in order
            if len(out[stem]) == len(MODE_ORDER)]


def question_detail(question_id):
    """What this question looks for, what has weights, and whether OCR runs.

    The untrained-class line is the important one: "the detector found nothing"
    and "nothing was ever trained to find this" look identical on a card, and
    only one of them is evidence of absence.
    """
    try:
        question = get_question(question_id)
    except KeyError:
        return html.Div("No question selected.", className="field-help")

    children = [html.Div(question.answer_semantics, className="field-help")]

    classes = [pq.REGISTRY.classes[cid] for cid in question.class_ids
               if cid in pq.REGISTRY.classes]
    if classes:
        trained = [c.name for c in classes if c.trained]
        untrained = [c.name for c in classes if not c.trained]
        children.append(html.Div(
            ["Looks for: ", html.Span(", ".join(c.name for c in classes),
                                      className="mono")], className="rc-muted"))
        if untrained:
            children.append(html.Div(
                [html.B("No trained weights"), " for ",
                 html.Span(", ".join(untrained), className="mono"),
                 ". Stage 2 reads annotation sidecars for these and labels "
                 "itself a stub — an empty box list here is not evidence the "
                 "object is absent."],
                className="banner banner-warn"))
        if trained:
            children.append(html.Div(f"Trained: {', '.join(trained)}",
                                     className="rc-muted"))

    if question.ocr.enabled:
        bits = [html.B("OCR runs for this question"),
                f" (scope: {question.ocr.scope}) in modes 1 and 2. Mode 3 reads "
                f"the text with the model instead — that comparison is the point."]
        if question.ocr.numeric is not None:
            rule = question.ocr.numeric
            bits.append(html.Div(
                f"Threshold checked as evidence: {rule.label} "
                f"{rule.comparator} {rule.limit:g} {rule.unit}.",
                className="rc-muted"))
        children.append(html.Div(bits, className="banner banner-info"))
    return html.Div(children)


# ─────────────────────────────── App ─────────────────────────────────────────

def layout():
    domain = _first_domain()
    return html.Div([
        html.Div([
            html.Div([
                html.H1([html.Span("Field Ops", className="accent"),
                         " pipeline — one photograph, three modes"]),
                html.P("Pick a domain and a question, drop a photograph, and "
                       "watch a hard quality gate, a quality-or-detector gate "
                       "and the vision model alone each answer it."),
                html.Div(className="rule"),
            ], className="masthead"),

            html.Div(id="status", className="statusbar", children=[
                html.Span(className="dot"),
                html.Span("Press Check host to see what is available before you run."),
            ]),

            html.Div([
                html.Div(controls()),
                html.Div([
                    dcc.Tabs(id="tabs", value="compare", parent_className="tabs-bar",
                             className="tabs-bar", children=[
                        dcc.Tab(label="Three modes", value="compare", className="tab",
                                selected_className="tab--selected"),
                        dcc.Tab(label="Stage detail", value="detail", className="tab",
                                selected_className="tab--selected"),
                        dcc.Tab(label="Prompts sent", value="prompts", className="tab",
                                selected_className="tab--selected"),
                    ]),
                    dcc.RadioItems(
                        id="mode", className="opt modes",
                        options=[{"label": MODES[m]["label"], "value": m}
                                 for m in MODE_ORDER],
                        value=MODE_ORDER[0], style={"marginTop": "12px"}),
                    # A VIEW control, not a run control. Every variant was
                    # rendered during the run, so switching costs a re-render
                    # and never a model call - which is why it sits up here
                    # with the tabs rather than in the sidebar with the
                    # settings that do force a re-run.
                    html.Div([
                        html.Span("Mode 3 overlay", className="field"),
                        dcc.RadioItems(
                            id="overlay", className="opt",
                            options=[
                                {"label": " box + label", "value": OVERLAY_BOX_LABEL},
                                {"label": " box only", "value": OVERLAY_BOX},
                                {"label": " off", "value": OVERLAY_OFF}],
                            value=DEFAULT_OVERLAY,
                            style={"display": "inline-block"}),
                        html.Div("Redraws from results already computed — it "
                                 "never re-runs the pipeline. Use “box only” "
                                 "when the caption covers the object.",
                                 className="field-help"),
                    ], style={"marginTop": "10px"}),
                    dcc.Loading(html.Div(id="panel", style={"marginTop": "20px"}),
                                type="dot", color="#EE3B2F"),
                ]),
            ], className="cols"),
            dcc.Store(id="staged", data={"dir": None, "files": []}),
        ], className="shell"),
    ])


app = Dash(__name__, external_stylesheets=FONTS,
           title="Field Ops — Domain · Question · 3 Modes", update_title="Running…",
           assets_folder=str(APP_DIR / "assets"), suppress_callback_exceptions=True)
app.layout = layout()
server = app.server

# The last run, so switching tab or mode re-reads rather than re-runs.
_STATE: dict = {"results": {}, "question_id": None, "cfg": None, "paths": []}
PROMPTS = default_store()


# ─────────────────────────────── Callbacks ───────────────────────────────────

@app.callback(Output("question", "options"), Output("question", "value"),
              Output("domain-blurb", "children"), Input("domain", "value"))
def on_domain(domain_id):
    """Switching domain replaces the question list. The value is reset to that
    domain's first question rather than left dangling: a stale id from the
    previous domain looks selected but resolves to nothing on Run."""
    options = _question_options(domain_id)
    blurb = ""
    if domain_id in pq.REGISTRY.domains:
        blurb = pq.REGISTRY.domains[domain_id].blurb
    return options, (options[0]["value"] if options else None), blurb


@app.callback(Output("question-detail", "children"), Input("question", "value"))
def on_question(question_id):
    return question_detail(question_id)


@app.callback(Output("domain", "options"), Output("domain", "value"),
              Output("reload-status", "children"),
              Input("reload-config", "n_clicks"), prevent_initial_call=True)
def on_reload(_clicks):
    """Re-read config/ without a restart, so a question can be added mid-session.

    Warnings are shown rather than swallowed: a question that failed to load is
    simply absent from the dropdown, and an operator who does not know why will
    assume the app is broken rather than the YAML.
    """
    pq.refresh()
    options = _domain_options()
    registry = pq.REGISTRY
    summary = (f"{len(registry.questions)} questions across "
               f"{len(registry.ordered_domains())} domains, from {registry.source}")
    if registry.warnings:
        return options, (options[0]["value"] if options else None), html.Div(
            [html.B(f"{summary}, with {len(registry.warnings)} problem(s):")]
            + [html.Div(f"· {w}") for w in registry.warnings],
            className="banner banner-warn")
    return options, (options[0]["value"] if options else None), f"Reloaded — {summary}."


@app.callback(Output("transport-help", "children"), Input("transport", "value"))
def on_transport(transport):
    if transport == TRANSPORT_PROXY:
        return ("POST /v1/infer with an X-API-Key header. Point the URL at the "
                "proxy, not the GPU box.")
    return ("POST /infer, no auth. Pointing this at the proxy is a 404 that "
            "reads as a dead server.")


def _stage_uploads(contents, filenames, staged):
    """Write uploads to disk on arrival, so a second drop adds to the first and
    the browser is not left holding base64 while photographs are collected."""
    directory = staged.get("dir") or str(
        default_config().runs_dir / "uploads" / new_run_id())
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    by_name = {entry["name"]: entry for entry in staged.get("files", [])}
    added, skipped = 0, []
    for content, name in zip(contents or [], filenames or []):
        safe = Path(name).name
        if Path(safe).suffix.lower() not in IMAGE_EXTENSIONS:
            # The dropzone should have refused it, but a filter that runs only
            # in the browser is a filter that can be bypassed.
            skipped.append(safe)
            continue
        try:
            _, b64 = content.split(",", 1)
            dest = root / safe
            dest.write_bytes(base64.b64decode(b64))
        except Exception:
            skipped.append(safe)
            continue
        if safe not in by_name:
            added += 1
        by_name[safe] = {"name": safe, "path": str(dest)}
    return {"dir": str(root), "files": sorted(by_name.values(), key=lambda e: e["name"])}, added, skipped


def _upload_note(staged, skipped=()):
    files = staged.get("files", [])
    parts = []
    if files:
        names = [e["name"] for e in files]
        shown = ", ".join(names[:3]) + (f" +{len(names) - 3} more" if len(names) > 3 else "")
        parts.append(html.Span(f"{len(names)} staged — {shown}", className="dz-count"))
    if skipped:
        parts.append(html.Div(f"Ignored {len(skipped)}: {', '.join(skipped[:3])} — "
                              f"not an image this pipeline reads.",
                              className="banner banner-warn"))
    return parts or ""


@app.callback(Output("upload-note", "children"), Output("staged", "data"),
              Output("uploads", "contents"),
              Input("uploads", "contents"), State("uploads", "filename"),
              State("staged", "data"), prevent_initial_call=True)
def on_upload(contents, filenames, staged):
    if not contents:
        return no_update, no_update, no_update
    staged, _added, skipped = _stage_uploads(contents, filenames, staged or {})
    # Clearing contents releases the browser's base64 AND lets the same file be
    # dropped again later - an unchanged prop would not fire this a second time.
    return _upload_note(staged, skipped), staged, None


@app.callback(Output("upload-note", "children", allow_duplicate=True),
              Output("staged", "data", allow_duplicate=True),
              Input("clear-uploads", "n_clicks"), prevent_initial_call=True)
def on_clear(_clicks):
    return "", {"dir": None, "files": []}


@app.callback(Output("status", "children"), Input("check", "n_clicks"),
              State("gpu-url", "value"), State("transport", "value"),
              State("api-key", "value"), State("vlm-model", "value"),
              State("checkpoint", "value"), prevent_initial_call=True)
def on_check(_clicks, gpu_url, transport, api_key, vlm_model, checkpoint):
    """One line per thing worth knowing BEFORE an audience is watching: the
    config tree, the segmenter, the OCR engine and the model server."""
    from pipeline.stage3_vlm import VLMClient

    cfg = _cfg_from_controls(gpu_url, vlm_model, VLM_MODE_LIVE, SEND_FULL, 65,
                             transport, api_key, [], checkpoint)
    bits = []

    registry = pq.REGISTRY
    bits.append(f"config: {len(registry.questions)} questions / "
                f"{len(registry.ordered_domains())} domains")
    if registry.warnings:
        bits.append(f"⚠ {len(registry.warnings)} config warning(s)")

    bits.append("u2netp present" if cfg.segmentation_model_path().exists()
                else "⚠ u2netp.onnx MISSING — the segmenter would try to download")

    ocr_ok, ocr_text = stage2b_ocr.available(cfg)
    bits.append(f"OCR: {ocr_text}" if ocr_ok else f"⚠ OCR: {ocr_text}")

    if cfg.use_model:
        checkpoint_path = cfg.resolve_yolox_checkpoint()
        bits.append(f"detector: {checkpoint_path}" if checkpoint_path
                    else "⚠ checkpoint not found")
    else:
        bits.append("detector: annotation sidecars (no checkpoint set)")

    try:
        client = VLMClient(cfg)
        error = client.ensure_registry()
        bits.append(f"⚠ model server: {error}" if error
                    else f"model server: {client.via} OK")
    except Exception as exc:
        bits.append(f"⚠ model server: {exc}")

    for problem in cfg.validate():
        bits.append(f"⚠ {problem}")

    warned = any(b.startswith("⚠") for b in bits)
    return [html.Span(className="dot" if warned else "dot dot-ok"),
            html.Span(" · ".join(bits))]


@app.callback(Output("panel", "children"),
              Input("run", "n_clicks"), Input("tabs", "value"), Input("mode", "value"),
              Input("overlay", "value"),
              State("domain", "value"), State("question", "value"),
              State("staged", "data"), State("gpu-url", "value"),
              State("transport", "value"), State("api-key", "value"),
              State("vlm-model", "value"), State("vlm-mode", "value"),
              State("send-mode", "value"), State("threshold", "value"),
              State("flags", "value"), State("checkpoint", "value"))
def on_run(n_clicks, tab, mode, overlay, domain_id, question_id, staged, gpu_url,
           transport, api_key, vlm_model, vlm_mode, send_mode, threshold, flags,
           checkpoint):
    """One callback for Run and for both selectors.

    Switching tab or mode must NOT re-run the pipeline - that is the whole
    reason all three modes run at once - so the run only happens when the Run
    button is what fired this. Reading the trigger is the ONLY thing this
    wrapper does; everything else is in execute_run(), which is a plain
    function so the tests can drive a real run without a callback context.
    """
    from dash import callback_context

    triggered = (callback_context.triggered[0]["prop_id"].split(".")[0]
                 if callback_context.triggered else "")
    if triggered != "run":
        return _panel(tab, mode, overlay)
    return execute_run(tab, mode, question_id, staged, gpu_url, transport, api_key,
                       vlm_model, vlm_mode, send_mode, threshold, flags, checkpoint,
                       overlay=overlay)


def execute_run(tab, mode, question_id, staged, gpu_url, transport, api_key,
                vlm_model, vlm_mode, send_mode, threshold, flags, checkpoint,
                overlay=DEFAULT_OVERLAY):
    """Run all three modes over the staged photographs and render `tab`."""
    if not question_id:
        return html.Div("Pick a domain and a question first.", className="empty")
    paths = [Path(e["path"]) for e in (staged or {}).get("files", [])
             if Path(e["path"]).exists()]
    if not paths:
        return html.Div("Drop a photograph first.", className="empty")

    cfg = _cfg_from_controls(gpu_url, vlm_model, vlm_mode, send_mode, threshold,
                             transport, api_key, flags, checkpoint)
    cfg.run_id = new_run_id()
    try:
        results = run_all_modes(paths, question_id, cfg, prompts=PROMPTS)
    except Exception as exc:
        # A traceback in the panel beats one in a terminal nobody is looking at.
        return html.Div([html.B("The run failed: "), str(exc),
                         html.Details([html.Summary("Traceback"),
                                       html.Pre(traceback.format_exc())])],
                        className="banner banner-warn")

    _STATE.update({"results": results, "question_id": question_id, "cfg": cfg,
                   "paths": paths})
    return _panel(tab, mode, overlay)


def _panel(tab, mode, overlay=DEFAULT_OVERLAY):
    results = _STATE.get("results") or {}
    question_id = _STATE.get("question_id")
    if not results or not question_id:
        return html.Div("Pick a domain and a question, drop a photograph, then "
                        "press Run all three modes.", className="empty")
    question = get_question(question_id)
    pairs = _by_stem(results)

    if tab == "prompts":
        return _prompts_panel(question, pairs)

    if tab == "detail":
        records = sort_for_display(results.get(mode, []))
        return html.Div(
            [html.Div([html.B(MODES[mode]["label"]), " — ", MODES[mode]["blurb"]],
                      className="banner banner-info")]
            + [detail_card(r, question, mode, overlay) for r in records])

    # "Three modes": the headline view.
    stats = agreement(results)
    tiles = [tile(stats["total"], "photos"),
             tile(stats["unanimous"], "all three agree", "tile-yes"),
             tile(stats["split"], "modes disagree", "tile-no")]
    header = [html.Div(tiles, className="tiles"),
              html.Div([html.B(question.label), " — ", question.answer_semantics],
                       className="banner banner-info")]
    if question.ocr.enabled:
        header.append(html.Div(
            ["Modes 1 and 2 ran OCR on this question; mode 3 did not. ",
             "Where mode 3 differs, the Stage detail tab shows what PP-OCRv6 "
             "read and what the model said it saw."], className="banner banner-info"))
    return html.Div(header + [side_by_side(stem, per_mode, question)
                              for stem, per_mode in pairs])


def _prompts_panel(question, pairs):
    """Exactly what each mode sent. Mode 3's heavy lifting is only assessable
    if you can see the three prompts it was given, and modes 1-2's prompt is the
    one place the OCR block's real wording appears."""
    blocks = [html.Div([html.B("Modes 1 and 2 · system prompt")],
                       className="card-title"),
              html.Pre(question.system_prompt, className="prompt-box")]

    for stem, per_mode in pairs:
        record = per_mode[MODE_ORDER[0]]
        detections = select_relevant(
            record.detection.detections if record.detection else [], question)
        blocks += [
            html.Div([html.B(f"Modes 1 and 2 · user prompt · {record.filename}")],
                     className="card-title"),
            html.Pre(render_user_prompt(question, detections, record.ocr),
                     className="prompt-box"),
        ]
        if question.ocr.enabled and record.ocr is not None and not record.ocr.ran:
            blocks.append(html.Div(
                "The OCR block is absent above because the stage did not run or "
                "read nothing — an error never becomes a line of prompt.",
                className="banner banner-warn"))

    blocks.append(html.Div([html.B("Mode 3 · three legs, three system prompts")],
                           className="card-title"))
    for leg in LEGS:
        blocks += [
            html.Div(LEG_LABELS[leg], className="field"),
            html.Pre(PROMPTS.get(question.id, leg) if PROMPTS is not None
                     else default_leg_system(question, leg), className="prompt-box"),
            html.Div("user prompt (fixed — it carries the JSON contract)",
                     className="field-help"),
            html.Pre(render_leg_user(question, leg), className="prompt-box"),
        ]
    if question.ocr.enabled:
        blocks.append(html.Div(
            "None of mode 3's legs carries an OCR block. That is deliberate: "
            "the model reads the text itself, and feeding it PP-OCRv6's output "
            "would erase the comparison this screen exists to show.",
            className="banner banner-info"))
    return html.Div(blocks)


def main() -> None:
    app.run(debug=False,
            host=os.environ.get("DASH_HOST", "0.0.0.0"),
            port=int(os.environ.get("DASH_PORT", "7873")))


if __name__ == "__main__":
    main()
