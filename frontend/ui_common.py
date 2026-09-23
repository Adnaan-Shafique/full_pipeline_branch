"""Small helpers shared by the Dash UIs.

Deliberately tiny, and deliberately NOT in demo_dash.py, which is frozen
(FROZEN.md). Anything here must be importable with only dash installed - no
pipeline imports, so the frontend/backend arrow is not disturbed.
"""
from __future__ import annotations


def warm_resource_registry(app) -> bool:
    """Render the index once, in-process, before the server starts taking
    requests. Returns True if the registry ended up populated.

    ── The failure this prevents ────────────────────────────────────────────
    Dash only fills `app.registered_paths` while GENERATING THE INDEX PAGE -
    `_collect_and_register_resources()` is reached from `_generate_scripts_html`
    and `_generate_css_dist_html`, and from nowhere else. `_setup_server()` does
    not do it. So a freshly started server has an EMPTY registry until some
    browser asks for `/`.

    That is fine until a browser already has the page open. Restart the server
    with a tab still sitting on it - which is what happens every single time you
    edit code and restart - and the page does not re-request `/`; it goes
    straight for its component chunks. The new process has never rendered an
    index, so the registry is empty and
    `/_dash-component-suites/dash/dcc/async-upload.js` comes back 500 with

        DependencyException: "dash" is not a registered library.
        Registered libraries are: []

    The visible symptom is not an error message. It is a MISSING CONTROL: the
    chunk that failed to load is the one that draws the upload dropzone, so the
    page renders with no way to add a photograph and nothing on screen says why.
    That is what makes it worth a startup call rather than a note in a document.

    Whether it bites depends on the dash build - 4.4.1 happens to register
    "dash" incidentally while setting up its websocket worker, and survives.
    Not every build does, and relying on an incidental side effect is not a
    plan.

    ── Why the test client rather than the private method ───────────────────
    `app._generate_scripts_html()` would populate the registry in one call, but
    it is private and has moved between versions. A GET through Flask's test
    client goes down exactly the path a browser takes, uses only public API, and
    costs one in-process render at startup.

    Never raises: a UI that fails to warm up still starts, and the first real
    page load fills the registry as it always did.
    """
    try:
        with app.server.test_client() as client:
            client.get("/")
        return bool(app.registered_paths)
    except Exception:
        return False
