# Running the demo as a service

Two units, two ports, both startable at boot and both restarted if they die.
Templates live in `deploy/`.

| Unit | App | Port | What it is |
|---|---|---|---|
| `fieldops-demo-modes.service` | `frontend/demo_dash_modes.py` | 7872 | three modes over a folder of photographs |
| `fieldops-demo-pipeline.service` | `frontend/demo_dash_pipeline.py` | 7873 | domain → question → one photograph → three modes |

They are independent and can run together — different ports, no shared state
beyond the read-only `config/` tree and `demo_runs/`. The pipeline unit also
carries a commented `FIELDOPS_CONFIG_DIR`, for pointing a host at a second
question set without touching the checkout.

**Do not hand-edit the templates in `deploy/`.** Each carries the checkout path
in five places, and a unit with one of them wrong fails with a bare
`203/EXEC` — which names no path at all and reads the same whether it was the
interpreter, the app file or `WorkingDirectory`. `deploy/make_unit.sh` derives
all five from its own location, checks each exists, and prints the unit:

```bash
cd /path/to/your/checkout            # wherever you actually cloned it
./deploy/make_unit.sh pipeline       # read it first; it writes nothing itself
```

The templates remain as the readable reference for what it generates and why
each line is there.

The route is proxy, via `llm_proxy_v4` on FALCONPRD — see `PROXY.md`. The one
thing the script cannot check for you is the virtualenv: SETUP.md records one
on this VM built against a python with no `_ctypes`, which imports fine and
then fails deep inside a dependency. The script prefers a venv that passes that
check and warns loudly when none does.

## What is already running, and stopping the old ones

Before installing anything, find out what is there. A second copy of the same
app on the same port does not both-start — one wins, the other restarts every
five seconds forever (`Restart=always`), and `systemctl status` on the one you
just installed shows it failing while the page loads perfectly from the other
one. That is a confusing half-hour.

### Find them

```bash
# Anything named for this demo, running or not, and any leftover unit files
systemctl list-units   --all 'fieldops*' 'demo*' 'dash*'
systemctl list-unit-files      'fieldops*' 'demo*' 'dash*'
ls -l /etc/systemd/system/*.service ~/.config/systemd/user/*.service 2>/dev/null

# A user service is invisible to the commands above - it has its own manager
systemctl --user list-units --all 'fieldops*' 'demo*'
```

Units are only half of it. A demo started by hand in a tmux pane months ago is
still holding its port and answers nothing to systemd:

```bash
# Who owns the demo ports right now - the authoritative answer
sudo ss -lptn 'sport = :7870 or sport = :7871 or sport = :7872 or sport = :7873'

# Any python running one of these apps, however it was started
ps -eo pid,user,etime,cmd | grep -E 'demo_dash|dash' | grep -v grep

# Panes someone left behind
tmux ls 2>/dev/null; screen -ls 2>/dev/null
```

`ss` is the one to trust. If a port is held but no unit claims it, it is a
stray process — which also means nothing will restart it once you stop it, and
nothing restarted it across the last reboot either.

### Read one before you touch it

```bash
systemctl cat  fieldops-demo-modes            # the unit as installed, paths and all
systemctl show fieldops-demo-modes -p ExecStart -p WorkingDirectory -p Environment
systemctl status fieldops-demo-modes
journalctl -u fieldops-demo-modes --since -1h --no-pager | tail -40
```

`systemctl cat` is what tells you whether an old unit points at the previous
checkout or at this one. An old unit pointing at a directory that still exists
is the dangerous case: it starts, it serves, and it serves **last month's
code** — with nothing on screen saying so, since the page looks identical.

### Stop them

Stopping and disabling are different, and only doing the first is why a service
you "turned off" is back after a reboot:

```bash
sudo systemctl stop    fieldops-demo-modes     # now
sudo systemctl disable fieldops-demo-modes     # and at boot
sudo systemctl status  fieldops-demo-modes     # expect inactive (dead)
```

Remove it only once you are sure you will not want it back:

```bash
sudo rm /etc/systemd/system/fieldops-demo-modes.service
sudo systemctl daemon-reload
sudo systemctl reset-failed                    # clears the corpse from list-units
```

For a user service, the same commands with `--user` and no `sudo`. For a stray
process with no unit, `kill <pid>` — plain, not `-9`: the app catches SIGINT
(`KillSignal=SIGINT`) and a run mid-batch finishes rather than leaving a
half-written folder under `demo_runs/`.

### Then confirm the port is actually free

```bash
sudo ss -lptn 'sport = :7873'                  # expect no output
```

If something still holds it, the new unit will flap rather than fail cleanly,
and its journal will say `Address already in use` once every five seconds.

## 1. The secret, separately

The API key does **not** go in the unit file. Unit files are world-readable and
`systemctl cat` prints them to anyone who asks.

```bash
sudo install -m 600 /dev/null /etc/fieldops-demo.env
sudo vi /etc/fieldops-demo.env        # contents from deploy/fieldops-demo.env.example
sudo chown root:root /etc/fieldops-demo.env
ls -l /etc/fieldops-demo.env          # expect -rw-------
```

## 2. The unit

Generate it, read it, then install it. The step that needs root is a `tee` you
can see the input to:

```bash
./deploy/make_unit.sh pipeline --user "$(id -un)" --group "$(id -gn)"
./deploy/make_unit.sh pipeline | sudo tee /etc/systemd/system/fieldops-demo-pipeline.service
sudo systemctl daemon-reload
sudo systemctl enable --now fieldops-demo-pipeline
```

`modes` in place of `pipeline` gives the 7872 unit. A second instance for a
second operator — see "Sharing the link" below — takes `--port` and its own
filename:

```bash
./deploy/make_unit.sh pipeline --port 7874 \
    | sudo tee /etc/systemd/system/fieldops-demo-pipeline-b.service
```

Re-run the script and re-install after moving or re-cloning the checkout. That
is the whole reason it exists: a unit still pointing at last month's directory
fails with the same silent `203/EXEC`.

## 3. Check it

```bash
systemctl status fieldops-demo-pipeline
journalctl -u fieldops-demo-pipeline -f       # Ctrl-C to stop following
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7873/    # expect 200
```

Then open `http://10.19.75.122:7873/` and press **Check host**, which reports
how many questions loaded, whether the OCR models are present, and the model
route. It should say the proxy route, not "answers will be MOCK". If it says
MOCK, the environment file is not reaching the process — `systemctl show
fieldops-demo-pipeline -p Environment` shows what it actually got.

On 7872 the equivalent button is **Load / check model**.

## If you have no root

A user service needs no sudo, but it stops when you log out unless lingering is
enabled — and enabling lingering itself needs root. Without either, the service
dies with your SSH session, which is not a service.

```bash
mkdir -p ~/.config/systemd/user
./deploy/make_unit.sh pipeline --env-file "$HOME/.config/fieldops-demo.env" \
    > ~/.config/systemd/user/fieldops-demo-pipeline.service
# A user unit has no User=/Group= and wants a different target.
sed -i -e '/^User=/d' -e '/^Group=/d' \
       -e 's|WantedBy=multi-user.target|WantedBy=default.target|' \
    ~/.config/systemd/user/fieldops-demo-pipeline.service
install -m 600 deploy/fieldops-demo.env.example ~/.config/fieldops-demo.env
vi ~/.config/fieldops-demo.env         # the real key
systemctl --user daemon-reload
systemctl --user enable --now fieldops-demo-pipeline
sudo loginctl enable-linger $USER      # needs root ONCE; without it, see above
```

## The firewall

Binding 0.0.0.0 is not enough — the host still has to allow the port:

```bash
sudo firewall-cmd --permanent --add-port=7872/tcp && sudo firewall-cmd --reload
sudo firewall-cmd --permanent --add-port=7873/tcp && sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

If the page loads from the VM itself (`curl 127.0.0.1:7872`) but not from a
laptop, it is the firewall, not the app.

## Sharing the link with more than one person

The units above bind `0.0.0.0`, so once the firewall allows the port the link
is `http://<host>:7873/` and anyone on the network can open it. Three things
about that are worth knowing **before** you send it round, because none of them
announces itself on screen.

### One run at a time, for everybody

`demo_dash_pipeline.py` keeps the last run's results in `_STATE`, a single
module-level dict (`frontend/demo_dash_pipeline.py:582`). Uploads are per
browser — they are staged into a run-id folder held in a `dcc.Store` — but the
**results are not**. Switching tab, mode or overlay does not re-run the
pipeline, by design; it re-renders from `_STATE`. So:

> Alice runs her photograph. Bob runs his. Alice clicks **Detail** and sees
> **Bob's** photograph and Bob's answers, under her own question.

Nothing errors, nothing says "this is not yours", and the screen is entirely
plausible. Treat 7873 as a **single-operator screen**: one person drives while
the others watch, either over a shared screen or by agreeing who runs next. If
two people genuinely need to run at once, give them a process each on different
ports (`DASH_PORT=7874` and a second unit) — that is one copied unit file and
costs nothing, since the config tree is read-only and `demo_runs/` is keyed by
run id.

This is the same constraint SERVICE.md's gunicorn note refers to, but it is not
a gunicorn problem: it holds at `--workers 1`, and with the built-in server, and
with one process on one port. The worker count only decides whether a *single*
user also sees blanks at random.

### The proxy API key is readable by anyone who opens the page

The key field is `type="password"`, which masks it on screen and does nothing
else. Dash serialises the layout — including that input's value — into
`/_dash-layout`, so anyone who can load the page can read the key out of
devtools or `curl`. It is not a secret from your viewers.

On a closed network with colleagues that is usually fine, and it is the trade
this demo already makes. Do not put the link anywhere wider on the assumption
the field is masked. If the key must not travel, start the process with
`FIELDOPS_VLM_API_KEY` unset and have the operator paste it at run time — it is
then in that one browser rather than in every page load.

### Mode 3's prompts are global and they persist

Editing a mode-3 system prompt through the UI writes `config/prompts.yaml` and
takes effect immediately **for everyone using that process**, and survives a
restart. There is one copy per machine, not per viewer. A teammate
experimenting with the answer leg changes what the next person's run does, with
nothing on their screen saying the prompt is not the default. Before a demo,
check what is in it:

```bash
cat config/prompts.yaml        # absent means every leg is on its built-in default
```

### Anyone who has the link can run the pipeline

There is no login. A viewer can upload photographs, spend GPU time, and fill
`demo_runs/` — which nothing prunes, and which grows by annotated copies of
every photograph across three modes. That is appropriate for a known audience
on a closed network and not for a link that might get forwarded. If you need
even a weak fence, put it in front rather than in the app: an nginx reverse
proxy with basic auth on 443, forwarding to `127.0.0.1:7873`, and bind the app
to `127.0.0.1` by setting `DASH_HOST=127.0.0.1` in the unit so the port is not
reachable directly.

## Updating

```bash
cd /path/to/your/checkout
git pull
for t in tests/test_*.py; do ./venv_yolox/bin/python "$t" >/dev/null || echo "FAILED $t"; done
sudo systemctl restart fieldops-demo-pipeline
```

A `git pull` into the same directory needs no new unit. **Re-cloning somewhere
else does** — re-run `deploy/make_unit.sh` and re-install, or the unit keeps
serving the old checkout with nothing on the page saying which one it is.

Restart is required for a code change: the process imports everything once at
start. **Hard-reload the browser tab** afterwards (Ctrl-Shift-R) — a cached page
keeps the old callback graph and fails in confusing ways when the layout has
changed.

## What the service does NOT survive

`config/prompts.yaml` — mode 3's tuned prompts — lives in the project directory
and is read **once at import**. Editing prompts through the UI writes the file
and takes effect immediately in that process, but editing the file by hand
needs a restart.

Run outputs accumulate under `demo_runs/` and nothing prunes them. Each run
writes annotated copies of every photograph across three modes, so a long
rehearsal is measured in gigabytes. Before a demo:

```bash
du -sh demo_runs/
# keep the last few, delete the rest
```

The service user must own the project directory: the app writes `demo_runs/`,
`config/prompts.yaml` and the staged uploads folder. Running it as a user who
can only read the checkout fails at the first run, not at startup.

## Werkzeug, not gunicorn

This runs Dash's built-in server. That is a development server, and it says so
in the log — for a demo on a closed network in front of a known audience it is
the right trade: no extra dependency, no extra failure mode, and it is threaded,
so a slow pipeline run does not block the page.

If this ever outlives the demo, `frontend/demo_dash_modes.py` already exposes
`server = app.server`, so the move is one line:

```
ExecStart=.../venv_yolox/bin/gunicorn --chdir app --workers 1 --threads 4 \
          --timeout 600 --bind 0.0.0.0:7872 demo_dash_modes:server
```

Keep `--workers 1`. Results are held in a module-level dict, so a second worker
would answer half the requests from an empty set and the mode switcher would
show blanks at random.

That dict is also why one process serves one operator, not a team — see
"Sharing the link with more than one person" above. `--workers 1` fixes the
blanks; it does not make two people's runs independent, and nothing here does.
