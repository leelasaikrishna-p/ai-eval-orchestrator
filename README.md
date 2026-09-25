# AI Eval Orchestrator

An agent that decides, on its own, how to evaluate a GenAI system's output —
chaining golden-dataset comparison, groundedness checking, LLM-as-judge
scoring, and drift detection, with a human-approval gate — instead of
running a fixed checklist every time.

## Why this project

Most "AI quality" demos either check output against a fixed rubric (not
agentic — a human decided every branch in advance) or call themselves
"agentic" for wrapping a single LLM call in an API (not evaluation — no
real quality gate). This project tries to actually earn both words: real
evaluation methodology (golden sets, groundedness, drift), and a real
orchestrator that decides what to run next based on what it's already seen.

## Status

**M1 — done.** Four evaluation tools work standalone, verified against a
mock provider. **Sign-off policy — done.** Combines the four tools' verdicts
into an approve/escalate decision, with a gate structure derived from
actually measuring each tool's reliability against a real model (see below)
rather than assumed.

- [x] **M1** — four evaluation tools working standalone
- [x] **Decision policy** — provider-aware gate structure (`decision_policy.py`)
- [x] **M2** — each tool exposed as its own MCP server (stdio transport), verified end-to-end over the real protocol
- [x] **M3** — orchestrator agent chains tool calls, makes real branching decisions
- [x] **M4** — CI + a real human-approval gate, verified live on GitHub Actions

## The orchestrator (M3)

`orchestrator.py` is the actual agentic piece: it connects to the three
per-answer MCP servers as a real client, hands their schemas to the model
via Ollama's native tool-calling (`/api/chat`), and lets the **model**
decide which tool(s) to call and in what order — not a hardcoded sequence.
The model's job stops at gathering evidence; the final approve/escalate
verdict is computed by the tested `decision_policy.sign_off()`, not the
model's own opinion. Agentic investigation, governed decision.

```bash
source .venv/bin/activate
python3 orchestrator.py   # runs all three sample cases, prints the full trace
```

### What a real run against llama3.1:8b actually showed

- **Real short-circuiting, working as designed:** on the hallucinated answer
  (score 0.0 from `golden_eval_tool`), the model reasoned *"This alone is
  enough evidence... I do not need to call the other tools"* and stopped
  after one call — genuine agentic behavior, not a scripted pipeline.
- **The governed decision overrode the model's own wrong opinion.** In one
  run, two tool calls failed (the model sent malformed arguments — a real
  reliability limit of this small local model on multi-turn tool calls),
  and the model's own text concluded *"I would recommend approving."*
  `decision_policy.sign_off()` disagreed and escalated anyway, because
  approving requires full evidence and two checks never actually ran. The
  tested code was right; the model's free-form judgment was wrong. That's
  the entire argument for keeping the verdict deterministic instead of
  trusting the agent's own conclusion.
- **A real bug this caught:** the first version of the orchestrator stored
  a failed tool call's error as if it were a real result, which would have
  crashed `sign_off()` the moment it tried to read a `"grounded"` key that
  didn't exist. Fixed to treat a failed call as *not run* (absent evidence),
  not a malformed result — see `tests/test_orchestrator_verdict.py` for the
  regression test.

## CI + the human-approval gate (M4)

Two workflows, both verified live on real GitHub Actions infrastructure,
not just written and assumed to work:

- **`ci.yml`** — runs all four test suites (mock/no-LLM) on every push.
- **`evaluate-and-approve.yml`** — the Harness-shaped demo: an `evaluate`
  job computes a verdict via `decision_policy.sign_off()` against one of
  the real recorded score sets from the Ollama run, feeding a
  `human_approval` job gated on a genuine GitHub Environment
  (`production-approval`, with a real required reviewer configured via the
  API) — which feeds a final `publish` job.

Both branches proven live:
- **Escalate case (`q3`, the hallucination):** `human_approval` genuinely
  paused — GitHub would not run it until a real click on "Review
  deployments" in the Actions UI. It failed once for a real reason: the
  `reasons` output contained an apostrophe (*"doesn't match..."*) that broke
  shell quoting when interpolated directly into a `run:` script via `${{ }}`.
  Fixed by passing it through an env var instead — the general fix for any
  arbitrary string content in a GitHub Actions step.
- **Approve case (`clean`):** `human_approval` was skipped entirely —
  `evaluate` fed straight into `publish` with no pause.

Trigger it yourself: **Actions tab → Evaluate and Approve → Run workflow**,
pick a case (`q1`, `q3`, `q5`, or `clean`).

## MCP servers (M2)

Each tool is wrapped as its own MCP server in `mcp_servers/`, using the
**stdio transport** — the server runs as a local subprocess, communicating
over stdin/stdout, exactly like Claude Desktop's local MCP integrations.
Not internet-accessible by design at this stage: no network, no hosting, no
auth to worry about. (MCP also supports an HTTP transport for services that
do need to be network-reachable — a possible M4+ stretch goal, not needed
to prove the agentic architecture itself.)

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Each server just sits waiting for an MCP client to connect over stdio --
# that's correct behavior, not a hang. Point an MCP client's config at the
# script path to actually use it, e.g.:
python3 mcp_servers/drift_check_server.py
```

`tests/test_mcp_servers.py` spins up each server as a real subprocess and
talks to it over the actual MCP protocol (not just calling the Python
function directly) — proof the wrapping works, not just that it imports.

### Sign-off policy: why the gates are structured this way

Running all four tools against a real model (`llama3.1:8b` via Ollama)
surfaced real, measured differences in reliability, not assumed ones:

| Tool | Behavior observed | Trust level |
|---|---|---|
| `drift_check` | No LLM involved — pure arithmetic | **Tier 1 — always trusted** |
| `golden_eval` | 0 false positives; correctly zeroed a hallucinated answer | **Tier 1 — trusted hard gate** |
| `groundedness_check` | 1 false positive (flagged a *correct* paraphrase as unsupported) | Tier 2 — advisory only |
| `llm_judge` | 1 false negative (rated a factually wrong answer 8/10) | Tier 2 — advisory only |

Tier 2 uses **OR, not AND**: either check flagging a problem is enough to
escalate, even alone. This was a deliberate correction — an early AND-based
draft (both must agree) would have let the real error past, because
`llm_judge` missed it and only `groundedness_check` caught it. Since the
only two outcomes are *approve* or *escalate to a human* — never a silent
auto-reject — a false escalation only costs a minute of review, while a
false approval ships a wrong answer. That asymmetry is why the policy is
biased toward escalating when any single credible signal fires.
`tests/test_decision_policy.py` runs the actual scores from that real
model run (not synthetic data) and includes a test proving the AND version
would have failed on the exact case OR gets right.

This structure is per-provider (`GATE_PROFILES` in `decision_policy.py`),
not fixed — once Gemini/Groq/Anthropic are measured the same way, a
provider proven reliable enough can graduate `groundedness_check` from
Tier 2 into a trusted Tier 1 hard gate.

## The four tools

| Tool | Question it answers | LLM call? |
|---|---|---|
| `golden_eval` | Does this answer mean the same thing as a known-correct reference? | yes |
| `groundedness_check` | Does every claim in this answer actually trace back to the retrieved context? | yes |
| `llm_judge` | How good is this answer on a correctness/clarity/tone rubric? | yes |
| `drift_check` | Have this run's aggregate scores regressed vs. a historical baseline? | no — pure arithmetic |

The demo domain is a small fictional invoicing product ("Ledgerly") with a
handful of help-center docs and a golden Q&A set — see `data/`. Nothing here
is real product data.

## Running it

**Zero-cost path (default): [Ollama](https://ollama.com), local, no API key.**

```bash
brew install ollama          # or download from ollama.com
ollama serve                 # leave running in a separate terminal
ollama pull llama3.1:8b      # one-time download, ~4.7GB

python3 tools/golden_eval.py
python3 tools/groundedness_check.py
python3 tools/llm_judge.py
python3 tools/drift_check.py       # no model needed, runs immediately
```

Each script runs a small demo against `data/sample_candidate_runs.json`,
which deliberately includes a clean-pass answer, a hallucinated/ungrounded
answer, and a subtly-wrong answer — so you can see each tool's judgment
differ across the three.

**Run the smoke tests** (no model needed — uses a mock provider):

```bash
python3 tests/test_tools_smoke.py
# or, with pytest installed:
pip install -r requirements.txt && pytest
```

## Swapping providers

The whole project talks to models through one interface (`llm_provider.py`),
so switching providers is a one-line change, not a rewrite:

```bash
PROVIDER=ollama python3 tools/llm_judge.py      # default
PROVIDER=mock python3 tools/llm_judge.py        # canned response, for tests
```

**Planned:** `PROVIDER=gemini` and `PROVIDER=groq` (both have genuine free
API tiers) and `PROVIDER=anthropic`, each a drop-in class in
`llm_provider.py` implementing the same `generate(prompt) -> str` method.
Nothing in the four tools or the orchestrator needs to change when a new
provider is added.

## What's next

All four milestones are done (see Status above). Possible follow-ups, not
required to prove the core architecture:

- Measure Gemini/Groq/Anthropic the same way Ollama was measured, and let a
  provider proven reliable enough graduate `groundedness_check` from Tier 2
  into a trusted Tier 1 hard gate (see `GATE_PROFILES` in `decision_policy.py`).
- An HTTP-transport MCP deployment, if this ever needs to be network-reachable
  rather than local-only (see MCP servers section above).
- A real Harness pipeline alongside the GitHub Actions one, since Harness is
  what this pattern is designed for professionally.
