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
- [ ] **M3** — orchestrator agent chains tool calls, makes real branching decisions
- [ ] **M4** — human-approval gate + GitHub Actions wrapper + demo write-up

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
Nothing in the four tools or the future orchestrator needs to change when a
new provider is added.

## Roadmap

- **M2 (MCP):** wrap each tool in its own MCP server so any MCP-compatible
  client (Claude, or a custom orchestrator) can discover and call them.
- **M3 (orchestrator):** an agent loop that, given a new candidate answer,
  decides which tool to call first, reads the result, and decides what to
  call next — e.g. skip straight to human escalation if groundedness fails,
  only run `drift_check` if the first two checks pass.
- **M4 (CI + write-up):** wire the orchestrator into a GitHub Actions
  workflow (the free stand-in for the Harness pipelines this pattern is
  designed for professionally), add an explicit human-approval step before
  any verdict is final, and write up the finished project as a portfolio
  case study.
