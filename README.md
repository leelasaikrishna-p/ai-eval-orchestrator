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

**M1 — done.** The four evaluation tools work standalone, verified against
a mock provider (see `tests/test_tools_smoke.py`) so the logic is proven
independent of any specific model.

- [x] **M1** — four evaluation tools working standalone
- [ ] **M2** — each tool exposed as its own MCP server
- [ ] **M3** — orchestrator agent chains tool calls, makes real branching decisions
- [ ] **M4** — human-approval gate + GitHub Actions wrapper + demo write-up

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
