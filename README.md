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

### And against gemini-3.5-flash-lite

```bash
PROVIDER=gemini python3 orchestrator.py
```

- **Reliable tool calling:** 6 of 6 tool calls had valid arguments (llama
  sent malformed ones), so the clean answer (q1) got all three checks and
  was **approved** — on llama it escalated because two calls failed.
- **Smarter stopping:** on the hallucination (q3) it stopped after
  `golden_eval` = 0, like llama. On the subtly wrong answer (q5) it called
  `golden_eval` (0.80, not decisive), then `groundedness_check`, saw the
  contradiction, and stopped on its own without the judge — the right call,
  since under the OR rule no judge score could rescue it.
- **A third real bug it caught:** q5's verdict was right (escalate) but the
  *reason* was wrong — "insufficient evidence: not all checks were run" —
  because the policy checked for missing evidence before checking for real
  failures. That contradicted the policy's own rule (approving needs full
  evidence; escalating doesn't), so actual flags are now evaluated first and
  reported as the reason. Regression test included.

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

This structure is per **model** (`GATE_PROFILES` in `decision_policy.py`,
keyed `provider/model`), not fixed, because two models from the same
provider can judge very differently. A model proven reliable enough can
graduate `groundedness_check` from Tier 2 into a trusted Tier 1 hard gate;
an unmeasured model always gets the conservative profile.

### A second model: gemini-3.5-flash-lite

The same three labeled answers, measured the same way:

| Tool | q1 (correct) | q3 (hallucinated) | q5 (subtly wrong) | vs. llama3.1:8b |
|---|---|---|---|---|
| `golden_eval` | 1.00 ✓ | 0.00 ✓ | 0.80 — missed | same pattern: catches gross errors, not subtle ones |
| `groundedness_check` | grounded ✓ | flagged ✓ | flagged ✓ | **better** — 3/3, no false positive on the paraphrase |
| `llm_judge` | 9.3 ✓ | **9.0** ✗ | **10.0** ✗ | **worse** — two false negatives instead of one |

Through the policy, all three verdicts are right (approve, escalate,
escalate), and **AND would have failed again**: a judge giving the subtly
wrong answer 10/10 would have approved it. `test_decision_policy.py` replays
these real scores too.

**The bigger finding:** `llm_judge` only sees the question and the answer --
never the golden answer or the source context. So its "correctness" score
really measures whether an answer *sounds* right, which is why a fluent
hallucination scores 9-10 on both models. That's a design limit, not a model
limit: the fix is reference-guided judging (give the judge the golden answer
or context), or scoping the judge to clarity and tone and leaving
correctness to `golden_eval` and `groundedness_check`.

(The judge scores in these tables are the model's own overall score, as
recorded at the time. The judge now computes its overall score in code -- see
below.)

## Run-level drift (`tools/run_eval.py`)

The per-answer gate above decides whether one answer ships. Drift asks a
different question: **has this whole run gotten worse than this model's
normal?**

```bash
python3 tools/run_eval.py --save-baseline   # record this model's baseline
python3 tools/run_eval.py                   # compare a new run to it (exit 1 on drift)
```

- **A full, fixed pass, not the agent.** The orchestrator skips checks once
  it has enough evidence, so its averages depend on what it chose to call and
  aren't comparable run to run. `run_eval.py` runs all three LLM tools on
  every labeled answer, so two runs are always measured the same way.
- **Three run averages:** mean `golden_eval` score (0-1), share of answers
  judged grounded (0-1), and mean `llm_judge` score (0-10).
- **Drift math (no LLM):** for each metric,
  `relative_drop = (baseline - current) / baseline`; flagged above 10%; a rise
  is never flagged; any flag escalates the run (`run_level_check`).
- **One baseline per provider/model** in `data/baselines/`, like gate
  profiles -- a different model has a different normal. The committed
  `ollama__llama3.1_8b.json` comes from a real full pass (an earlier version
  of this repo shipped hand-written placeholder numbers; they were replaced).
- **A real edge case it exposed:** llama judged all three answers
  ungrounded (including its known false positive on the correct one), so its
  groundedness baseline is **0.0**. A relative drop from 0 is undefined, so
  drift on that metric can't be measured for llama -- the check now reports
  that explicitly instead of silently passing.
- **The two recorded baselines show why "normal" is per model:**

  | Run average | llama3.1:8b | gemini-3.5-flash-lite | Correct for these 3 answers |
  |---|---|---|---|
  | golden similarity | 0.57 | 0.60 | — |
  | grounded share | **0.00** | **0.33** | **0.33** (only q1 is correct) |
  | judge score | 7.53 | **9.77** | should be low for q3 and q5 |

  Gemini's groundedness matches the correct rate exactly; its judge rates
  everything high, including 9.3 for the hallucination. Comparing a Gemini
  run against llama's baseline would report a large "improvement" that is
  just a different model's normal.
- **What a flag means here:** the candidate answers are fixed, so a drift
  flag means the *evaluator's* judgments changed (for example, a provider
  silently updating a model). Pointed at a live RAG bot's answers, the same
  check would catch the bot regressing.
- **Honest limit:** with three labeled answers, one answer changing moves an
  average by about 33%, so this demonstrates the mechanism, not a
  statistically meaningful signal. A larger golden set comes first.

**The judge's overall score is computed in code** (equal-weight mean of
correctness, clarity and tone), not taken from the model, so it is always
consistent with its breakdown. The model's own number is kept as
`model_reported_score` for comparison; a response without a usable breakdown
is a failed check, not a score.

## The four tools

| Tool | Question it answers | LLM call? |
|---|---|---|
| `golden_eval` | Does this answer mean the same thing as a known-correct reference? | yes |
| `groundedness_check` | Does every claim in this answer actually trace back to the retrieved context? | yes |
| `llm_judge` | How good is this answer on a correctness/clarity/tone rubric? | yes |
| `drift_check` | Have this run's averages regressed vs. this model's recorded baseline? | no — pure arithmetic |

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
python3 tools/drift_check.py       # no model needed: real baseline vs synthetic runs
python3 tools/run_eval.py          # full pass on every answer + drift vs baseline
```

Each script runs a small demo against `data/sample_candidate_runs.json`,
which deliberately includes a clean-pass answer, a hallucinated/ungrounded
answer, and a subtly-wrong answer — so you can see each tool's judgment
differ across the three.

**Run the smoke tests** (no model needed — uses a mock provider):

```bash
python3 tests/test_tools_smoke.py
# all 49 tests run offline -- each tests/*.py file also runs standalone
# or, with pytest installed:
pip install -r requirements.txt && pytest
```

## Swapping providers

The whole project talks to models through one interface (`llm_provider.py`),
so switching providers is a one-line change, not a rewrite:

```bash
PROVIDER=ollama python3 tools/llm_judge.py      # default
PROVIDER=gemini python3 tools/llm_judge.py      # Google Gemini (free tier, API key)
PROVIDER=mock python3 tools/llm_judge.py        # canned response, for tests
```

### Gemini

```bash
export GEMINI_API_KEY=...            # create one at https://aistudio.google.com/apikey
PROVIDER=gemini python3 orchestrator.py
```

`GEMINI_MODEL` overrides the default model (`gemini-3.5-flash-lite` -- a
pinned version, chosen over `gemini-3.8-flash`, whose free tier allows only
20 requests and was returning 503 "high demand", and over `-latest` aliases,
which can change model underneath a measurement). Google retires model
versions regularly (`gemini-2.5-flash` already returns 404 for new keys), so
if the default stops working, list what your key can use:

```bash
curl -s -H "x-goog-api-key: $GEMINI_API_KEY" "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200" | python3 -c "import json,sys; [print(m['name']) for m in json.load(sys.stdin).get('models',[]) if 'generateContent' in m.get('supportedGenerationMethods',[])]"
```

Rate limits (429) and "high demand" errors (503) are retried with backoff
(honoring `Retry-After`, about two minutes total) before giving up. The provider
uses the REST API through the standard library -- no SDK -- and translates
between the project's Ollama-style chat format and Gemini's function-calling
format at the edge, so the orchestrator never branches on provider. The key
travels in a request header, never the URL, and must never be committed
(`.env` is gitignored).

Two details that mattered:
- **The MCP SDK starts tool servers with a minimal environment**, so
  `PROVIDER` and the API key are forwarded to them explicitly (and nothing
  else) -- otherwise the tools would silently keep using the default provider.
- **Gemini's gate profile is provisional** -- a copy of the most conservative
  measured profile -- until Gemini is measured against the labeled sample
  runs the same way llama3.1:8b was. Any unmeasured provider gets that
  conservative profile rather than unearned trust.

**Planned:** `PROVIDER=groq` (fast hosted Llama, free tier) and
`PROVIDER=anthropic`, each a drop-in class in `llm_provider.py`. Nothing in
the four tools or the orchestrator needs to change when a provider is added.

## What's next

All four milestones are done (see Status above). Possible follow-ups, not
required to prove the core architecture:

- Measure Gemini (now wired in), Groq and Anthropic the same way Ollama was measured, and let a
  provider proven reliable enough graduate `groundedness_check` from Tier 2
  into a trusted Tier 1 hard gate (see `GATE_PROFILES` in `decision_policy.py`).
- An HTTP-transport MCP deployment, if this ever needs to be network-reachable
  rather than local-only (see MCP servers section above).
- A real Harness pipeline alongside the GitHub Actions one, since Harness is
  what this pattern is designed for professionally.
