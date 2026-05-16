# prompt-forge

Automated prompt engineering for LLM tool-calling agents: generate, evaluate, and refine system prompts against a deterministic test suite.

## Running the Demo

### Production Mode (default)

```bash
python demo_run.py
```

Single-case failure signals converge at v1 to prevent Critic overfit.
This is the recommended mode — refine only triggers when failure signal
is strong enough (≥ 2 cases).

Example output:
```
v1: 15/16 (93.8%)
Failures: {'hallucinated_call': 1}
Lever 1: v1 converged (dominant failure count 1 < threshold 2)
```

### Demo Mode (for showcase)

```bash
python demo_run.py --demo
```

Lowers refine threshold to 1 to demonstrate the v1 → v2 → v3 chain on
small test sets. Lever 2 (regression guard) protects against overfit-induced
degradation.

Example output (illustrative):
```
v1: 14/16 (87.5%) → refine
v2: 15/16 (93.8%) → refine
v3: 16/16 (100.0%) → converged
```

### Why Two Modes?

Production mode reflects the design principle that prompt refinement
should require sufficient signal — refining for a single failed case
risks introducing new failures elsewhere (observed in stress testing).

Demo mode exists because on a 16-case test set, failures are often
sparsely distributed (1 case per category), and v1 generator quality
is already near-ceiling with modern models like Gemini 2.5 Flash.
Showing the full refine chain requires lowering the threshold for
demonstration purposes.

In production with 200+ case test sets, failures are dense enough
that the default threshold (2) is the correct setting — refine triggers
naturally when there is enough signal to optimise against.

## Roadmap

### v0.3 (current) — Narrow Tool-Call Evaluator with Informational Fields

- 7 failure modes with deterministic detectors (plugin registry)
- Each mode: `(detector, mitigation_hint, severity)` triple in `lib/failure_modes.py`
- Informational fields (`reason`, `description`, etc.) — presence + type checked, content not compared
- Minimum failure threshold (Lever 1) + Regression guard (Lever 2) for stable refine loops

### v0.4 — General Prompt Evaluator (Planned)

The current evaluator is optimised for narrow tool-calling scenarios.
A more general design separates concerns into three layers:

**Layer 1 — Deterministic checks** (current behaviour preserved)
- No-call vs call mismatch
- Action type / function name
- Required structured params (IDs, enums, codes)

**Layer 2 — Behaviour-level LLM judge**
- Expected behaviour oracle (e.g. `"escalate_because_required_identifier_missing"`)
- Judge rubric per test case (1–3 natural-language criteria)
- LLM judges whether actual behaviour satisfies expected behaviour,
  NOT whether free-text params are semantically equal

**Layer 3 — Informational fields**
- Free-text fields (`reason` / `description` / `summary` / etc.)
- Presence + type only, no content judgement

This separation moves intent-alignment from "is this reason text equivalent?"
to "did the agent do the right thing?" — which is both more robust to LLM
phrasing variance and more aligned with what production systems actually care about.

### v0.5 — Adversarial Test Generation (Planned)

Auto-generate boundary cases that target the dominant failure mode identified
by the Critic, reducing manual test curation effort.
