---
name: bug-fix-runtime
description: Diagnose and fix runtime errors and production regressions with a repeatable workflow. Use when the user reports stack traces, runtime exceptions, production-only bugs, or asks to debug and verify a fix with tests, lint, and build checks.
---

# Runtime Bug Fix Workflow

## Quick Start

Use this workflow when a bug report includes runtime failures, stack traces, or production regressions.

Copy this checklist and update progress:

```markdown
Bug-Fix Progress
- [ ] 1) Capture error context
- [ ] 2) Reproduce locally
- [ ] 3) Identify root cause
- [ ] 4) Implement minimal safe fix
- [ ] 5) Verify with tests + lint + build
- [ ] 6) Summarize root cause and prevention
```

## Workflow

### 1) Capture error context

Gather reproducible context before code changes:

- Error message and stack trace
- Repro steps and expected vs actual behavior
- Runtime/environment details
- Known scope (which paths/features are affected)

Use the helper script to generate a context artifact:

```bash
python ".cursor/skills/bug-fix-runtime/scripts/collect_debug_context.py" --error-file "error.txt" --repro "npm run dev"
```

If no error file exists:

```bash
python ".cursor/skills/bug-fix-runtime/scripts/collect_debug_context.py" --error-text "Paste stack trace headline here"
```

### 2) Reproduce locally

- Run the smallest command that reproduces the failure.
- Prefer stable repro steps over broad/full-suite runs first.
- Do not edit code until a reproducible failure condition exists (or explicit evidence shows why local repro is impossible).

### 3) Identify root cause

- Trace symptom -> failing boundary -> upstream source.
- Confirm root cause with evidence (logs, code path, data shape, config mismatch).
- Validate at least one alternative hypothesis and reject it with evidence.

### 4) Implement minimal safe fix

- Change the smallest surface area that resolves the issue.
- Keep behavior-compatible for unaffected paths.
- Add or update tests close to the failing behavior when feasible.
- Avoid unrelated refactors during incident-style fixes.

### 5) Verify with tests + lint + build

Run validation commands in order and fail fast:

```bash
python ".cursor/skills/bug-fix-runtime/scripts/verify_fix.py" \
  --check "npm test" \
  --check "npm run lint" \
  --check "npm run build"
```

Adjust commands to project tooling as needed.

### 6) Summarize root cause and prevention

Use this response template:

```markdown
## Bug Fix Report

### Root cause
[Precise defect and why it happened]

### Fix applied
[Minimal change and rationale]

### Verification
- [command]: [pass/fail]
- [command]: [pass/fail]
- [command]: [pass/fail]

### Residual risk
[Any remaining edge cases, if any]

### Prevention
[Test, guardrail, or monitoring improvement]
```

## Guardrails

- Prefer evidence over assumptions.
- Never claim success without running verification.
- Keep terminology consistent: "root cause", "repro", "verification".
- Use forward-slash relative paths in commands.

## Utility Scripts

- `scripts/collect_debug_context.py`: Generates a structured debug context markdown file.
- `scripts/verify_fix.py`: Runs checks (tests/lint/build), writes a verification report, and exits non-zero on failure.
