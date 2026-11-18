"""Lean checking through the official clever-bench API (ProblemViewTask.submit_async).

Two checks per attempt, matching the paper's stage-wise metrics:
  1. implementation check: implementation + #test cases, proof left as `sorry`
     -> compiled = Lean reports no errors (tests pass)
  2. full check: implementation + helper lemmas + proof
     -> proved = no errors AND no `sorry` anywhere
Step 2 only runs if step 1 passes.

The official checker only looks for `sorry`. We additionally reject text that
could make Lean accept a file without a real proof (see BANNED).
"""
import re

from clever_bench.lean_problem import Lemma

import config
import data

BANNED = re.compile(
    r"\bsorry\b|\badmit\b|\baxiom\b|\bnative_decide\b|implemented_by|@\[extern|\bunsafe\b|"
    r"#[a-z_]+|\bmacro(_rules)?\b|\belab\b|\bsyntax\b|\bnotation\b|\brun_cmd\b|\brun_tac\b|"
    r"\battribute\s*\[|\bset_option\s+debug|\bopaque\b|\binstance\b.*\bDecidable\b.*:=\s*sorry")


def banned_reason(*parts):
    for part in parts:
        m = BANNED.search(part or "")
        if m:
            return f"Forbidden construct `{m.group(0)}` used."
    return None


def _view(pid, impl, proof, helpers):
    v = data.task().get_view(pid)          # ground-truth spec, signature, tests, theorem
    v.implementation = impl
    v.correctness_proof = proof            # None -> clever-bench inserts `by sorry`
    v.correctness_helper_lemmas = [Lemma(statement=helpers, proof="")] if helpers and proof else []
    return v


async def _submit(view):
    r = await data.task().submit_async(view, timeout_in_ms=config.LEAN_TIMEOUT_MS)
    return r


def feedback(result, max_errors=3, max_chars=1800):
    """Readable Lean feedback: each error with the offending source line."""
    if result.error_message == "Timeout":
        return "Lean timed out (proof search or evaluation took too long)."
    out = result.error_message or ""
    code_lines = (result.lean_code or "").splitlines()
    blocks, current = [], None
    for line in out.splitlines():
        m = re.match(r".*?\.lean:(\d+):(\d+):\s*(error|warning):\s*(.*)", line)
        if m:
            if current:
                blocks.append(current)
            current = None
            if m.group(3) == "error":
                ln = int(m.group(1))
                src = code_lines[ln - 1].strip() if 0 < ln <= len(code_lines) else ""
                current = [f"error at line {ln} `{src[:120]}`: {m.group(4)}"]
        elif current is not None:
            current.append(line)
    if current:
        blocks.append(current)
    if not blocks:
        return out[-max_chars:] or "Lean reported a problem but no error text."
    text = "\n\n".join("\n".join(b[:12]) for b in blocks[:max_errors])
    if len(blocks) > max_errors:
        text += f"\n\n(+{len(blocks) - max_errors} more errors)"
    return text[:max_chars]


async def check(pid, impl, helpers, proof):
    """Return dict(compiled, proved, stage, feedback, raw_error)."""
    reason = banned_reason(impl, helpers, proof)
    if reason:
        return {"compiled": False, "proved": False, "stage": "banned",
                "feedback": reason, "raw_error": reason}

    r1 = await _submit(_view(pid, impl, None, None))
    if not r1.compilation_ok:
        return {"compiled": False, "proved": False, "stage": "impl",
                "feedback": feedback(r1), "raw_error": r1.error_message}

    r2 = await _submit(_view(pid, impl, proof, helpers))
    if r2.compilation_ok and r2.correctness_ok:
        return {"compiled": True, "proved": True, "stage": "ok", "feedback": "", "raw_error": ""}
    fb = feedback(r2) if not r2.compilation_ok else "Proof incomplete: Lean reports `sorry`."
    return {"compiled": True, "proved": False, "stage": "proof",
            "feedback": fb, "raw_error": r2.error_message}
