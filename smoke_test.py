"""Offline smoke test: runs the WHOLE pipeline with a fake LLM and a fake Lean.

It checks the plumbing (prompts, parsing, leakage guards, cheat detection,
repair loop, logging, resume, analysis) without an API key or a Lean install.
It proves nothing about real results. Run it after any code change:

    python smoke_test.py
"""
import asyncio
import random
import re
import shutil
import tempfile
from pathlib import Path

import config

TMP = Path(tempfile.mkdtemp(prefix="clever_rr_smoke_"))
config.LOG_DIR = TMP / "logs"
config.RESULTS_DIR = TMP / "results"
config.LOG_DIR.mkdir()
config.RESULTS_DIR.mkdir()
config.PROVIDER = "mock"
config.SEEDS = [0, 1]

import analyze
import checker
import data
import llm
import prompts
import run_experiment
from clever_bench.task import ValidationResult

ok = lambda cond, msg: print(("PASS  " if cond else "FAIL  ") + msg) or cond
results = []

# ------------------------------------------------------------------ 1. data + leakage
split = data.make_split()
results.append(ok(not set(split["dev"]) & set(split["test"]), "dev and test splits are disjoint"))
results.append(ok(len(data.problem_ids()) == 161, "161 CLEVER problems load"))
leaks = 0
for pid in data.problem_ids():
    ref = (data.benchmark().get_problem(pid).implementation or "").strip()
    if ref and ref != "sorry":
        _, user = prompts.build_prompt(data.prompt_view(pid), data.fixed_fewshot(), [])
        leaks += ref in user
results.append(ok(leaks == 0, f"no reference implementation appears in any prompt (44 files have one)"))
results.append(ok(all(e["id"] != "sample_3" for e in data.fixed_fewshot()),
                  "sample 3 (duplicate of problem 55) is not a fixed few-shot example"))

# ------------------------------------------------------------------ 2. parsing
p_impl, p_help, p_proof = prompts.parse(
    "Sure!\n[GENERATED IMPLEMENTATION]\n```lean\ndef implementation (x y: Int) : Int :=\nx + y\n```\n[END]\n"
    "[HELPER LEMMAS]\n[END]\n[PROOF]\nunfold problem_spec\nsimp\n[END]")
results.append(ok(p_impl == "x + y", "parser strips code fences and a repeated signature"))
results.append(ok(p_proof.startswith("by\n"), "parser adds `by` to a bare tactic proof"))
results.append(ok(prompts.parse("no tags here")[0] is None, "parser returns None on a malformed answer"))

# ------------------------------------------------------------------ 3. cheat detection
for bad in ["by sorry", "by\n  native_decide", "x + y\n#exit", "axiom cheat : False", "by admit",
            "x\n@[implemented_by foo] def g := 1", "macro \"simp\" : tactic => `(tactic| sorry)"]:
    results.append(ok(checker.banned_reason(bad) is not None, f"rejects {bad.splitlines()[-1][:40]!r}"))
results.append(ok(checker.banned_reason("by\n  unfold problem_spec\n  simp [List.sum_cons]") is None,
                  "accepts an ordinary proof"))

# ------------------------------------------------------------------ 4. Lean feedback formatting
fake = ValidationResult("3", False, False, False,
                        "temp/ab12.lean:5:2: error: unsolved goals\nx y : Int\n⊢ x + y - x - y = 0\n"
                        "temp/ab12.lean:7:0: warning: unused variable\n",
                        "l1\nl2\nl3\nl4\n  simp [foo]\nl6\nl7")
fb = checker.feedback(fake)
results.append(ok("line 5 `simp [foo]`" in fb and "unsolved goals" in fb and "warning" not in fb,
                  "feedback shows the failing source line and drops warnings"))

# ------------------------------------------------------------------ 5. full pipeline, mocked
def mock_llm(system, user, tag):
    rng = random.Random(tag)
    repairing = "[LEAN FEEDBACK]" in user
    retrieved = "Possibly useful lemmas" in user
    r = rng.random()
    if r < 0.1:
        return "I think the answer is 42."                      # malformed
    impl = "BAD" if r < 0.35 and not repairing else "0"
    good = rng.random() < (0.15 + 0.25 * repairing + 0.1 * retrieved)
    proof = "by\n  GOOD" if good else "by\n  simp"
    return f"[GENERATED IMPLEMENTATION]\n{impl}\n[END]\n[HELPER LEMMAS]\n[END]\n[PROOF]\n{proof}\n[END]"


async def mock_submit(view):
    code = f"{view.implementation}\n{view.correctness_proof}"
    if "BAD" in (view.implementation or ""):
        return ValidationResult(view.problem_id, False, False, False,
                                "temp/x.lean:3:0: error: unknown identifier 'BAD'", "a\nb\nBAD")
    if view.correctness_proof is None:                         # implementation-only check
        return ValidationResult(view.problem_id, False, False, True, "Sorries found", code)
    if "GOOD" in view.correctness_proof:
        return ValidationResult(view.problem_id, False, True, True, "No sorries found", code)
    return ValidationResult(view.problem_id, False, False, False,
                            "temp/x.lean:2:2: error: unsolved goals\n⊢ False", "a\n  simp")

llm.MOCK_FN = mock_llm
checker._submit = mock_submit


class Args:
    configs = list(config.CONFIGS)
    split = "dev"
    seeds = config.SEEDS
    limit = 8
    workers = 4


asyncio.run(run_experiment.run(Args()))
logs = sorted(config.LOG_DIR.glob("*.jsonl"))
results.append(ok(len(logs) == 8, f"one log per config x seed ({len(logs)} files)"))

# resume: running again must not add rows
before = sum(1 for f in logs for _ in open(f))
asyncio.run(run_experiment.run(Args()))
after = sum(1 for f in logs for _ in open(f))
results.append(ok(before == after, "re-running skips finished problems (resume works)"))

import json
rows = [json.loads(l) for f in logs for l in open(f)]
c1 = [r for r in rows if r["config"] == "C1_baseline"]
results.append(ok(all(r["attempt"] <= config.K_ATTEMPTS for r in rows), "never exceeds the attempt budget k"))
results.append(ok(all(not r["retrieved"] or r["retrieved"][0]["id"].startswith("sample") for r in c1),
                  "C1 uses only the fixed few-shot examples"))

import sys
sys.argv = ["analyze.py", "--split", "dev"]
analyze.main()
out = config.RESULTS_DIR / "dev"
for name in ["summary.csv", "curve.csv", "curve.png", "errors.csv", "errors.png", "paired_tests.csv"]:
    results.append(ok((out / name).exists(), f"analysis wrote {name}"))

print(f"\n{sum(results)}/{len(results)} checks passed   (outputs in {TMP})")
if not all(results):
    raise SystemExit(1)
