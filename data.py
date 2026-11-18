"""Load CLEVER through the official `clever-bench` package and fix the dev/test split.

IMPORTANT (leakage): 44 of the 161 CLEVER problem files contain the authors'
reference implementation (and a few contain full proofs), and the official
PROOF_GENERATION view does NOT hide them. This module therefore builds prompts
from an explicit ALLOW-LIST of fields and never exposes `implementation` or
`correctness_proof` of an evaluation problem.
"""
import json
import random
from functools import lru_cache

from clever_bench.benchmark import Benchmark
from clever_bench.task import ProblemViewTask, TaskComponent

import config

# Fields the model is allowed to see for an evaluation problem.
PROMPT_FIELDS = ("problem_spec_nl", "problem_spec_formal_ground_truth",
                 "implementation_signature", "test_cases_lean",
                 "correctness_theorem", "helper_definitions")


@lru_cache(maxsize=1)
def benchmark():
    b = Benchmark()               # src/lean4/human_eval, 161 problems
    b.load_all()
    return b


@lru_cache(maxsize=1)
def samples():
    """The 6 solved sample problems shipped with CLEVER (not part of HumanEval)."""
    s = Benchmark(is_sample=True)
    s.load_all()
    return s


@lru_cache(maxsize=1)
def task():
    """Official task object; its submit_async runs Lean on a problem view."""
    return ProblemViewTask(benchmark(), TaskComponent.PROOF_GENERATION,
                           report_dir=str(config.ROOT / ".logs" / "reports"))


def problem_ids():
    return [p.problem_id for p in benchmark().problems]


def prompt_view(pid):
    """Allow-listed fields of one evaluation problem, as a plain dict."""
    view = task().get_view(pid)
    d = {f: getattr(view, f) for f in PROMPT_FIELDS}
    d["pid"] = pid
    d["has_reference_impl"] = has_reference(pid)
    return d


def has_reference(pid):
    """True if the problem file ships a reference implementation (used only for
    analysis and contamination checks, never shown to the model)."""
    impl = benchmark().get_problem(pid).implementation or ""
    return impl.strip() not in ("", "sorry")


def make_split():
    """Seeded dev/test split, saved once and never regenerated."""
    path = config.DATA_DIR / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    ids = problem_ids()
    rng = random.Random(config.SPLIT_SEED)
    rng.shuffle(ids)
    split = {"seed": config.SPLIT_SEED,
             "dev": sorted(ids[:config.N_DEV]),
             "test": sorted(ids[config.N_DEV:config.N_DEV + config.N_TEST])}
    path.write_text(json.dumps(split, indent=2))
    return split


def render_example(nl, spec, impl_sig, impl, theorem, proof, helper_lemmas=""):
    """One solved problem, written in exactly the format the model must answer in."""
    out = (f"[NL DESCRIPTION]\n{nl}\n[SPECIFICATION]\n{spec}\n"
           f"[IMPLEMENTATION SIGNATURE]\n{impl_sig}\n"
           f"[THEOREM STATEMENT]\n{theorem}\n"
           f"--- solution ---\n[GENERATED IMPLEMENTATION]\n{impl}\n[END]\n")
    if helper_lemmas.strip():
        out += f"[HELPER LEMMAS]\n{helper_lemmas}\n[END]\n"
    out += f"[PROOF]\n{proof}\n[END]"
    return out


# Sample 3 (Fibonacci) is the same task as HumanEval problem 55; its docstring
# says so and the authors exclude it from few-shot prompts. Never show it for 55.
SAMPLE_DUPLICATES = {3: [55]}


def sample_examples():
    """The 6 CLEVER samples as rendered few-shot examples."""
    out = []
    for p in samples().problems:
        lemmas = "\n\n".join(f"{l.statement.strip()}\n{l.proof.strip()}"
                             for l in p.correctness_helper_lemmas)
        out.append({
            "id": f"sample_{p.problem_id}", "kind": "example", "source": "clever_sample",
            "blocked_for": SAMPLE_DUPLICATES.get(p.problem_id, []),
            "query": f"{p.problem_spec_nl}\n{p.problem_spec_formal_ground_truth}",
            "text": render_example(p.problem_spec_nl, p.problem_spec_formal_ground_truth,
                                   p.implementation_signature, p.implementation,
                                   p.correctness_theorem, p.correctness_proof, lemmas),
        })
    return out


# The paper's few-shot prompts use 1-2 of these samples. C1/C2 always get the
# same two; C3/C4 get the two retrieved ones instead (same count, fair comparison).
# Sample 3 is excluded (duplicate of problem 55, see SAMPLE_DUPLICATES).
FIXED_FEWSHOT_IDS = ("sample_1", "sample_2")   # list recursion + integer arithmetic


def fixed_fewshot():
    by_id = {e["id"]: e for e in sample_examples()}
    return [by_id[i] for i in FIXED_FEWSHOT_IDS][:config.N_FEWSHOT]
