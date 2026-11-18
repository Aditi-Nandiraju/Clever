"""Prompt construction and answer parsing.

The system prompt and the tagged answer format are adapted from the CLEVER
paper's few-shot prompts (Thakur et al., 2025, Figs. 17 and 19) and the
clever-prover code, so the baseline matches the paper's setup. Cite them.
"""
import re

SYSTEM = """You are an expert Lean 4 programmer and theorem prover (Lean 4 + Mathlib).
You are given a programming problem: a natural-language description, a formal
specification `problem_spec`, the signature of a function `implementation`, test
cases, and the theorem `correctness` stating that the implementation satisfies the
specification.

Your job:
1. Write the body of `implementation` (only the body: the signature is already given).
   It must be computable and pass the test cases.
2. Optionally write helper lemmas that your proof uses.
3. Write a complete proof of `correctness`.

Rules: never use `sorry`, `admit`, `axiom`, `native_decide`, `#exit` or other commands.
Prefer simple structural recursion that mirrors the specification; it makes the proof easier.

Answer in exactly this format and nothing else:
[GENERATED IMPLEMENTATION]
<implementation body>
[END]
[HELPER LEMMAS]
<zero or more complete Lean lemmas with proofs, or leave empty>
[END]
[PROOF]
by
  <tactics>
[END]"""


def problem_block(p):
    helpers = "\n\n".join(p.get("helper_definitions") or [])
    out = f"[NL DESCRIPTION]\n{p['problem_spec_nl']}\n"
    if helpers.strip():
        out += f"[HELPER DEFINITIONS (already in scope)]\n{helpers}\n"
    out += (f"[SPECIFICATION]\n{p['problem_spec_formal_ground_truth']}\n"
            f"[IMPLEMENTATION SIGNATURE]\n{p['implementation_signature']}\n"
            f"[TEST CASES]\n{p['test_cases_lean']}\n"
            f"[THEOREM STATEMENT]\n{p['correctness_theorem']}")
    return out


def context_block(examples, lemmas):
    parts = [f"### Example {i + 1}\n{e['text']}" for i, e in enumerate(examples)]
    if lemmas:
        parts.append("### Possibly useful lemmas (names are exact)\n" +
                     "\n".join(l["text"] for l in lemmas))
    return "\n\n".join(parts)


def repair_block(prev):
    out = ("\n\n### Your previous attempt was rejected by Lean\n"
           f"[GENERATED IMPLEMENTATION]\n{prev['impl'] or ''}\n[END]\n")
    if prev.get("helper_lemmas"):
        out += f"[HELPER LEMMAS]\n{prev['helper_lemmas']}\n[END]\n"
    out += (f"[PROOF]\n{prev['proof'] or ''}\n[END]\n"
            f"[LEAN FEEDBACK]\n{prev['feedback']}\n\n")
    if prev["stage"] == "proof":
        out += ("The implementation compiled and passed the tests; only the proof failed. "
                "Keep the implementation unless it makes the proof hard, and fix the proof.")
    elif prev["stage"] == "parse":
        out += "Your answer did not follow the required format. Use the exact tags."
    else:
        out += "The implementation itself failed to compile or failed a test. Fix it, then prove it."
    return out + " Answer in the same format."


def build_prompt(problem, examples, lemmas, prev=None):
    user = context_block(examples, lemmas)
    user += "\n\n### Now solve this problem\n" + problem_block(problem)
    if prev is not None:
        user += repair_block(prev)
    return SYSTEM, user


# ------------------------------------------------------------------ parsing
def _section(text, tag):
    m = re.search(rf"\[{tag}\]\s*(.*?)\s*\[END\]", text, re.S)
    if not m:                                   # tolerate a missing final [END]
        m = re.search(rf"\[{tag}\]\s*(.*?)\s*(?=\n\[[A-Z ]+\]|\Z)", text, re.S)
    if not m:
        return None
    body = re.sub(r"^```[a-zA-Z0-9]*\s*$", "", m.group(1), flags=re.M).strip()
    return body


def parse(text):
    """Return (impl, helper_lemmas, proof); impl/proof are None if missing."""
    impl = _section(text, "GENERATED IMPLEMENTATION")
    helpers = _section(text, "HELPER LEMMAS") or ""
    proof = _section(text, "PROOF")
    if impl:
        # models often repeat the signature: drop everything up to its ':='
        if re.match(r"\s*(?:def|partial def)\s+implementation\b", impl):
            impl = impl.split(":=", 1)[1].strip() if ":=" in impl else None
    if proof:
        proof = re.sub(r"^\s*theorem\s+correctness[\s\S]*?:=\s*", "", proof)  # repeated statement
        if not re.match(r"\s*(by\b|fun\b|⟨|\(|show\b|Exists\b)", proof):
            proof = "by\n" + proof
    return impl or None, helpers, proof or None
