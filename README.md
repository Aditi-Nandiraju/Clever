# Retrieval + Repair for Verified Code Generation on CLEVER

Code for the experiment: does **retrieving relevant lemmas/examples** and **repairing with Lean's
error messages** help an LLM produce Lean implementations with machine-checked correctness proofs
on the CLEVER benchmark (Thakur et al., NeurIPS 2025, arXiv:2505.13938)?

Task studied: CLEVER's *implementation certification* stage. Given the description, the
ground-truth spec, the signature, the tests and the `correctness` theorem, the model writes the
implementation, optional helper lemmas and the proof. Lean checks everything through the official
[`clever-bench`](https://github.com/trishullab/clever) package.

| Config | Prompt examples | After a failed attempt |
|---|---|---|
| `C1_baseline` | 2 fixed CLEVER samples | fresh sample, same prompt |
| `C2_repair` | 2 fixed CLEVER samples | previous attempt + Lean feedback added |
| `C3_retrieval` | 2 retrieved examples + 8 retrieved lemmas | fresh sample, same prompt |
| `C4_both` | 2 retrieved examples + 8 retrieved lemmas | previous attempt + Lean feedback added |

All configs get the **same budget of k = 5 model calls per problem**, so any difference comes from
the prompt content, not from extra attempts.

## Files

| File | What it does |
|---|---|
| `config.py` | every setting (model, k, seeds, split sizes, timeouts) |
| `data.py` | loads CLEVER via `clever-bench`, fixed dev/test split, few-shot examples, leakage guards |
| `corpus.py` | builds `data/corpus.jsonl`: Lean core + Batteries + Mathlib lemmas, solved examples |
| `retriever.py` | embedding retriever (falls back to TF-IDF if the model cannot load) |
| `prompts.py` | system prompt, prompt building, repair block, answer parser |
| `llm.py` | Gemini / OpenAI-compatible (Groq, OpenRouter, Ollama...) / local HF model, with a reply cache |
| `checker.py` | Lean checking through `clever-bench`, cheat detection, readable error feedback |
| `solve.py` | the generate -> check -> repair loop for one problem |
| `run_experiment.py` | runs configs x seeds x problems, resumable, logs to `logs/` |
| `errors.py` | classifies failed attempts into error types |
| `analyze.py` | tables, charts, paired significance tests -> `results/` |
| `smoke_test.py` | offline test of the whole pipeline with a fake model and fake Lean |

## Setup

Lean 4 + Mathlib needs about 8 GB RAM and 10 GB disk. Google Colab (free) works if your laptop is
smaller. The first install takes 15-40 minutes, mostly downloading prebuilt Mathlib.

```bash
# 1. Python packages
pip install -r requirements.txt

# 2. Lean 4 + the CLEVER Lean project (Linux / Colab; on Mac/Windows see the CLEVER README)
clever-bench-install
#   -> installs elan + Lean, then runs `lake exe cache get && lake build` inside
#      the clever_bench package. Check it ends without errors.

# 3. Model key (pick ONE provider, then set PROVIDER and MODEL in config.py)
export GEMINI_API_KEY=...                     # PROVIDER = "gemini"
# or: export OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_API_KEY=...
#                                              # PROVIDER = "openai_compat"

# 4. Check the plumbing (no key or Lean needed)
python smoke_test.py                           # expect 26/26 checks passed
```

**Colab:** open `clever_colab.ipynb` (T4 GPU runtime). It installs Lean + Mathlib and Ollama
(`qwen2.5-32k`) and keeps logs/cache/results in Google Drive. In Colab, prefix shell commands with `!`. Store the key with Colab's *Secrets* panel, not in the
notebook. Never commit a key to GitHub.

## Run order

```bash
# A. Build the retrieval corpus (needs step 2 above, so Mathlib source exists)
python corpus.py                       # ~13k lemmas + 6 samples + dev reference solutions
python retriever.py 3                  # eyeball: retrieved items for problem 3

# B. Smoke test with the real model + real Lean on 3 dev problems
python run_experiment.py --configs C1_baseline --split dev --limit 3 --seeds 0
#    Open logs/C1_baseline_s0_dev.jsonl and read a few records before going further.

# C. Dev runs: tune the prompt / error rules here, never on test
python run_experiment.py --split dev --seeds 0
python analyze.py --split dev
python corpus.py --add-solved          # optional: add Lean-verified dev solutions as examples

# D. Test runs (all reported numbers). Freeze config.py and prompts.py first.
python run_experiment.py --configs C1_baseline C2_repair       # core: RQ1
python run_experiment.py --configs C3_retrieval C4_both        # extension: RQ2, RQ3
python analyze.py
```

Runs can be stopped and restarted: finished problems are skipped. LLM replies are cached in
`cache/`, so re-running costs nothing. Delete `cache/` only if you change the model.

**Rough cost.** 40 problems x 4 configs x 3 seeds x up to 5 attempts = at most 2,400 model calls
and up to 4,800 Lean checks (each takes a few seconds to a minute or more). If time is short:
run C1 + C2 first, use `--seeds 0 1`, or lower `N_TEST`. Say in the report what you cut.

## Outputs (`results/test/`)

| File | Use in the report |
|---|---|
| `summary.csv` | main table: compile % and proved % per config, mean +/- std over seeds |
| `curve.png` / `curve.csv` | proved % vs attempts used, one line per config |
| `errors.png` / `errors.csv` | why attempts failed, per config |
| `repair_effect.csv` | after each failed attempt: proved / advanced / same error / different error |
| `paired_tests.csv` | exact McNemar test on (problem, seed) pairs, e.g. C1 vs C2 |
| `per_problem.csv` | which problems each config solved (for case studies) |

Compare with the paper's Table 1, *Impl Cert. Proved* column (0.6-8.7% on all 161 problems with a
600 s budget). Your numbers come from a 40-problem subset and a 5-call budget, so treat that as a
rough reference, not a head-to-head comparison.

## Things found in the CLEVER files that matter for the report

1. **Reference solutions ship inside 44 of the 161 problem files**, and the official
   `PROOF_GENERATION` view does not hide the `implementation` field. `data.py` builds prompts from
   an allow-list, and `smoke_test.py` checks that no reference implementation reaches any prompt.
   Logs carry `has_reference_impl`, so you can report results separately for those problems
   (public solutions also mean possible training-data contamination).
2. **Sample 3 (Fibonacci) is a duplicate of test problem 55.** Its docstring says so, and the
   authors exclude it from few-shot prompts. Here it is never a fixed example, and the retriever
   never returns it for problem 55.
3. **The official checker only looks for `sorry`.** `#exit`, new `axiom`s, `native_decide`,
   `implemented_by`, or a redefined tactic macro could make Lean accept a file without a real
   proof. `checker.py` rejects these before calling Lean.
4. **Basic lemmas live in Lean core, not Mathlib** (`List.length_append`, `Nat.sub_add_cancel`).
   The corpus indexes Lean core and Batteries as well as Mathlib.

## Known limitations (state them in the report)

- 40-problem subset and 3 seeds: small differences will not be statistically significant.
- Implementation certification only; spec generation and end-to-end success are not tested.
- One generator model.
- Prompt length differs between configs: the fixed samples are longer than most retrieved items.
  Report `mean_prompt_tokens` from `summary.csv` next to the results.
- `errors.py` uses regex rules: validate them by reading real errors before the final analysis.
- The cheat filter is conservative: it may reject a legitimate answer that has a `#` command or
  the word `macro`, `syntax` or `notation`, even inside a comment. Such attempts are logged as
  `banned`, so count them and report the number.

## Credits

Benchmark, problem files, Lean checking and the answer-tag format: CLEVER and clever-prover
(Thakur et al., 2025), MIT license. Lemmas: Lean 4 core, Batteries and Mathlib (Apache 2.0).
