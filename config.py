"""All experiment settings in one place.

Change values here, not inside the other files, so every run is reproducible.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"      # saved LLM replies: re-runs cost nothing
LOG_DIR = ROOT / "logs"         # one JSONL file per config x seed x split
RESULTS_DIR = ROOT / "results"  # CSV tables and PNG charts for the report
for _d in (DATA_DIR, CACHE_DIR, LOG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data split
SPLIT_SEED = 42
N_DEV = 20      # prompt tuning + mining retrieval examples
N_TEST = 40     # every reported number comes from these problems only

# ---------------------------------------------------------------- generator
# PROVIDER: "gemini" | "openai_compat" | "hf_local" | "mock"
#   gemini        -> Google AI Studio key in env var GEMINI_API_KEY
#   openai_compat -> any OpenAI-style endpoint (Groq, OpenRouter, OpenAI, Ollama, vLLM)
#                    set OPENAI_BASE_URL and OPENAI_API_KEY
#   hf_local      -> open model on a Colab/Kaggle GPU via transformers
#   mock          -> no model; used only by the smoke test
PROVIDER = "gemini"
MODEL = "gemini-2.5-flash"     # keep FIXED across all four configs
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 4096

# ---------------------------------------------------------------- experiment
K_ATTEMPTS = 5          # equal attempt budget for every config
N_FEWSHOT = 2           # examples in the prompt (fixed in C1/C2, retrieved in C3/C4)
N_LEMMAS = 8            # retrieved Mathlib lemmas (C3/C4 only)
LEAN_TIMEOUT_MS = 180_000
SEEDS = [0, 1, 2]
WORKERS = 2             # problems checked in parallel (each Lean run uses ~2-4 GB RAM)

CONFIGS = {
    "C1_baseline":  {"retrieval": False, "repair": False},
    "C2_repair":    {"retrieval": False, "repair": True},
    "C3_retrieval": {"retrieval": True,  "repair": False},
    "C4_both":      {"retrieval": True,  "repair": True},
}

# ---------------------------------------------------------------- retrieval
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Mathlib source folder; after `lake exe cache get` it lives inside the CLEVER
# Lean project at  <clever lean4 dir>/.lake/packages/mathlib/Mathlib
MATHLIB_DIR = None      # None = locate automatically (see corpus.py)
MATHLIB_FOLDERS = ["Data/List", "Data/Nat", "Data/Int", "Data/String",
                   "Data/Finset", "Data/Rat", "Data/Char", "Order/Basic"]
# Most everyday lemmas (List.length_append, Nat.sub_add_cancel, ...) live in Lean's
# core library and in Batteries, not Mathlib. Both are indexed too when found.
CORE_DIR = None         # None = <lean toolchain>/src/lean (via `lean --print-prefix`)
CORE_FOLDERS = ["Init/Data/List", "Init/Data/Nat", "Init/Data/Int",
                "Init/Data/String", "Init/Data/Char", "Init/Data/Option"]
BATTERIES_FOLDERS = ["Batteries/Data/List", "Batteries/Data/Nat",
                     "Batteries/Data/Int", "Batteries/Data/String"]
