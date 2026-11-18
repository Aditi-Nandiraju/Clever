"""Build the retrieval corpus  data/corpus.jsonl.

Entries (one JSON object per line):
  kind="lemma"    Mathlib theorem/lemma: full name + statement
  kind="example"  a solved problem rendered in the answer format:
                  - the 6 CLEVER samples
                  - DEV-split problems whose file ships a complete reference proof
                  - DEV-split problems our own runs solved (Lean-verified), via --add-solved

Leakage rule: nothing from a TEST-split problem ever enters the corpus.

Usage:
  python corpus.py                       # lemmas + samples + dev references
  python corpus.py --mathlib /path/to/Mathlib
  python corpus.py --add-solved          # append dev solutions found in logs/
"""
import argparse
import glob
import json
import re
from pathlib import Path

import config
import data

DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:protected\s+|nonrec\s+)*(theorem|lemma)\s+([^\s:({\[]+)")
NAMESPACE = re.compile(r"^namespace\s+(\S+)")
SECTION = re.compile(r"^(?:noncomputable\s+)?section\b\s*(\S*)")
END = re.compile(r"^end\b\s*(\S*)")


def find_sources(mathlib=None, core=None):
    """Return [(root_dir, folders)] for Mathlib, Lean core and Batteries."""
    from clever_bench.benchmark import get_clever_lean_project_path
    pkgs = Path(get_clever_lean_project_path()) / ".lake" / "packages"
    sources = []

    m = Path(mathlib or config.MATHLIB_DIR or pkgs / "mathlib" / "Mathlib")
    if not m.exists():
        raise SystemExit("Mathlib source not found. Run `lake exe cache get` in the CLEVER "
                         "Lean project first, or pass --mathlib /path/to/Mathlib")
    sources.append((m, config.MATHLIB_FOLDERS))

    c = core or config.CORE_DIR
    if c is None:
        try:
            import subprocess
            prefix = subprocess.run(["lean", "--print-prefix"], capture_output=True,
                                    text=True, timeout=60, cwd=get_clever_lean_project_path()).stdout.strip()
            c = Path(prefix) / "src" / "lean"
        except Exception:
            c = None
    if c and Path(c).exists():
        sources.append((Path(c), config.CORE_FOLDERS))
    else:
        print("warning: Lean core source not found; core lemmas (List.length_append, ...) "
              "will be missing. Pass --core <toolchain>/src/lean")

    if (pkgs / "batteries").exists():
        sources.append((pkgs / "batteries", config.BATTERIES_FOLDERS))
    return sources


def humanize(name):
    """List.length_append -> 'List length append' (helps the embedding model)."""
    return re.sub(r"[._]", " ", name)


def extract_lemmas(sources):
    out, seen = [], set()
    for mathlib_dir, folders in sources:
        for folder in folders:
            for f in sorted(Path(mathlib_dir, folder).rglob("*.lean")):
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
                stack = []          # entries: ("ns", name) or ("sec", name)
                in_comment = False
                for i, raw in enumerate(lines):
                    line = raw.strip()
                    if in_comment:
                        in_comment = "-/" not in line
                        continue
                    if line.startswith("/-"):
                        in_comment = "-/" not in line
                        continue
                    if m := NAMESPACE.match(line):
                        stack.append(("ns", m.group(1)))
                        continue
                    if m := SECTION.match(line):
                        stack.append(("sec", m.group(1)))
                        continue
                    if (m := END.match(line)) and stack:
                        stack.pop()
                        continue
                    if line.startswith("private"):
                        continue
                    m = DECL.match(line)
                    if not m:
                        continue
                    short = m.group(2)
                    prefix = ".".join(n for kind, n in stack if kind == "ns")
                    name = short if short.startswith("_root_.") or not prefix else f"{prefix}.{short}"
                    name = name.removeprefix("_root_.")
                    if name in seen:
                        continue
                    stmt_lines = []
                    for nxt in lines[i:i + 8]:
                        stmt_lines.append(nxt.strip())
                        if ":=" in nxt or nxt.strip().endswith(" where") or nxt.strip() == "|":
                            break
                    stmt = " ".join(stmt_lines).split(":=")[0].strip()
                    stmt = re.sub(r"^(?:@\[[^\]]*\]\s*)?(?:protected\s+|nonrec\s+)*(theorem|lemma)\s+\S+",
                                  f"theorem {name}", stmt)
                    if len(stmt) > 400:           # skip giant statements; they rarely help
                        continue
                    seen.add(name)
                    out.append({"id": name, "kind": "lemma", "source": str(f.relative_to(mathlib_dir)),
                                "query": f"{humanize(name)} {stmt}", "text": stmt})
    return out


def dev_reference_examples(dev_ids):
    """Dev problems whose file has a full reference implementation AND proof."""
    out = []
    for pid in dev_ids:
        p = data.benchmark().get_problem(pid)
        impl, proof = (p.implementation or "").strip(), (p.correctness_proof or "").strip()
        if not impl or impl == "sorry" or not proof or "sorry" in proof:
            continue
        lemmas = "\n\n".join(f"{l.statement.strip()}\n{l.proof.strip()}"
                             for l in p.correctness_helper_lemmas)
        out.append({"id": f"dev_ref_{pid}", "kind": "example", "source": "clever_dev_reference",
                    "blocked_for": [], "query": f"{p.problem_spec_nl}\n{p.problem_spec_formal_ground_truth}",
                    "text": data.render_example(p.problem_spec_nl, p.problem_spec_formal_ground_truth,
                                                p.implementation_signature, impl,
                                                p.correctness_theorem, proof, lemmas)})
    return out


def dev_solved_examples(dev_ids):
    """Dev problems solved (Lean-verified) in our own logs: one per problem."""
    out, done = [], set()
    for f in sorted(glob.glob(str(config.LOG_DIR / "*_dev.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r["proved"] or r["pid"] in done or r["pid"] not in dev_ids:
                continue
            v = data.prompt_view(r["pid"])
            done.add(r["pid"])
            out.append({"id": f"dev_solved_{r['pid']}", "kind": "example", "source": "own_dev_run",
                        "blocked_for": [], "query": f"{v['problem_spec_nl']}\n{v['problem_spec_formal_ground_truth']}",
                        "text": data.render_example(v["problem_spec_nl"], v["problem_spec_formal_ground_truth"],
                                                    v["implementation_signature"], r["impl"],
                                                    v["correctness_theorem"], r["proof"],
                                                    r.get("helper_lemmas") or "")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathlib", default=None, help="path to the Mathlib/ source folder")
    ap.add_argument("--core", default=None, help="path to <toolchain>/src/lean")
    ap.add_argument("--add-solved", action="store_true")
    args = ap.parse_args()

    split = data.make_split()
    dev, test = set(split["dev"]), set(split["test"])
    entries = extract_lemmas(find_sources(args.mathlib, args.core))
    n_lemmas = len(entries)
    entries += data.sample_examples()
    entries += dev_reference_examples(sorted(dev))
    if args.add_solved:
        entries += dev_solved_examples(dev)

    # Leakage guard: no example may come from a test problem.
    for e in entries:
        m = re.match(r"dev_(?:ref|solved)_(\d+)", e["id"])
        assert not (m and int(m.group(1)) in test), f"test problem leaked: {e['id']}"

    path = config.DATA_DIR / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_ex = len(entries) - n_lemmas
    print(f"wrote {path}: {n_lemmas} lemmas, {n_ex} examples")


if __name__ == "__main__":
    main()
