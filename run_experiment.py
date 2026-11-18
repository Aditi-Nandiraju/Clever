"""Run configurations x seeds x problems and log every attempt to logs/.

Examples
  python run_experiment.py --configs C1_baseline --split dev --limit 3 --seeds 0   # smoke test
  python run_experiment.py --configs C1_baseline C2_repair                           # RQ1 (core)
  python run_experiment.py --configs C3_retrieval C4_both                            # RQ2/RQ3

Safe to stop and restart: finished problems are skipped, half-finished ones redone.
"""
import argparse
import asyncio
import json
import time

import config
import data
from solve import solve


def completed(path):
    """Keep only problems that finished (proved, or used all k attempts);
    rewrite the log without half-finished problems so they are redone cleanly."""
    if not path.exists():
        return set()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["pid"], []).append(r)
    done = {pid for pid, rs in by_pid.items()
            if any(r["proved"] for r in rs) or len(rs) >= config.K_ATTEMPTS}
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if r["pid"] in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return done


async def run(args):
    split = data.make_split()
    ids = split[args.split][:args.limit] if args.limit else split[args.split]
    retriever = None
    if any(config.CONFIGS[c]["retrieval"] for c in args.configs):
        from retriever import Retriever
        retriever = Retriever()
        print("retriever:", retriever.backend)

    for name in args.configs:
        cfg = config.CONFIGS[name]
        for seed in args.seeds:
            path = config.LOG_DIR / f"{name}_s{seed}_{args.split}.jsonl"
            todo = [pid for pid in ids if pid not in completed(path)]
            print(f"\n== {name} seed {seed} ({args.split}): {len(todo)} problems to run")
            sem = asyncio.Semaphore(args.workers)
            with open(path, "a", encoding="utf-8") as f:
                async def one(pid):
                    async with sem:
                        await solve(data.prompt_view(pid), name, cfg, seed, retriever, f)
                t0 = time.time()
                await asyncio.gather(*(one(pid) for pid in todo))
                print(f"== {name} seed {seed} done in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(config.CONFIGS), choices=list(config.CONFIGS))
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
