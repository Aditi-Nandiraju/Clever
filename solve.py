"""Solve one problem under one configuration: generate -> check -> (repair) -> ...

All four configurations share this loop and the same budget of K_ATTEMPTS
model calls. They differ only in the prompt:
  retrieval=False : the fixed few-shot examples, no lemmas
  retrieval=True  : retrieved examples + retrieved lemmas
  repair=False    : every attempt is a fresh sample of the same prompt
  repair=True     : attempts 2..k also contain the previous attempt + Lean feedback
"""
import asyncio
import json
import time

import config
import data
import llm
from checker import check
from prompts import build_prompt, parse


async def solve(problem, cfg_name, cfg, seed, retriever, log_file):
    pid = problem["pid"]
    if cfg["retrieval"]:
        examples = retriever.query(problem, config.N_FEWSHOT, "example")
        lemmas = retriever.query(problem, config.N_LEMMAS, "lemma")
    else:
        examples, lemmas = data.fixed_fewshot(), []

    prev = None
    for attempt in range(1, config.K_ATTEMPTS + 1):
        system, user = build_prompt(problem, examples, lemmas, prev if cfg["repair"] else None)
        tag = f"{cfg_name}/s{seed}/p{pid}/a{attempt}"
        t0 = time.time()
        text, usage, cached = await asyncio.to_thread(llm.generate, system, user, tag)
        t_llm = time.time() - t0

        impl, helpers, proof = parse(text)
        if impl is None or proof is None:
            res = {"compiled": False, "proved": False, "stage": "parse",
                   "feedback": "Answer did not follow the required format "
                               "([GENERATED IMPLEMENTATION] ... [END] and [PROOF] ... [END]).",
                   "raw_error": "parse_fail"}
        else:
            res = await check(pid, impl, helpers, proof)

        record = {
            "config": cfg_name, "seed": seed, "pid": pid, "attempt": attempt,
            "compiled": res["compiled"], "proved": res["proved"], "stage": res["stage"],
            "feedback": res["feedback"], "raw_error": (res["raw_error"] or "")[-4000:],
            "impl": impl, "helper_lemmas": helpers, "proof": proof,
            "retrieved": [{"id": x["id"], "score": round(x.get("score", 0), 4)} for x in examples + lemmas],
            "prompt_chars": len(system) + len(user), "tokens_in": usage.get("in"),
            "tokens_out": usage.get("out"), "cached": cached,
            "sec_llm": round(t_llm, 1), "sec_total": round(time.time() - t0, 1),
            "has_reference_impl": problem["has_reference_impl"],
            "retriever": retriever.backend if (cfg["retrieval"] and retriever) else None,
            "model": f"{config.PROVIDER}:{config.MODEL}",
        }
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()
        print(f"  {cfg_name} s{seed} p{pid:<3} a{attempt}: {res['stage']}"
              f"{'  PROVED' if res['proved'] else ''}")
        if res["proved"]:
            return True
        prev = {"impl": impl, "helper_lemmas": helpers, "proof": proof,
                "stage": res["stage"], "feedback": res["feedback"]}
    return False
