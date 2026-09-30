import os

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "data/third_party/agentdojo/src")

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.benchmark import run_task_without_injection_tasks
from agentdojo.task_suite.load_suites import get_suite

from agentdojo.logging import LOGGER_STACK, OutputLogger
from agentdojo_llm import RawIDLLM
from agentdojo_bringup import SYSTEM

def run_one(a, suite, user_task_id, arm, logdir):

    LOGGER_STACK.set([])
    trace = []
    llm = RawIDLLM(a.base_url, a.model, a.tokenizer, family=a.family, arm=arm,
                   max_tokens=a.max_tokens, trace=trace)
    pipeline = AgentPipeline([SystemMessage(SYSTEM), InitQuery(), llm,
                              ToolsExecutionLoop([ToolsExecutor(), llm])])
    pipeline.name = f"benign-{a.family}-arm{arm}"
    ut = suite.user_tasks[user_task_id]
    err = None
    try:
        with OutputLogger(str(logdir), live=None):
            utility, _ = run_task_without_injection_tasks(
                suite, pipeline, ut, logdir=logdir, force_rerun=True,
                benchmark_version=a.version)
        util = bool(utility)
    except Exception as e:
        util = False
        err = f"{type(e).__name__}: {str(e)[:200]}"
    return {"suite": suite.name, "user_task": user_task_id, "condition": arm,
            "utility": util, "turns": len(trace),
            "truncated": any(t["finish_reason"] == "length" for t in trace),
            "context_exhausted": any(t.get("context_exhausted") for t in trace),
            "untrusted_spans": sum(t["n_untrusted_spans"] for t in trace),
            "verified_turns": sum(1 for t in trace if t["prompt_ids_verified"]),
            "turn_errors": sum(1 for t in trace if t["error"]),
            "error": err}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--family", default="qwen")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--arms", default="reserved,split")
    ap.add_argument("--suites", default="banking,slack,travel")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-prefix", default="data/results/agentdojo_utility_qwen")
    a = ap.parse_args()

    arms = a.arms.split(",")
    suites = {s: get_suite(a.version, s) for s in a.suites.split(",")}
    logdir = Path("logs/agentdojo_utility")
    logdir.mkdir(parents=True, exist_ok=True)

    for rep in range(1, a.repeats + 1):
        rows, t0 = [], time.time()
        jobs = [(s, ut, arm) for s, suite in suites.items()
                for ut in suite.user_tasks for arm in arms]
        print(f"repeat {rep}: {len(jobs)} runs "
              f"({len(suites)} suites x arms {arms})", flush=True)
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(run_one, a, suites[s], ut, arm, logdir)
                    for s, ut, arm in jobs]
            for k, f in enumerate(as_completed(futs)):
                rows.append(f.result())
                if (k + 1) % 25 == 0:
                    print(f"  {k+1}/{len(futs)}  ({time.time()-t0:.0f}s)", flush=True)
        out = f"{a.out_prefix}_r{rep}.jsonl"
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        for arm in arms:
            sub = [r for r in rows if r["condition"] == arm]
            u = sum(1 for r in sub if r["utility"]) / len(sub) * 100
            print(f"  arm {arm}: utility {u:.1f}%  (n={len(sub)}, "
                  f"errors {sum(1 for r in sub if r['error'])})", flush=True)
        print(f"wrote {out}", flush=True)

if __name__ == "__main__":
    main()
