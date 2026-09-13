#!/usr/bin/env python3
"""One-task, no-model engine/verifier smoke. Output is not an agent benchmark result."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from internalization.benchmarks.common import BenchmarkConfig
from internalization.benchmarks.isolated import IsolatedEnvironment
from internalization.core.types import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--action", help="Single probe action; no model call is made")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = replace(BenchmarkConfig.load(args.config), max_steps=1)
    default_action = {"terminalbench2":'{"action":"exec","command":"printf HI_ENGINE_SMOKE"}',
        "swebench_pro":'{"action":"final"}', "hotpotqa":'{"action":"final","answer":"","supporting_facts":[]}',
        "lawbench":""}[config.benchmark]
    args.output.mkdir(parents=True, exist_ok=False)
    env = IsolatedEnvironment(config,args.output/"environment")
    try:
        env.reset(args.task_id,0)
        result = env.step(args.action if args.action is not None else default_action)
        if not result.done: raise RuntimeError("Step-limit verifier did not finalize")
        report = {"benchmark":config.benchmark,"engine_smoke":"passed","model_used":False,
            "task_id":args.task_id,"reward":result.reward,"tool_calls":result.tool_calls}
        write_json(args.output/"report.json",report)
        print(json.dumps(report,indent=2))
    except Exception as exc:
        write_json(args.output/"report.json",{"benchmark":config.benchmark,"engine_smoke":"failed",
            "model_used":False,"error":str(exc)})
        raise
    finally: env.close()


if __name__ == "__main__": main()
