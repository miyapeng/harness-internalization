#!/usr/bin/env python3
"""No-model smoke check of the installed official package, data, REPL, and grader."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from internalization.benchmarks.appworld import AppWorldConfig, AppWorldProcess, AppWorldEnvironment
from internalization.core.types import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/appworld.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = replace(AppWorldConfig.load(args.config), max_steps=1)
    process = AppWorldProcess(config, args.output / "datasets")
    try: datasets = process.call("datasets")
    finally: process.close()
    environment = AppWorldEnvironment(config, args.output / "episode", training=True)
    try:
        environment.reset(datasets["splits"]["train"][0], 0)
        outcome = environment.step('print("APPWORLD_SMOKE_OK")')
        execution_output = outcome.observation.rsplit("[Execution output]\n", 1)[-1].strip()
        if not outcome.action_valid or execution_output != "APPWORLD_SMOKE_OK":
            raise RuntimeError("Official AppWorld interpreter smoke failed")
        report = {"package_version":datasets["package_version"],
            "task_counts":{k:len(v) for k,v in datasets["splits"].items()},
            "split_sha256":datasets["split_sha256"], "engine_and_grader":"passed",
            "smoke_task_success":outcome.success, "benchmark_agent_evaluation":False}
        write_json(args.output / "report.json", report)
        print(json.dumps(report, indent=2))
    finally: environment.close()


if __name__ == "__main__": main()
