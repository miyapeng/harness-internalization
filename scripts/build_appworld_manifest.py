#!/usr/bin/env python3
"""Read official task IDs through the isolated AppWorld worker, then pin development splits."""
import argparse
from dataclasses import asdict
from pathlib import Path
from internalization.benchmarks.appworld import AppWorldConfig, AppWorldProcess
from internalization.benchmarks.appworld_manifest import build_manifest
from internalization.core.types import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/appworld.json"))
    parser.add_argument("--revision", required=True, help="Downloaded data revision/checksum")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    process = AppWorldProcess(AppWorldConfig.load(args.config), args.output.with_suffix(".worker"))
    try: source = process.call("datasets")
    finally: process.close()
    manifest = build_manifest(source["splits"], args.revision, seed=args.seed)
    write_json(args.output, {**asdict(manifest), "manifest_hash":manifest.fingerprint,
        "split_seed":args.seed, "package_version":source["package_version"],
        "official_split_sha256":source["split_sha256"], "official_splits":source["splits"],
        "allocation":"train/search only official train; dev/retirement from train+dev; test held out",
        "scenario_families_disjoint":True})


if __name__ == "__main__": main()
