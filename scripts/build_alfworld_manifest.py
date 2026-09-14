#!/usr/bin/env python3
"""Split actual ALFWorld game paths, never placeholder parquet row numbers."""
import argparse
import random
from pathlib import Path

from internalization.manifest import TaskManifest
from internalization.core.manifest import loop_cohort_names
from internalization.records import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="Directory containing train/valid_seen/valid_unseen")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", required=True, help="Downloaded dataset version or checksum")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cohort-size", type=int, default=60)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--legacy-modules", action="store_true")
    args = p.parse_args()
    if args.cohort_size < 30: raise ValueError("Use >=30 tasks per held-out cohort")
    root = args.data_root.resolve(strict=True)
    train = sorted(str(p) for p in (root / "train").rglob("game.tw-pddl"))
    random.Random(args.seed).shuffle(train)
    names = loop_cohort_names(args.cycles,versioned=not args.legacy_modules)
    if len(train) <= args.cohort_size * len(names): raise ValueError("Insufficient real train game files")
    partitions = {name: train[i*args.cohort_size:(i+1)*args.cohort_size] for i, name in enumerate(names)}
    partitions["train"] = train[len(names)*args.cohort_size:]
    for split in ("valid_seen", "valid_unseen"):
        partitions["test_" + split] = sorted(str(p) for p in (root / split).rglob("game.tw-pddl"))
    manifest = TaskManifest("alfworld", args.revision, {k: tuple(v) for k, v in partitions.items()})
    manifest.validate_loop(args.cycles,versioned=not args.legacy_modules,cohort_minimum=30)
    write_json(args.output, {"benchmark": manifest.benchmark, "environment_revision": manifest.environment_revision,
        "partitions": partitions, "split_seed": args.seed, "manifest_hash": manifest.fingerprint,
        "loop_cycles":args.cycles,"loop_mode":"legacy_modules" if args.legacy_modules else "versioned"})


if __name__ == "__main__": main()
