#!/usr/bin/env python3
"""Import downloaded data; published evaluation sources always remain held out."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from internalization.benchmarks.common import BENCHMARKS
from internalization.benchmarks.importers import import_tasks, make_catalog_manifest
from internalization.core.types import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Published held-out tasks/data")
    parser.add_argument("--train-source", type=Path, help="Independent train tasks; HotpotQA official train JSON")
    parser.add_argument("--revision", required=True, help="Dataset commit/version; content hashes are also recorded")
    parser.add_argument("--image-lock", type=Path, help="Pro instance ID -> {tag, digest}")
    parser.add_argument("--cohort-size", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.image_lock.read_text()) if args.image_lock else None
    test = import_tasks(args.benchmark, args.source, image_lock=lock)
    train = import_tasks(args.benchmark, args.train_source, split="train", image_lock=lock) if args.train_source else ()
    catalog, manifest = make_catalog_manifest(args.benchmark, args.revision, test, train, cohort_size=args.cohort_size)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"catalog.json", catalog)
    write_json(args.output/"manifest.json", asdict(manifest))
    print(json.dumps({"benchmark":args.benchmark, "counts":{k:len(v) for k,v in manifest.partitions.items()},
        "training_ready":bool(train), "manifest_hash":manifest.fingerprint}, indent=2))


if __name__ == "__main__": main()
