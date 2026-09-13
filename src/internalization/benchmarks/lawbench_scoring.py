"""Run complete LawBench category scoring in its external dependency environment."""
import argparse
import json
from pathlib import Path
from .common import BenchmarkConfig
from .lawbench import OfficialLawBenchScorer
from ..core.types import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    scorer = OfficialLawBenchScorer(BenchmarkConfig(**request["config"]))
    write_json(args.response, scorer.score(request["predictions"], args.response.parent/"official"))


if __name__ == "__main__": main()
