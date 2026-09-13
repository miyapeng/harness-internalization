#!/usr/bin/env python3
"""LawBench final predictions followed by complete-category official scoring."""
import argparse
import json
from pathlib import Path

from internalization.benchmarks.common import BenchmarkConfig, Catalog
from internalization.benchmarks.lawbench_final import evaluate
from internalization.core.manifest import TaskManifest
from internalization.core.serialization import harness_from_dict
from internalization.core.types import write_json
from internalization.harness.module import Harness
from internalization.training.teacher_backend import FrozenHFBackend


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/lawbench.json"))
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--state",type=Path)
    parser.add_argument("--partition",default="test")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=BenchmarkConfig.load(args.config)
    manifest=TaskManifest.load(args.manifest)
    if config.benchmark!=manifest.benchmark or manifest.benchmark!="lawbench":
        raise ValueError("LawBench manifest/config mismatch")
    checkpoint=str(args.checkpoint.resolve(strict=True))
    harness=Harness()
    if args.state:
        state=json.loads(args.state.read_text())
        if str(Path(state["checkpoint"]).resolve(strict=True))!=checkpoint: raise ValueError("Checkpoint/state mismatch")
        harness=harness_from_dict({"version":state["harness_version"],"modules":state["active_modules"]})
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"protocol.json",{"manifest_hash":manifest.fingerprint,"checkpoint":checkpoint,
        "harness_version":harness.version,"partition":args.partition,"seeds":args.seeds,"selection":False,
        "catalog_hash":Catalog(config.catalog,"lawbench").fingerprint})
    model=FrozenHFBackend(checkpoint,device=args.device,**config.model_options)
    result=evaluate(config,model,harness,manifest.partitions[args.partition],args.seeds,args.output)
    write_json(args.output/"aggregate.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
