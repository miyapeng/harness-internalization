#!/usr/bin/env python3
"""LawBench final predictions followed by complete-category official scoring."""
import argparse
import json
from pathlib import Path

from internalization.benchmarks.common import BenchmarkConfig, Catalog
from internalization.benchmarks.lawbench_final import evaluate
from internalization.core.manifest import TaskManifest
from internalization.core.accepted_state import add_evaluation_agent_arguments, evaluation_agent
from internalization.core.types import write_json
from internalization.harness.revision import HarnessRevision
from internalization.training.teacher_backend import FrozenHFBackend


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/lawbench.json"))
    add_evaluation_agent_arguments(parser)
    parser.add_argument("--partition",default="test")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=BenchmarkConfig.load(args.config)
    manifest=TaskManifest.load(args.manifest)
    if config.benchmark!=manifest.benchmark or manifest.benchmark!="lawbench":
        raise ValueError("LawBench manifest/config mismatch")
    agent=evaluation_agent(args,manifest)
    checkpoint,harness=agent.checkpoint,agent.harness
    if isinstance(harness,HarnessRevision):
        raise ValueError("Unsupported: native complete-category LawBench evaluator does not yet execute code revisions; no empty-Harness fallback")
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"protocol.json",{"manifest_hash":manifest.fingerprint,"checkpoint":checkpoint,
        "harness_version":harness.version,"agent":agent.to_dict(),"partition":args.partition,"seeds":args.seeds,"selection":False,
        "catalog_hash":Catalog(config.catalog,"lawbench").fingerprint})
    model=FrozenHFBackend(checkpoint,device=args.device,**config.model_options)
    result=evaluate(config,model,harness,manifest.partitions[args.partition],args.seeds,args.output)
    write_json(args.output/"aggregate.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
