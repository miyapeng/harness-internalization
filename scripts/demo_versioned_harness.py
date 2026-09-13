#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from internalization.revision_demo import run_demo
from internalization.outer_loop import LoopConfig

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("configs/versioned_demo.json"))
    parser.add_argument("--scenario",choices=("mixed","tool_only","unsupported","rollback","attribution_failed"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    loop=cfg["loop"];loop["seeds"]=tuple(loop["seeds"])
    result=run_demo(args.output,args.scenario or cfg["scenario"],LoopConfig(**loop))
    print(json.dumps({"checkpoint":result["checkpoint"],"harness_revision":result["harness_revision"]["version"],
                      "cycles":[{"reason":e["reason"],"model":e["model_decision"],"behavior":e["module_decision"]} for e in result["archive"]]},indent=2))

if __name__=="__main__": main()
