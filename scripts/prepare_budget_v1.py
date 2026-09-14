#!/usr/bin/env python3
"""Build NEW source-constrained manifests; consumes local official assets only."""
import argparse
import json
from pathlib import Path
from internalization.benchmarks.budget_data import alfworld_rows,hotpot_rows,write_dataset
from internalization.core.sampling import BUDGET_SIZES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=('alfworld','webshop','hotpotqa'),required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--train',type=Path)
    p.add_argument('--distractor-dev',type=Path)
    p.add_argument('--webshop-assets',type=Path,help='Pinned full-resource options JSON; run using WebShop Python')
    p.add_argument('--split-seed',type=int,default=42)
    for name,size in BUDGET_SIZES.items(): p.add_argument('--'+name.replace('_','-')+'-size',type=int,default=size)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.benchmark=='alfworld':
        if args.data_root is None: p.error('--data-root required')
        rows=alfworld_rows(args.data_root)
    elif args.benchmark=='hotpotqa':
        if args.train is None or args.distractor_dev is None: p.error('--train and --distractor-dev required')
        rows=hotpot_rows(args.train,args.distractor_dev)
    else:
        if args.webshop_assets is None: p.error('--webshop-assets required')
        from internalization.benchmarks.webshop import export_tasks
        rows=export_tasks(json.loads(args.webshop_assets.read_text()))
    sizes={name:getattr(args,name+'_size') for name in BUDGET_SIZES}
    report=write_dataset(args.output,args.benchmark,args.revision,rows,args.split_seed,sizes)
    print(json.dumps({k:report[k] for k in ('benchmark','status','requested_sizes','manifest_hash')},indent=2))

if __name__=='__main__': main()
