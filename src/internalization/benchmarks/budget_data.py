"""Exact-size, source-constrained grouped splits. Never rewrites an existing manifest."""
from collections import Counter, defaultdict, deque
from pathlib import Path
import json
import random

from .common import sha256
from ..core.manifest import TaskManifest
from ..core.sampling import BUDGET_SIZES
from ..core.types import digest, write_json


def grouped_selection(rows, count, seed):
    groups = defaultdict(list)
    for row in rows: groups[row["group_id"]].append(row)
    strata = defaultdict(list)
    for key, group in sorted(groups.items()):
        strata[tuple(sorted({r["task_type"] for r in group}))].append(key)
    rng = random.Random(seed)
    for keys in strata.values(): rng.shuffle(keys)
    # Equal-turn stratified ordering, then an exact subset-sum of whole groups.
    queues = [deque(keys) for _,keys in sorted(strata.items())]
    order = []
    while any(queues):
        for queue in queues:
            if queue: order.append(queue.popleft())
    paths = {0:()}
    for key in order:
        size = len(groups[key])
        for n in sorted(paths,reverse=True):
            if n+size<=count and n+size not in paths: paths[n+size]=paths[n]+(key,)
        if count in paths: break
    if count not in paths:
        raise ValueError(f"Cannot allocate exactly {count} tasks as whole source groups; available={len(rows)}")
    selected = set(paths[count])
    return [r for key in paths[count] for r in groups[key]], [r for r in rows if r["group_id"] not in selected]


def make_budget_manifest(benchmark, revision, rows, *, split_seed=42, sizes=None):
    sizes = dict(BUDGET_SIZES if sizes is None else sizes)
    if set(sizes)!=set(BUDGET_SIZES) or any(type(n) is not int or n<1 for n in sizes.values()):
        raise ValueError("Specify a positive, separate size for every internal partition")
    if min(sizes[p] for p in sizes if p.startswith('retirement_'))<30 or sizes['dev']<30:
        raise ValueError("Statistical cohort minimum remains 30")
    if benchmark not in ('alfworld','webshop','hotpotqa') or not revision or revision in ('main','master','latest'):
        raise ValueError("Supported benchmark and immutable source revision required")
    if len({r['id'] for r in rows}) != len(rows): raise ValueError("Duplicate source task IDs")
    for row in rows:
        if any(not row.get(k) for k in ('id','group_id','task_type','source','source_hash','split')):
            raise ValueError("Missing real source/group/type metadata")
    expected = {'alfworld':{'official_train','valid_seen','valid_unseen'},
                'hotpotqa':{'official_train','distractor_dev'},
                'webshop':{'official_train','official_eval','official_test'}}[benchmark]
    if {r['source'] for r in rows} != expected: raise ValueError("Missing or unexpected official source roles")
    finals = {'alfworld':{'valid_seen':'test_valid_seen','valid_unseen':'test_valid_unseen'},
              'hotpotqa':{'distractor_dev':'test'},'webshop':{'official_test':'test'}}[benchmark]
    final_groups={r['group_id'] for r in rows if r['source'] in finals}
    pools=defaultdict(list)
    excluded=[]
    for row in rows:
        if row['source'] in finals: continue
        if row['group_id'] in final_groups: excluded.append(row['id'])
        else: pools[row['source']].append(row)
    partitions={name:tuple(r['id'] for r in rows if r['source']==source) for source,name in finals.items()}
    selected_rows={r['id']:r for r in rows}
    assigned_groups=set()
    # Eval budgets first in WebShop; shared train/eval product instances cannot cross partitions.
    names=['dev','retirement_0','retirement_1','retirement_2','search','train']
    for index,name in enumerate(names):
        source='official_eval' if benchmark=='webshop' and name not in ('train','search') else 'official_train'
        available=[r for r in pools[source] if r['group_id'] not in assigned_groups]
        selected,_=grouped_selection(available,sizes[name],split_seed+index)
        partitions[name]=tuple(r['id'] for r in selected)
        assigned_groups.update(r['group_id'] for r in selected)
    manifest=TaskManifest(benchmark,revision,partitions)
    manifest.validate_loop(3,versioned=True,cohort_minimum=30)
    report={'status':'prepared_not_executed','benchmark':benchmark,'revision':revision,'split_seed':split_seed,
        'grouping':'same-instance groups; equal-turn task-type strata; exact whole-group subset sum',
        'requested_sizes':sizes,'manifest_hash':manifest.fingerprint,
        'source_counts':dict(Counter(r['source'] for r in rows)),
        'excluded_internal_aliases_of_final_instances':excluded,
        'unused_internal_count':sum(len(v) for v in pools.values())-sum(sizes.values()),
        'partitions':{name:{'count':len(ids),'sources':dict(Counter(selected_rows[i]['source'] for i in ids)),
            'task_types':dict(Counter(selected_rows[i]['task_type'] for i in ids)),
            'groups':len({selected_rows[i]['group_id'] for i in ids}),'task_ids':list(ids)} for name,ids in partitions.items()}}
    return manifest,report


def alfworld_rows(root):
    root=Path(root).resolve(strict=True)
    rows=[]
    for split in ('train','valid_seen','valid_unseen'):
        for game in sorted((root/split).rglob('game.tw-pddl')):
            metadata=game.parent/'traj_data.json'
            record=json.loads(metadata.read_text())
            kind=record['task_type']
            # All trials of a task/object/scene instance stay together.
            instance=game.parent.parent.name
            rows.append({'id':str(game),'split':'train' if split=='train' else 'test',
                'source':'official_train' if split=='train' else split,'source_hash':digest([sha256(game),sha256(metadata)]),
                'task_type':kind,'group_id':digest(['alfworld',instance])})
    return rows


def hotpot_rows(train, distractor_dev):
    from .importers import import_tasks
    rows=[]
    for path,source,split in ((train,'official_train','train'),(distractor_dev,'distractor_dev','test')):
        for row in import_tasks('hotpotqa',path,split=split):
            record=row['record']
            if record.get('type') not in ('bridge','comparison'): raise ValueError('Official HotpotQA task type missing')
            # Question identity groups duplicate or re-IDed copies without grouping unrelated questions by title.
            row.update(source=source,task_type=record['type']+':'+record.get('level','unknown'),
                group_id=digest(['hotpotqa',' '.join(record['question'].lower().split()),sorted(set(t for t,_ in record['supporting_facts']))]))
            rows.append(row)
    return rows


def write_dataset(output, benchmark, revision, rows, split_seed=42, sizes=None):
    output=Path(output)
    if output.exists(): raise FileExistsError('Use a new output directory; historical manifests are immutable')
    manifest,report=make_budget_manifest(benchmark,revision,rows,split_seed=split_seed,sizes=sizes)
    output.mkdir(parents=True)
    catalog={'schema_version':1,'benchmark':benchmark,'revision':revision,'tasks':rows}
    write_json(output/'catalog.json',catalog)
    write_json(output/'manifest.json',{'benchmark':benchmark,'environment_revision':revision,
        'partitions':manifest.partitions,'manifest_hash':manifest.fingerprint,'split_seed':split_seed,
        'catalog_hash':digest(catalog),'distribution_report':'distribution.json'})
    write_json(output/'distribution.json',report)
    return report
