"""CPU/mock parity artifact producer; run with PYTHONPATH=src:tests.

No production API, data download, or GPU training. See CLEANUP_REPORT.md.
"""
import json,tempfile
from pathlib import Path
from test_revision_training import RevisionTrainingTests
from test_budget_v1 import BudgetTests
from internalization.core.execution_config import resolve_execution
from internalization.evaluation.retirement import evaluate_retirement
from test_core import RetirementTests
result={'configs':{},'training':{},'retirement':{}}
for path in Path('configs').glob('*execution.json'):
 result['configs'][str(path)]=resolve_execution(json.loads(path.read_text()))
for path in Path('configs/budget_v1').glob('*.json'):
 if not path.stem.endswith(('_backend','_environment')):result['configs'][str(path)]=resolve_execution(json.loads(path.read_text()))
for name,kwargs in [('normal',{}),('noop',{'noop':True}),('shared',{'shared':True})]:
 with tempfile.TemporaryDirectory() as d:
  p=RevisionTrainingTests().run_training(Path(d),**kwargs)
  rows=[json.loads(line) for line in (Path(d)/'training/training.jsonl').read_text().splitlines()]
  result['training'][name]={'events':p.events,'batches':[{k:v.tolist() for k,v in b.items()} for b in p.batches],
   'scores':[{k:r[k] for k in ('student_prompt','enhanced_context','response_ids','signal')} for r in rows if r['kind']=='revision_teacher_state']}
case=BudgetTests();case.setUp()
try:
 seen,requests,deployment,_=case.run_loop()
 result['search']={'tasks':[r.tasks for r in requests],'evaluations':[(tasks,path.name,seeds,revision) for tasks,path,seeds,revision in seen],
   'decisions':[(r['reason'],r['model_decision'],r['module_decision']) for r in deployment['archive']]}
finally:case.doCleanups()
fixture=RetirementTests()
for c,d in [(1,1),(1,0),(0,0)]:
 cells=fixture.cells();cells['C']=fixture.rows(c,100,2);cells['D']=fixture.rows(d,40,0)
 result['retirement'][f'{c},{d}']=evaluate_retirement(cells,fixture.policy)
import sys
Path(sys.argv[1]).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
