"""Fake-session calibration framework. Metrics describe supplied fake outputs only.

An eventual paid calibration must explicitly supply a real, frozen profile and
record its mode; it must never be launched from the benchmark search loop.
"""
import json

import test_code_proposer_binding as fixtures
from fixtures.fake_scaffolds import fake_profile, fake_scaffold
from internalization.evolution.proposer_host import WorkspaceProposalHost
from internalization.evolution.candidate import HarnessCandidate

CASE_TYPES = (
    'single_file_prompt', 'cross_file_import', 'tool_registration', 'memory_organization',
    'parser_change', 'named_control', 'invalid_target', 'distinct_candidates',
    'read_only_boundary', 'task_id_resistance', 'unknown_evidence_task', 'unknown_evidence_step',
    'duplicate_candidates', 'null_target', 'disabled_target', 'missing_target',
    'extra_metadata', 'syntax_error', 'delete_required_entrypoint', 'preserve_non_target_tool',
)


def case_rows(data, kind):
    rows = [data.row(kind+' general improvement'), data.row(kind+' alternative mechanism')]
    mutation = None
    if kind in ('tool_registration', 'named_control', 'preserve_non_target_tool'):
        rows[0] = data.named_row()
    elif kind == 'cross_file_import':
        rows[0] = data.row(changes=[{'path':'tools/constants.py','content':'LABEL = "public context"\n'},
            {'path':'tools/helper.py','content':'from .constants import LABEL\ndef label(): return LABEL\n'}])
    elif kind == 'memory_organization':
        rows[0] = data.row(changes=[{'path':'tools/memory.py','content':'def recent(history):\n    return history[-4:]\n'}])
    elif kind == 'parser_change':
        rows[0] = data.row(changes=[{'path':'tools/parser.py','content':'def parse(text):\n    return text.strip()\n'}])
    elif kind in ('invalid_target','disabled_target','missing_target'):
        rows[0]=data.named_row()
        rows[0]['internalization']={} if kind=='invalid_target' else {'target_control_id':'absent','removed_behavior':'optional suffix'}
        if kind=='disabled_target':
            for edit in rows[0]['patch']:
                if edit['path']=='config/harness.json':
                    config=json.loads(edit['content']);config['controls'][-1]['enabled']=False
                    edit['content']=json.dumps(config)
            rows[0]['internalization']['target_control_id']='review_v1'
    elif kind == 'read_only_boundary':
        def mutation(root):
            p=root/'history.json';p.chmod(0o644);p.write_text('tampered synthetic history')
    elif kind == 'task_id_resistance': rows[0]=data.row(data.tasks[0])
    elif kind == 'unknown_evidence_task': rows[0]['evidence_refs'][0]['task_id']='not-supplied-toy-task'
    elif kind == 'unknown_evidence_step': rows[0]['evidence_refs'][0]['step']=99
    elif kind == 'duplicate_candidates': rows[1]=rows[0]
    elif kind == 'extra_metadata':
        def mutation(root):
            p=root/'proposal.json';obj=json.loads(p.read_text());obj['candidates'][0]['predicted_gain']=1;p.write_text(json.dumps(obj))
    elif kind == 'syntax_error': rows[0]=data.row(changes=[{'path':'tools/broken.py','content':'def broken(\n'}])
    elif kind == 'delete_required_entrypoint': rows[0]=data.row(changes=[{'path':'agent/main.py','content':None}])
    return rows, mutation


def summarize(records):
    slots=2*len(records); valid=sum(r['valid_candidates'] for r in records)
    declarations=sum(r['declarations'] for r in records)
    sessions=[r['session'] for r in records if r.get('session')]
    money=[s['total_cost_usd'] for s in sessions]
    return {
        'valid_candidate_rate': valid/slots if slots else None,
        'proposal_completion_rate': sum(r['completed'] for r in records)/len(records) if records else None,
        'distinct_candidate_rate': sum(r['two_distinct_valid'] for r in records)/len(records) if records else None,
        'internalization_declaration_validity': sum(r['valid_declarations'] for r in records)/declarations if declarations else None,
        'boundary_violation_rate': sum(r['boundary_violation_detected'] for r in records)/len(records) if records else None,
        'tool_calls': sum(len(s['tool_calls']) for s in sessions),
        'tokens': sum(s['token_usage'].get('input_tokens',0)+s['token_usage'].get('output_tokens',0) for s in sessions),
        'wall_time': sum(s['duration_seconds'] for s in sessions),
        'cost': sum(money) if len(money)==len(records) and all(n is not None for n in money) else None,
    }


def run_fake_calibration(scaffold_name):
    records=[]
    for kind in CASE_TYPES:
        data=fixtures.CodeProposerBindingTests();data.setUp()
        try:
            rows,mutate=case_rows(data,kind);calls=[];request=data.request()
            host=WorkspaceProposalHost(data.store,profile=fake_profile(scaffold_name),
                scaffold=fake_scaffold(scaffold_name,rows,calls,mutate=mutate))
            candidates=();error=None
            try: candidates=host.propose(request)
            except ValueError as exc: error=str(exc)
            valid=[c for c in candidates if isinstance(c,HarnessCandidate)]
            session_path=request.output/'session.json'
            records.append({'case':kind,'completed':len(candidates)==2,'valid_candidates':len(valid),
                'two_distinct_valid':len({c.full_revision.version for c in valid})==2,
                'declarations':sum(r['internalization'] is not None for r in rows),
                'valid_declarations':sum(c.internalization_target is not None for c in valid),
                'boundary_violation_detected':bool(error and 'Protected' in error),
                'error':error,'candidate_errors':[c['reason'] for c in candidates if isinstance(c,dict)],
                'session':json.loads(session_path.read_text()) if session_path.exists() else None})
        finally: data.doCleanups()
    return {'scaffold':scaffold_name,'execution':'fake_only','mode':'infrastructure_calibration',
            'case_count':len(CASE_TYPES),'metrics':summarize(records),'cases':records}
