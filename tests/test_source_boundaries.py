"""Current code boundaries, replacing historical test-hash/demo-byte constraints."""
import ast
import importlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from internalization.core.interfaces import Components, ProposalRequest
from internalization.core.types import Journal
from internalization.evolution.code_proposer import CodeProposer, CONTRACT
from internalization.evolution.proposer import APITransport
from internalization.evolution.revision_search import public_history
from internalization.harness.revision import RevisionStore

ROOT=Path(__file__).resolve().parents[1]
RETIRED=('harness_modules','records','manifest','backends','teacher_harness','student_harness','retirement_eval','opid_adapter','demo','alfworld_eval')

class SourceBoundaryTests(unittest.TestCase):
    def test_production_has_no_retired_import_or_upstream_runtime_path(self):
        for directory in ('src','scripts','configs','examples'):
            for file in (ROOT/directory).rglob('*'):
                if not file.is_file() or file.suffix not in ('.py','.json','.sh','.yaml','.toml'):continue
                source=file.read_text()
                for forbidden in ('upstream/OPID','upstream/meta-harness','scripts/meta_proposer.py','scripts/train_opid.py','harness_modules/'):
                    self.assertNotIn(forbidden,source,str(file))
                if file.suffix=='.py':
                    tree=ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node,ast.ImportFrom):
                            self.assertNotIn(node.module,tuple('internalization.'+n for n in RETIRED),str(file))
                            self.assertFalse(node.module and node.module.startswith(('opid.','agent_system.','meta_harness.')),str(file))
                        elif isinstance(node,ast.Import):
                            for name in node.names:self.assertFalse(name.name.startswith(('opid.','agent_system.','meta_harness.')),str(file))
        self.assertFalse((ROOT/'harness_modules').exists())
        for name in RETIRED:self.assertFalse((ROOT/'src/internalization'/f'{name}.py').exists())
        self.assertFalse(any((ROOT/'docs/history').rglob('*.py')))

    def test_current_modules_import_and_explicit_components_are_supported(self):
        modules=('outer_loop','revision_loop','command_backend','core.accepted_state','evolution.code_proposer',
            'training.entrypoint','training.trainer','training.smoke','training.revision_scoring',
            'benchmarks.alfworld','benchmarks.appworld','benchmarks.hotpotqa','benchmarks.webshop',
            'benchmarks.terminalbench','benchmarks.swebench','benchmarks.lawbench_final')
        for name in modules:importlib.import_module('internalization.'+name)
        from internalization.core.adapters import components_for
        components=Components(None,None,None)
        class Backend:
            def components(self):return components
        self.assertIs(components_for(components),components)
        self.assertIs(components_for(Backend()),components)
        with self.assertRaisesRegex(TypeError,'legacy'):components_for(object())

    def test_configured_python_entrypoints_exist(self):
        from internalization.command_backend import CommandBackend
        with tempfile.TemporaryDirectory() as directory:
            for path in (ROOT/'configs').rglob('*backend.json'):
                config=json.loads(path.read_text())
                # Configuration-only fixtures; no catalog, evaluator or environment is opened.
                variables={name:'fixture-only' for file in (ROOT/'configs').rglob('*.json')
                    for name in re.findall(r'\$\{([^}]+)\}',file.read_text())}
                with patch.dict(os.environ,variables):
                    CommandBackend(config,Path(directory)/'cost.jsonl')
                for stage in ('propose','target','check_internalization','evaluate','train'):
                    for part in config.get(stage,[]):
                        if part.startswith('scripts/') and part.endswith('.py'):self.assertTrue((ROOT/part).is_file(),part)
                        if part.startswith('internalization.'):importlib.import_module(part)

    def test_retired_cli_rejected_and_current_help(self):
        for args in ([],['run'],['retirement']):
            result=subprocess.run([sys.executable,'-m','internalization.cli',*args,'--help'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        for command in ('demo','prepare-phase','validate-module'):
            result=subprocess.run([sys.executable,'-m','internalization.cli',command,'--help'],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('invalid choice',result.stderr)

    def test_code_proposer_mock_transport_preserves_request_auth_and_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=RevisionStore(root/'revisions')
            parent=store.import_directory(ROOT/'examples/versioned_harness/base')
            captured=[]
            def transport(payload):
                captured.append(payload)
                return {'choices':[{'message':{'content':json.dumps({'candidates':[
                    {'patch':[{'path':'prompts/system.txt','content':'new prompt'}],'rationale':'one'},
                    {'patch':[{'path':'prompts/system.txt','content':'other prompt'}],'rationale':'two'}]})}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':20}}
            proposer=CodeProposer(store,model='mock',transport=transport,max_tokens=123,temperature=.3)
            request=ProposalRequest('model',parent,(),(),(),({'source':'prior'},),0,2,root/'proposal')
            with patch('internalization.evolution.proposer.time.monotonic',side_effect=[10.,12.]):
                candidates=proposer.propose(request)
            self.assertEqual(len(candidates),2)
            for candidate in candidates:candidate.validate(parent)
            self.assertEqual(captured[0]['messages'][0],{'role':'system','content':CONTRACT})
            self.assertIn('prior',captured[0]['messages'][1]['content'])
            self.assertEqual((captured[0]['max_tokens'],captured[0]['temperature']),(123,.3))
            self.assertEqual(proposer.last_cost.input_tokens,10)
            self.assertEqual(proposer.last_cost.latency_s,2.)
            # Exercise HTTP construction locally, without making a network call.
            class Response:
                def __enter__(self):return self
                def __exit__(self,*a):pass
                def read(self):return json.dumps({'choices':[{'message':{'content':'{}'}}],
                    'usage':{'prompt_tokens':1,'completion_tokens':2}}).encode()
            with patch('internalization.evolution.proposer.urllib.request.urlopen',return_value=Response()) as urlopen:
                APITransport(model='mock',base_url='https://example.invalid/v1/',api_key='test-key').request_json('contract',{},root/'http')
            call=urlopen.call_args
            self.assertEqual(call.args[0].full_url,'https://example.invalid/v1/chat/completions')
            self.assertEqual(call.args[0].get_header('Authorization'),'Bearer test-key')
            self.assertEqual(call.kwargs['timeout'],300)

    def test_versioned_archive_preserves_failed_successful_and_filters_private_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            journal=Journal(Path(directory)/'events.jsonl')
            journal.append('candidate_failed',candidate={'id':'bad'},reason='retirement SECRET')
            journal.append('code_candidate',candidate={'id':'good'},status='eligible',search_gain={'mean':.2},dev_gain={'mean':.9})
            journal.append('cycle_complete',reason='rollback',task='retirement_0')
            rows=public_history(journal)
            self.assertEqual([r['status'] for r in rows],['failed','eligible'])
            for hidden in ('SECRET','retirement','dev_gain','rollback'):self.assertNotIn(hidden,json.dumps(rows))
            self.assertIn('SECRET',journal.path.read_text())
