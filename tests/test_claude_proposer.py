import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from internalization.evolution.claude_code import ClaudeCodeRunner, ClaudeCodeSessionResult, ClaudeCodePreflightError, parse_stream
from internalization.evolution.claude_proposer import ClaudeCodeProposer, canonical_diff
from internalization.evolution.claude_tool_guard import ALLOWED_TOOLS, check_tool
from internalization.evolution.candidate import HarnessCandidate
from internalization.core.execution_config import resolve_execution
import test_code_proposer_binding as binding_fixtures
from fixtures.fake_claude import editing_session


class ClaudeProposerTests(unittest.TestCase):
    def setUp(self):
        self.data = binding_fixtures.CodeProposerBindingTests(); self.data.setUp()
        self.addCleanup(self.data.doCleanups)
        self.root = self.data.root

    def run_proposal(self, rows=None, name='proposal', **kwargs):
        rows = rows or [self.data.named_row(), self.data.row()]
        calls = []
        proposer = ClaudeCodeProposer(self.data.store, runner=editing_session(rows, calls, **kwargs))
        request = self.data.request(name=name)
        return proposer.propose(request), proposer, calls, request

    def test_one_session_two_common_parent_candidates_and_deterministic_target(self):
        candidates, proposer, calls, request = self.run_proposal()
        self.assertEqual(len(calls), 1)
        for c in candidates:
            c.validate(self.data.parent)
            self.assertEqual(HarnessCandidate.from_dict(c.to_dict()), c)
        target = candidates[0].internalization_target
        self.assertIsNotNone(target); self.assertIsNone(candidates[1].internalization_target)
        self.assertEqual(target.target_control_id, 'review_v1')
        self.assertEqual([c['enabled'] for c in target.reduced_revision.config['controls']], [True, False])
        self.assertIn('tools/log_query.py', target.reduced_revision.files())
        self.assertNotEqual(target.reduced_revision.version, self.data.parent.version)
        expected = self.data.store.bind_patch(self.data.parent, self.data.named_row()['patch'])
        self.assertEqual({e.path:e for e in candidates[0].patch}, {e.path:e for e in expected})
        self.assertEqual(candidates[0].evidence_refs, ({'task_id': self.data.tasks[0], 'step': 0},))

    def test_shared_evidence_hardcoding_and_duplicate_checks(self):
        cases = []
        row = self.data.row(); row['evidence_refs'] = [{'task_id': 'private-test', 'step': 0}]; cases.append((row, 'evidence reference'))
        row = self.data.row(); row['evidence_refs'][0]['step'] = 99; cases.append((row, 'evidence reference'))
        cases.append((self.data.row('hardcode '+self.data.tasks[7]), 'exact supplied search task ID'))
        for i, (row, reason) in enumerate(cases):
            (bad, good), _, calls, _ = self.run_proposal([row, self.data.row()], name=f'bad-{i}')
            self.assertIn(reason, bad['reason']); good.validate(self.data.parent); self.assertEqual(len(calls), 1)
        (good, bad), _, calls, _ = self.run_proposal([self.data.row(), self.data.row()], name='duplicate')
        self.assertIn('duplicate', bad['reason']); self.assertEqual(len(calls), 1)

    def test_invalid_optional_declaration_is_harness_only_without_repair(self):
        for i, declaration in enumerate(({}, [], {'target_control_id':'missing', 'removed_behavior':'review'})):
            row = self.data.named_row(); row['internalization'] = declaration
            (candidate, _), _, calls, request = self.run_proposal([row, self.data.row()], name=f'declaration-{i}')
            candidate.validate(self.data.parent); self.assertIsNone(candidate.internalization_target)
            self.assertEqual(len(calls), 1)
            error = json.loads((request.output/'candidate_0/internalization_declaration_error.json').read_text())
            self.assertEqual(error['repair_model_calls'], 0)

    def test_workspace_search_only_and_exact_tool_and_setting_arguments(self):
        (_, _), proposer, calls, request = self.run_proposal()
        call = calls[0]; argv = call['argv']; workspace = request.output/'claude_workspace'
        self.assertEqual(Path(call['cwd']), workspace.resolve())
        self.assertEqual(argv[argv.index('--tools')+1].split(','), list(ALLOWED_TOOLS))
        self.assertEqual(argv[argv.index('--allowedTools')+1].split(','), list(ALLOWED_TOOLS))
        self.assertEqual(ALLOWED_TOOLS, ('Read', 'Glob', 'Grep', 'Edit', 'Write'))
        self.assertNotIn('--add-dir', argv); self.assertNotIn('--dangerously-skip-permissions', argv)
        for flag in ('--bare', '--strict-mcp-config', '--disable-slash-commands'): self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index('--setting-sources')+1], '')
        self.assertEqual(list(Path(argv[argv.index('--plugin-dir')+1]).iterdir()), [])
        self.assertEqual(json.loads(argv[argv.index('--mcp-config')+1]), {'mcpServers':{}})
        prompt = argv[argv.index('--append-system-prompt')+1]
        self.assertIn('EXPLICIT WORKFLOW', prompt); self.assertIn('Deep-read', prompt)
        scores = json.loads((workspace/'evidence/scores.json').read_text())
        self.assertEqual(len(scores), 8); self.assertEqual(len(list((workspace/'evidence').glob('trajectory_*.json'))), 4)
        data = ''.join(p.read_text() for p in (workspace/'evidence').iterdir())
        for secret in ('private-dev', 'private-retirement', 'private-test', 'old_checkpoint_path'):
            self.assertNotIn(secret, data)
        self.assertNotIn('HI_PROPOSER_API_KEY', call['env']); self.assertNotIn('NODE_OPTIONS', call['env'])
        self.assertNotIn('CLAUDE_CONFIG_DIR', str(workspace/'parent'))

    def test_read_write_boundary_and_symlink_rejection(self):
        (_, _), _, _, request = self.run_proposal()
        root = request.output/'claude_workspace'
        for tool, args in [('Bash', {'command':'true'}), ('Agent', {}), ('Read', {'file_path':str(self.root/'secret')}),
                           ('Write', {'file_path':'parent/agent/main.py'}), ('Edit', {'file_path':'history.json'}),
                           ('Glob', {'pattern':'../../*'}), ('Glob', {'pattern':'{../*,**}'}),
                           ('Glob', {'pattern':'{/private/*,**}'}), ('Grep', {'path':'..','pattern':'secret'})]:
            with self.assertRaises(ValueError): check_tool(root, tool, args)
        self.assertEqual(check_tool(root, 'Write', {'file_path':'candidate_0/tools/new.py'}), 'candidate_0/tools/new.py')
        self.assertEqual(check_tool(root, 'Write', {'file_path':'proposal.json'}), 'proposal.json')
        (root/'candidate_0/link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): check_tool(root, 'Read', {'file_path':'candidate_0/link/secret'})

    def test_protected_input_mutation_and_outside_tree_fail_closed(self):
        def change(root):
            path = root/'history.json'; path.chmod(0o644); path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Protected proposer input changed'): self.run_proposal(mutate=change)
        self.data.parent.files()
        def link(root): (root/'candidate_0/tools/link.py').symlink_to(root/'parent/agent/main.py')
        (bad, good), _, _, _ = self.run_proposal(name='link', mutate=link)
        self.assertIn('workspace_error', bad['invalid_proposal']); good.validate(self.data.parent)

    def test_no_binary_fails_before_copy_or_any_session(self):
        runner = ClaudeCodeRunner(which=lambda _:None, run_command=lambda *a,**k:self.fail('No subprocess'),
                                  process_factory=lambda *a,**k:self.fail('No session'))
        proposer = ClaudeCodeProposer(self.data.store, runner=runner)
        request = self.data.request()
        with self.assertRaisesRegex(ClaudeCodePreflightError, 'claude executable not found'): proposer.propose(request)
        self.assertFalse((request.output/'claude_workspace').exists())
        self.assertEqual(json.loads((request.output/'preflight_error.json').read_text())['proposal_sessions'], 0)

    def test_nonzero_and_timeout_preserve_logs_cost_and_no_candidates(self):
        for name, options in [('exit', {'exit_code':7}), ('timeout', {'timeout':True})]:
            calls=[]; runner=editing_session([self.data.row(),self.data.row('other')],calls,**options)
            proposer=ClaudeCodeProposer(self.data.store,runner=runner);request=self.data.request(name=name)
            with self.assertRaisesRegex(RuntimeError, 'session failed'): proposer.propose(request)
            session=json.loads((request.output/'claude_session/session.json').read_text())
            self.assertEqual(session['timed_out'], name=='timeout')
            self.assertEqual(session['exit_code'], -9 if name=='timeout' else 7)
            self.assertEqual(session['stderr'], 'scripted stderr')
            self.assertEqual(session['total_cost_usd'], .03)
            self.assertFalse((request.output/'candidate_0.json').exists())
            self.assertEqual(len(calls),1)

    def test_cost_cache_and_file_logs_are_not_double_counted(self):
        _, proposer, _, request=self.run_proposal()
        session=json.loads((request.output/'claude_session/session.json').read_text())
        self.assertEqual(proposer.last_cost.input_tokens,110)
        self.assertEqual(proposer.last_cost.output_tokens,20)
        self.assertEqual(proposer.last_cost.model_calls,1)
        self.assertEqual(session['token_usage']['cache_read_input_tokens'],7)
        self.assertEqual(session['files_read']['evidence/scores.json'],{'attempts':1,'completed':1})
        self.assertIn('candidate_0/tools/log_query.py',session['files_written'])
        self.assertEqual(len(session['raw_events']),4)
        protocol=json.loads((request.output/'proposer_protocol.json').read_text())
        self.assertEqual(protocol['session_id'],'fake-session');self.assertEqual(protocol['cli_version'],'fake-cli-1')
        self.assertEqual(protocol['spec_hash'],hashlib.sha256((request.output/'claude_workspace/PROPOSER_SPEC.md').read_bytes()).hexdigest())

    def test_config_only_new_proposer_fields_and_no_task_knob_changes(self):
        value=resolve_execution({'proposer':{'model':'claude-test-exact','max_turns':3}})
        self.assertEqual(value['proposer'],{'backend':'claude_code','model':'claude-test-exact','max_turns':3})
        for row in ({'temperature':0},{'max_tokens':8192},{'model':'sonnet'},{'max_turns':0},{'max_turns':True},{'backend':'api'}):
            with self.assertRaises(ValueError):resolve_execution({'proposer':row})

    def test_diff_add_update_delete_is_deterministic(self):
        self.assertEqual(canonical_diff({'a':'old','b':'remove'},{'a':'new','c':'added'}),
            [{'path':'a','content':'new'},{'path':'b','content':None},{'path':'c','content':'added'}])

    def test_claude_outer_loop_preserves_dev_null_preflight_and_attribution_branches(self):
        def proposer(rows):
            calls=[]
            return ClaudeCodeProposer(self.data.store,runner=editing_session(rows,calls)),calls
        self.data.proposer=proposer
        for mode in ('null','reject','preflight_fail','target'):
            self.data.root=self.root/mode
            result,calls,seen,checks,trains=self.data.loop(mode)
            self.assertEqual(len(calls),1);self.assertEqual(trains,[])
            if mode=='reject':
                self.assertEqual(checks,[]);self.assertEqual(result['harness_revision']['version'],self.data.parent.version)
            else:
                self.assertNotEqual(result['harness_revision']['version'],self.data.parent.version)
                self.assertEqual(len(checks),0 if mode=='null' else 1)
                self.assertEqual(result['archive'][0]['reason'],'attribution_failed' if mode=='target' else 'accepted_without_internalization')

    def test_foreign_evidence_rejected_before_any_session(self):
        calls=[];runner=editing_session([self.data.row(),self.data.row('other')],calls)
        proposer=ClaudeCodeProposer(self.data.store,runner=runner)
        for task in ('private-dev','private-retirement','private-test'):
            request=replace(self.data.request(),trajectories=(self.data.trace(task),))
            with self.assertRaisesRegex(ValueError,'outside search'):proposer.propose(request)
        request=replace(self.data.request(),history=({'dev_gain':1.},))
        with self.assertRaisesRegex(ValueError,'search feedback'):proposer.propose(request)
        self.assertEqual(calls,[])

    def test_metadata_extra_field_invalidates_candidate_only(self):
        def change(root):
            p=root/'proposal.json';data=json.loads(p.read_text());data['candidates'][0]['teacher_prompt']='forbidden';p.write_text(json.dumps(data))
        (bad,good),_,calls,_=self.run_proposal(mutate=change)
        self.assertIsInstance(bad,dict);good.validate(self.data.parent);self.assertEqual(len(calls),1)

    def test_stream_duplicate_usage_terminal_errors_and_unexpected_tools(self):
        root=self.root/'claude_workspace';root.mkdir()
        init={'type':'system','subtype':'init','model':'claude-test','tools':list(ALLOWED_TOOLS),'session_id':'session'}
        message={'type':'assistant','message':{'id':'same','usage':{'input_tokens':5,'output_tokens':2,'cache_read_input_tokens':4},'content':[]}}
        final={'type':'result','subtype':'success','session_id':'session','usage':{'input_tokens':6,'output_tokens':3,'cache_read_input_tokens':8},'total_cost_usd':.2}
        result=parse_stream('\n'.join(json.dumps(e) for e in (init,message,message,final)),ClaudeCodeSessionResult('claude-test',str(root),exit_code=0))
        self.assertTrue(result.succeeded);self.assertEqual(result.model_calls,1);self.assertEqual(result.cost.input_tokens,14)
        for events in ([init,message], [init,message,{**final,'subtype':'error_max_turns'}], [{**init,'tools':['Bash']},final]):
            result=parse_stream('\n'.join(json.dumps(e) for e in events),ClaudeCodeSessionResult('claude-test',str(root),exit_code=0))
            self.assertFalse(result.succeeded)

    def test_trusted_hook_runs_without_agent_bash_permission(self):
        from internalization.evolution import claude_tool_guard
        root=self.root/'claude_workspace';root.mkdir()
        for name,target,decision in [('Read','../private','deny'),('Write','parent/x','deny'),('Write','candidate_0/tools/x.py','allow')]:
            result=subprocess.run([sys.executable,'-I','-S',claude_tool_guard.__file__,str(root),str(self.root/'guard.jsonl')],
                input=json.dumps({'tool_name':name,'tool_input':{'file_path':target}}),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['hookSpecificOutput']['permissionDecision'],decision)
        self.assertEqual(len((self.root/'guard.jsonl').read_text().splitlines()),3)


if __name__=='__main__':unittest.main()
