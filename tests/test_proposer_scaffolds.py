from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_code_proposer_binding as binding
from fixtures.fake_scaffolds import fake_profile, fake_scaffold, PROFILE_IDS
from internalization.evolution.proposer_host import WorkspaceProposalHost
from internalization.evolution.proposer_profiles import resolve_profile, verify_runtime, resume_profile
from internalization.evolution.scaffolds import scaffold_for
from internalization.evolution.scaffolds.base import ScaffoldPreflightError
from internalization.evolution.scaffolds.codex import CodexScaffold
from internalization.evolution.scaffolds.qwen_code import QwenCodeScaffold
from internalization.evolution.scaffolds.isolation import WorkspaceCapsule
from internalization.evolution.candidate import HarnessCandidate
from internalization.core.execution_config import resolve_execution, config_hash, save_effective, request_execution


class ScaffoldLibraryTests(unittest.TestCase):
    def setUp(self):
        self.data = binding.CodeProposerBindingTests(); self.data.setUp()
        self.addCleanup(self.data.doCleanups)

    def run_case(self, name, rows=None, suffix='', **options):
        calls=[]; request=self.data.request(name=name+suffix)
        scaffold=fake_scaffold(name, rows or [self.data.named_row(),self.data.row()], calls, **options)
        host=WorkspaceProposalHost(self.data.store,profile=fake_profile(name),scaffold=scaffold)
        return host.propose(request),calls,request

    def test_explicit_profiles_strict_resolution_no_routing(self):
        for name, ident in PROFILE_IDS.items():
            value=resolve_profile({'profile':ident})
            self.assertEqual(value['scaffold'],name)
            self.assertEqual(resolve_profile(value),value)
            self.assertEqual(scaffold_for(name).scaffold_id,name)
        value=fake_profile('claude_code');value['model']='qwen-named-provider-id'
        self.assertEqual(resolve_profile(value)['scaffold'],'claude_code')
        for changes in ({'scaffold':'deepcode'},{'extra':True},{'provider':{'api_key':'secret'}}):
            with self.assertRaises(ValueError):resolve_profile({**value,**changes})
        with self.assertRaisesRegex(ValueError,'unsupported scaffold'):scaffold_for('deepcode')
        with self.assertRaises(ValueError):resolve_profile({'model':'qwen'})
        with self.assertRaises(ValueError):resolve_profile({'profile':'../hidden'})

    def test_all_scaffolds_have_identical_workspace_and_materialization(self):
        trees=[]; revisions=[]; specs=[]; workflows=[]
        for name in PROFILE_IDS:
            candidates,calls,request=self.run_case(name)
            self.assertEqual(len(calls),1)
            root=request.output/'proposer_workspace'
            self.assertEqual(set(p.name for p in root.iterdir()),{'parent','candidate_0','candidate_1','evidence','history.json','PROPOSER_SPEC.md','proposal.json'})
            tree={p.relative_to(root).as_posix():p.read_text() for p in root.rglob('*') if p.is_file()}
            trees.append(tree)
            for candidate in candidates:
                self.assertIsInstance(candidate,HarnessCandidate)
                candidate.validate(self.data.parent)
                self.assertEqual(HarnessCandidate.from_dict(candidate.to_dict()),candidate)
            revisions.append([c.full_revision.version for c in candidates])
            target=candidates[0].internalization_target
            target.validate_structure()
            self.assertEqual([c['enabled'] for c in target.reduced_revision.config['controls']],[True,False])
            self.assertIn('tools/log_query.py',target.reduced_revision.files())
            self.assertIsNone(candidates[1].internalization_target)
            protocol=json.loads((request.output/'proposer_protocol.json').read_text())
            specs.append(protocol['spec_hash']);workflows.append(protocol['workflow_hash'])
            self.assertEqual(protocol['resolved_profile'],fake_profile(name))
        self.assertEqual(trees[0],trees[1]);self.assertEqual(trees[1],trees[2])
        self.assertEqual(revisions[0],revisions[1]);self.assertEqual(revisions[1],revisions[2])
        self.assertEqual(len(set(specs)),1);self.assertEqual(len(set(workflows)),3)

    def test_shared_fail_soft_duplicates_evidence_hardcoding(self):
        for name in PROFILE_IDS:
            row=self.data.named_row();row['internalization']={'target_control_id':'missing','removed_behavior':'none'}
            candidates,calls,request=self.run_case(name,[row,self.data.row()],'-target')
            self.assertIsNone(candidates[0].internalization_target);self.assertEqual(len(calls),1)
            self.assertTrue((request.output/'candidate_0/internalization_declaration_error.json').exists())
            candidates,_,_=self.run_case(name,[self.data.row(),self.data.row()],'-duplicate')
            self.assertIn('duplicate',candidates[1]['reason'])
            row=self.data.row();row['evidence_refs'][0]['step']=100
            candidates,_,_=self.run_case(name,[row,self.data.row()],'-evidence')
            self.assertIn('evidence reference',candidates[0]['reason'])
            candidates,_,_=self.run_case(name,[self.data.row(self.data.tasks[7]),self.data.row()],'-memorize')
            self.assertIn('exact supplied search task ID',candidates[0]['reason'])

    def test_no_private_inputs_and_protected_mutation_rejected(self):
        for name in PROFILE_IDS:
            calls=[];scaffold=fake_scaffold(name,[self.data.row(),self.data.row('other')],calls)
            host=WorkspaceProposalHost(self.data.store,profile=fake_profile(name),scaffold=scaffold)
            for task in ('dev-private','retirement-private','test-private'):
                with self.assertRaisesRegex(ValueError,'outside search'):
                    host.propose(replace(self.data.request(),trajectories=(self.data.trace(task),)))
            self.assertEqual(calls,[])
            def mutate(root):
                p=root/'history.json';p.chmod(0o644);p.write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'Protected proposer input'):
                self.run_case(name,suffix='-tamper',mutate=mutate)
            self.data.parent.files()

    def test_unavailable_cost_is_null_not_zero(self):
        for name in ('codex','qwen_code'):
            _,_,request=self.run_case(name)
            result=json.loads((request.output/'session.json').read_text())
            self.assertIsNone(result['total_cost_usd'])
            self.assertTrue(result['token_usage']);self.assertTrue(result['raw_events'])
            self.assertTrue(result['tool_calls']);self.assertTrue(result['files_written'])
            if name=='codex':
                self.assertIsNone(result['model_calls']);self.assertIsNone(result['files_read'])

    def test_native_failure_timeout_and_accounting_survive(self):
        for name in ('codex','qwen_code'):
            for kind,options in [('exit',{'exit_code':4}),('timeout',{'timeout':True})]:
                with self.assertRaisesRegex(RuntimeError,'session failed'):
                    self.run_case(name,suffix=kind,**options)
                output=self.data.root/(name+kind)
                result=json.loads((output/'session.json').read_text())
                self.assertEqual(result['exit_code'],-9 if kind=='timeout' else 4)
                self.assertEqual(result['timed_out'],kind=='timeout')
                self.assertEqual(result['stderr'],'scripted stderr')
                self.assertTrue(result['raw_events'])

    def test_preflight_no_cli_and_no_isolation_fail_closed(self):
        for cls in (CodexScaffold,QwenCodeScaffold):
            with self.assertRaisesRegex(ScaffoldPreflightError,'executable not found'):
                cls(which=lambda _:None).preflight()
        with self.assertRaisesRegex(ScaffoldPreflightError,'bwrap executable required'):
            WorkspaceCapsule(which=lambda _:None).preflight()
        with self.assertRaisesRegex(ScaffoldPreflightError,'cannot enforce'):
            CodexScaffold().validate_limits({'max_turns':12,'max_tool_calls':None})

    def test_effective_config_roundtrip_resume_ignores_profile_file_mutation(self):
        execution=resolve_execution({'proposer':{'profile':'claude-sonnet5_claude-code_v1'}})
        with patch('internalization.evolution.proposer_profiles.PROFILE_ROOT',Path('/nonexistent')):
            self.assertEqual(resolve_execution(execution),execution)
            payload={'effective_config':execution,'effective_config_hash':config_hash(execution)}
            self.assertEqual(request_execution(payload),execution)
            self.assertEqual(resume_profile({'profile':execution['proposer']['profile_id']},execution['proposer']),execution['proposer'])
        with self.assertRaisesRegex(ValueError,'profile_id mismatch'):
            resume_profile({'profile':'qwen-coder_qwen-code_v1'},execution['proposer'])
        changed=deepcopy(execution['proposer']);changed['limits']['max_wall_time_s']=999
        with self.assertRaisesRegex(ValueError,'profile mismatch'):resume_profile(changed,execution['proposer'])
        save_effective(self.data.root,execution)
        for key,value in [('model','different'),('cli_version','different'),('profile_id','different')]:
            changed=deepcopy(execution);changed['proposer'][key]=value
            with self.assertRaisesRegex(ValueError,'configuration changed'):save_effective(self.data.root,changed)
        changed=deepcopy(execution);changed['proposer']['workflow_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'cannot silently resume'):resolve_execution(changed)
        profile=fake_profile('claude_code')
        with self.assertRaisesRegex(ScaffoldPreflightError,'version differs'):verify_runtime(profile,{'version':'other'})

    def test_deepseek_same_scaffold_provider_credentials_are_explicit(self):
        profile=resolve_profile({'profile':'deepseek-v41-flash_claude-code_v1'});profile['cli_version']='fake-cli-1'
        calls=[];scaffold=fake_scaffold('claude_code',[self.data.row(),self.data.row('other')],calls)
        host=WorkspaceProposalHost(self.data.store,profile=profile,scaffold=scaffold)
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'fake-key','ANTHROPIC_API_KEY':'wrong-provider','ANTHROPIC_BASE_URL':'https://wrong.invalid'}):
            host.propose(self.data.request())
        env=calls[0]['env']
        self.assertEqual(env['ANTHROPIC_API_KEY'],'fake-key')
        self.assertEqual(env['ANTHROPIC_BASE_URL'],profile['provider']['base_url'])
        self.assertNotIn('DEEPSEEK_API_KEY',env)

    def test_native_commands_keep_scoped_cwd_and_no_new_stage(self):
        for name in ('codex','qwen_code'):
            _,calls,request=self.run_case(name)
            command=calls[0]['argv'];env=calls[0]['env']
            self.assertEqual(Path(calls[0]['cwd']),request.output/'proposer_workspace')
            self.assertNotIn('--add-dir',command);self.assertNotIn('--include-directories',command)
            self.assertNotIn('ANTHROPIC_API_KEY',env)
            if name=='codex':
                self.assertIn('--json',command);self.assertIn('--ignore-user-config',command)
                self.assertIn('sandbox_workspace_write.network_access=false',command)
                self.assertEqual(calls[0]['stdin_text'].count('EXPLICIT WORKFLOW'),1)
            else:
                self.assertIn('--max-session-turns',command)
                self.assertIn('--max-wall-time',command)
                self.assertEqual(calls[0]['stdin_text'],'')
                self.assertEqual(command[command.index('--append-system-prompt')+1].count('EXPLICIT WORKFLOW'),1)
        root=Path(__file__).resolve().parents[1]
        for path in ('src/internalization/outer_loop.py','src/internalization/revision_loop.py','src/internalization/command_backend.py'):
            source=(root/path).read_text()
            for forbidden in ('CodexScaffold','QwenCodeScaffold','ClaudeCodeScaffold','propose_target'):
                self.assertNotIn(forbidden,source)

    def test_capsule_mounts_only_runtime_and_workspace(self):
        _,_,request=self.run_case('codex')
        capsule=WorkspaceCapsule();capsule.binary='/fake/bwrap'
        home=request.output/'scaffold_session/isolated_home'
        argv=capsule.wrap(['/usr/bin/codex','exec'],request.output/'proposer_workspace',home)
        self.assertNotIn('--ro-bind / /',' '.join(argv))
        bindings=[argv[i+1:i+3] for i,x in enumerate(argv) if x in ('--bind','--ro-bind')]
        for src,dest in bindings:
            self.assertNotEqual(src,str(self.data.root))
            self.assertNotEqual(src,str(Path.home()))
        writable=[argv[i+2] for i,x in enumerate(argv) if x=='--bind']
        self.assertEqual(set(writable),{'/home/proposer',*(str(request.output/'proposer_workspace'/n)
                                                      for n in ('candidate_0','candidate_1','proposal.json'))})
        with self.assertRaisesRegex(ScaffoldPreflightError,'ambient shell wrappers'):
            capsule.wrap(['/data/user/bin/codex','exec'],request.output/'proposer_workspace',home)

    def test_qwen_native_hook_blocks_non_file_tools_and_protected_paths(self):
        _,_,request=self.run_case('qwen_code')
        home=request.output/'scaffold_session/isolated_home'
        workspace=request.output/'proposer_workspace'
        for name,args,expected in [('write_file',{'file_path':'candidate_0/prompts/new.txt'},'allow'),
                ('read_file',{'file_path':'evidence/scores.json'},'allow'),
                ('write_file',{'file_path':'parent/agent/main.py'},'deny'),
                ('run_shell_command',{'command':'not executed'},'deny'),
                ('web_search',{},'deny'),('agent',{},'deny'),('read_file',{'file_path':'../checkpoint'},'deny')]:
            result=subprocess.run(['python3.12','-I','-S',str(home/'guard.py'),str(workspace),str(home/'test-audit.jsonl')],
                input=json.dumps({'tool_name':name,'tool_input':args}),text=True,capture_output=True,check=True)
            self.assertEqual(json.loads(result.stdout)['hookSpecificOutput']['permissionDecision'],expected)

    def test_skill_workflow_and_spec_resources_are_shared(self):
        root=Path(__file__).resolve().parents[1]
        self.assertEqual((root/'.claude/skills/harness-internalization-proposer/SKILL.md').read_bytes(),
                         (root/'src/internalization/evolution/workflows/claude_code.md').read_bytes())
        profile=fake_profile('claude_code');profile['cli_version']=None
        with self.assertRaisesRegex(ScaffoldPreflightError,'exact CLI version'):verify_runtime(profile,{'version':'whatever'})


if __name__=='__main__':unittest.main()
