"""Independent synthetic fixtures: these tests are not official benchmark scores."""
import asyncio
import csv
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from internalization.benchmarks.common import BenchmarkConfig, Catalog, score01, tree_hash
from internalization.benchmarks.hotpotqa import HotpotQAEnvironment, metrics
from internalization.benchmarks.importers import import_tasks, make_catalog_manifest
from internalization.benchmarks.isolated import IsolatedEnvironment, WorkerProcess, environment_factory
from internalization.benchmarks.lawbench import LawBenchEnvironment, OfficialLawBenchScorer, CATEGORIES
from internalization.benchmarks.swebench import SWEBenchProEnvironment, DockerWorkspace, OfficialProScorer
from internalization.benchmarks.terminalbench import TerminalBench2Environment, HarborSession
from internalization.benchmarks.aggregate import aggregate
from internalization.core.types import Cost, EpisodeResult
from internalization.command_backend import serialize_harness
from internalization.harness.module import Harness, HarnessModule
from internalization.harness.runtime import Completion
from internalization.training import entrypoint
import test_behavior_policy as fixtures


def hotpot_record():
    return {"_id":"q1", "question":"Which city contains the tower?", "answer":"SECRET_CITY",
        "supporting_facts":[["Tower",0]], "context":[["Tower",["The tower is in PUBLIC_CITY."]],
        ["River",["The river flows west."]]]}


def make_config(root, benchmark, records, **kwargs):
    catalog = root/(benchmark+"-catalog.json")
    catalog.write_text(json.dumps({"benchmark":benchmark,"revision":"synthetic-fixture-v1","tasks":records}))
    return BenchmarkConfig(benchmark, str(catalog), options={"python":sys.executable},
                           max_steps=1 if benchmark=="lawbench" else 3, **kwargs)


def hotpot_config(root, split="train"):
    return make_config(root,"hotpotqa",[{"id":"q1","split":split,"record":hotpot_record()}])


class AdapterTests(unittest.TestCase):
    def test_worker_keeps_virtualenv_interpreter_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); executable=root/"venv/bin/python"; executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            config=replace(hotpot_config(root),options={"python":str(executable)})
            with patch("internalization.benchmarks.isolated.subprocess.Popen",side_effect=RuntimeError("probe")) as start:
                with self.assertRaisesRegex(RuntimeError,"probe"): WorkerProcess(config,root/"worker")
            self.assertEqual(start.call_args.args[0][0],str(executable))

    def test_lawbench_final_defers_undefined_singletons_to_official_category_scoring(self):
        from internalization.benchmarks.lawbench_final import evaluate
        class Model:
            snapshot_id="cpu-mock"
            def generate(self,prompt,*,purpose):
                if "SECRET" in prompt: raise AssertionError("Gold answer leaked")
                return Completion("prediction",Cost(2,1,1),(3,))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=make_config(root,"lawbench",[{"id":f"3-4:{i}","split":"test","category":"3-4",
                "record":{"instruction":"I","question":str(i),"answer":"SECRET无期" if i==0 else "刑期:3个月"}} for i in range(2)])
            with patch("internalization.benchmarks.lawbench_final.aggregate_lawbench",return_value={"official":.5}) as grade:
                result=evaluate(config,Model(),Harness(),("3-4:0","3-4:1"),(0,),root/"evaluation")
            self.assertEqual(len(grade.call_args.args[2]),2)
            self.assertFalse(result["per_example_reward_assigned"])
            predictions=list((root/"evaluation/predictions").glob("*.json"))
            self.assertEqual(len(predictions),2)
            self.assertTrue(all(json.loads(p.read_text())["per_example_reward"] is None for p in predictions))

    def test_hotpot_normalization_and_joint_metric(self):
        gold={"answer":"The red tower", "supporting_facts":[["Tower",0],["Tower",1]]}
        result=metrics("red",[["Tower",0]],gold)
        self.assertEqual(result["prec"],1.)
        self.assertEqual(result["recall"],.5)
        self.assertEqual(result["sp_recall"],.5)
        self.assertAlmostEqual(result["joint_f1"],.4)
        self.assertNotAlmostEqual(result["joint_f1"],result["f1"]*result["sp_f1"])
        self.assertEqual(metrics("RED tower!",gold["supporting_facts"],gold)["joint_em"],1.)

    def test_hotpot_yes_no_and_duplicate_facts(self):
        gold={"answer":"yes", "supporting_facts":[["A",0]]}
        self.assertEqual(metrics("yes indeed",[["A",0]],gold)["f1"],0.)
        self.assertEqual(metrics("YES",[["A",0],["A",0]],gold)["joint_f1"],1.)

    def test_hotpot_public_retrieval_history_and_hidden_grade(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); env=HotpotQAEnvironment(hotpot_config(root),root/"episode")
            public=env.reset("q1",0)["observation"]
            self.assertNotIn("SECRET_CITY",public)
            self.assertNotIn("PUBLIC_CITY",public)
            step=env.step('{"action":"lookup","title":"Tower"}')
            self.assertIn("PUBLIC_CITY",step["observation"])
            self.assertEqual((step["reward"],step["tool_calls"]),(0.,1))
            step=env.step('{"action":"final","answer":"wrong","supporting_facts":[]}')
            self.assertTrue(step["done"])
            self.assertEqual(step["reward"],0.)
            self.assertNotIn("SECRET_CITY",step["observation"])
            with self.assertRaises(RuntimeError): env.step("again")

    def test_hotpot_invalid_action_counts_toward_limit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); env=HotpotQAEnvironment(replace(hotpot_config(root),max_steps=1),root/"episode")
            env.reset("q1",0)
            result=env.step('{"action":"final","answer":3,"supporting_facts":null}')
            self.assertFalse(result["action_valid"])
            self.assertTrue(result["done"])
            self.assertEqual(result["success"],0.)
            self.assertTrue((root/"episode/grade.json").exists())

    def test_hotpot_search_only_fixed_context(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=hotpot_config(root)
            env=HotpotQAEnvironment(replace(config,options={"top_k":1}),root/"episode")
            env.reset("q1",0)
            result=env.step('{"action":"search","query":"river"}')
            self.assertIn("flows west",result["observation"])
            self.assertNotIn("SECRET_CITY",result["observation"])
            with self.assertRaises(ValueError): HotpotQAEnvironment(replace(config,options={"setting":"fullwiki"}),root)

    def test_catalog_rejects_train_and_search_on_test_and_drift(self):
        with tempfile.TemporaryDirectory() as d:
            config=hotpot_config(Path(d),"test"); catalog=Catalog(config.catalog,"hotpotqa")
            with self.assertRaises(ValueError): catalog.get("q1",training=True)
            with self.assertRaises(ValueError): catalog.check_selection(["q1"])
            catalog.check_selection(["q1"],final=True)
            with self.assertRaises(ValueError): Catalog(config.catalog,"hotpotqa","bad-hash")

    def test_catalog_import_hashes_native_ids_and_no_fake_training_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); file=root/"data.json"; file.write_text(json.dumps([hotpot_record()]))
            rows=import_tasks("hotpotqa",file)
            self.assertEqual(rows[0]["id"],"q1")
            catalog,manifest=make_catalog_manifest("hotpotqa","v1",rows)
            self.assertEqual(set(manifest.partitions),{"test"})
            with self.assertRaises(ValueError): manifest.partition("test")
            with self.assertRaises(ValueError): make_catalog_manifest("hotpotqa","v1",rows,rows)

    def test_train_partition_keeps_three_disjoint_retirement_cohorts(self):
        train=[{"id":str(i),"split":"train","record":{"question":str(i)}} for i in range(270)]
        test=[{"id":"heldout","split":"test","record":{"question":"heldout"}}]
        _,manifest=make_catalog_manifest("hotpotqa","v1",test,train)
        self.assertEqual([len(manifest.partitions[f"retirement_{i}"]) for i in range(3)],[30]*3)
        self.assertFalse(any(k.startswith("acceptance_") for k in manifest.partitions))
        self.assertEqual(len(manifest.partitions["train"]),120)
        with self.assertRaises(ValueError): make_catalog_manifest("hotpotqa","v1",test,train,cohort_size=1)

    def test_lawbench_import_preserves_category_row_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"1-2.json").write_text(json.dumps([{"instruction":"I","question":"Q","answer":"SECRET"}]))
            row=import_tasks("lawbench",root)[0]
            self.assertEqual(row["category"],"1-2")
            self.assertTrue(row["id"].endswith(":0"))
            self.assertEqual(row["split"],"test")

    def test_tb2_import_pins_complete_task_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); task=root/"task"; (task/"tests").mkdir(parents=True)
            (task/"task.toml").write_text('version="1.0"')
            (task/"instruction.md").write_text("PUBLIC")
            (task/"tests/test.sh").write_text("SECRET")
            row=import_tasks("terminalbench2",root)[0]
            self.assertEqual(row["task_hash"],tree_hash(task))
            (task/"tests/test.sh").write_text("changed")
            config=make_config(root,"terminalbench2",[row])
            env=TerminalBench2Environment(config,root/"episode")
            with self.assertRaisesRegex(ValueError,"changed"): asyncio.run(env.reset("task",0))

    def test_configuration_rejects_wrong_budgets(self):
        with self.assertRaises(ValueError): BenchmarkConfig("lawbench","x")
        with self.assertRaises(ValueError): BenchmarkConfig("hotpotqa","x",max_context=10)
        for value in (float("nan"),-1,2):
            with self.assertRaises(ValueError): score01(value)

    def test_real_hotpot_worker_process_and_terminal_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); env=IsolatedEnvironment(hotpot_config(root),root/"worker",training=True)
            try:
                self.assertNotIn("SECRET_CITY",env.reset("q1",5))
                self.assertEqual(env.step('{"action":"search","query":"tower"}').tool_calls,1)
                result=env.step('{"action":"final","answer":"SECRET_CITY","supporting_facts":[["Tower",0]]}')
                self.assertEqual(result.reward,1.)
                self.assertTrue(result.done)
            finally: env.close()
            self.assertEqual(json.loads((root/"worker/identity.json").read_text())["seed"],5)

    def test_entrypoint_final_guard_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=hotpot_config(root,"test"); path=root/"config.json"; path.write_text(json.dumps(asdict(config)))
            request=root/"request.json"; request.write_text(json.dumps({"stage":"evaluate","task_ids":["q1"]}))
            with patch.object(sys,"argv",["worker","--request",str(request),"--response",str(root/"out/response.json"),
                "--benchmark","hotpotqa","--env-config",str(path)]), patch.object(entrypoint,"FrozenHFBackend") as model:
                with self.assertRaises(ValueError): entrypoint.main("evaluate")
                model.assert_not_called()

    def test_evaluation_entrypoint_uses_actual_hotpot_worker(self):
        class Model:
            snapshot_id="mock"
            def generate(self,prompt,*,purpose):
                return Completion('{"action":"final","answer":"SECRET_CITY","supporting_facts":[["Tower",0]]}',Cost(3,2,1),(3,4))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=hotpot_config(root,"test"); path=root/"config.json"; path.write_text(json.dumps(asdict(config)))
            request=root/"request.json"; response=root/"out/response.json"
            request.write_text(json.dumps({"stage":"evaluate","task_ids":["q1"],"checkpoint":"mock",
                                           "harness":serialize_harness(Harness()),"seeds":[0]}))
            with patch.object(sys,"argv",["worker","--request",str(request),"--response",str(response),
                "--benchmark","hotpotqa","--env-config",str(path),"--final-evaluation"]), \
                patch.object(entrypoint,"FrozenHFBackend",return_value=Model()): entrypoint.main("evaluate")
            result=json.loads(response.read_text())
            self.assertEqual(result["benchmark_metrics"]["metrics"]["joint_f1"],1.)
            self.assertEqual(result["episodes"][0]["success"],1.)

    def test_two_batch_training_entrypoint_uses_worker_and_same_behavior_policy(self):
        from internalization.training import verl_backend
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=hotpot_config(root); path=root/"config.json"; path.write_text(json.dumps(asdict(config)))
            policy=fixtures.BehaviorPolicyTests().policy(); original=policy.generate
            def generate(prompt,*,purpose):
                result=original(prompt,purpose=purpose)
                if purpose=="rollout_action":
                    return Completion('{"action":"final","answer":"wrong","supporting_facts":[]}',result.cost,result.response_ids)
                return result
            policy.generate=generate
            full=Harness((HarnessModule.from_source(fixtures.SOURCE),))
            request=root/"request.json"; response=root/"out/response.json"
            request.write_text(json.dumps({"stage":"train","student_checkpoint":"mock","teacher_checkpoint":"mock",
                "full_harness":serialize_harness(full),"reduced_harness":serialize_harness(Harness()),
                "target":"target","task_ids":["q1"],"optimizer_steps":2}))
            with patch.object(sys,"argv",["worker","--request",str(request),"--response",str(response),
                "--benchmark","hotpotqa","--env-config",str(path)]), \
                patch.object(verl_backend,"VerlPolicy",return_value=policy), \
                patch.object(entrypoint,"FrozenHFBackend",return_value=fixtures.BehaviorPolicyTests().reference()):
                entrypoint.main("train")
            self.assertEqual(policy.updates,2)
            self.assertTrue(Path(json.loads(response.read_text())["checkpoint"]).is_dir())
            self.assertTrue(all(b["response_mask"].tolist()==[[1,1],[1,1]] for b in policy.batches))
            rows=[json.loads(l) for l in (root/"out/training.jsonl").read_text().splitlines()]
            updates=[r for r in rows if r["kind"]=="update"]
            self.assertEqual([r["behavior_snapshot"] for r in updates],["policy-0","policy-1"])
            self.assertTrue(all(r["teacher_snapshot"]==r["behavior_snapshot"] for r in updates))

    def test_lawbench_official_cli_dispatches_all_twenty_categories_without_rescaling(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=BenchmarkConfig("lawbench","unused",max_steps=1,
                options={"evaluator_root":str(root),"evaluator_commit":"a"*40})
            def run(argv,**kwargs):
                predictions=Path(argv[argv.index("-i")+1])/"student"
                self.assertEqual({p.stem for p in predictions.iterdir()},set(CATEGORIES))
                with Path(argv[argv.index("-o")+1]).open("w") as file:
                    writer=csv.DictWriter(file,fieldnames=["task","model_name","score","abstention_rate"])
                    writer.writeheader()
                    for category in CATEGORIES: writer.writerow(dict(task=category,model_name="student",score=.73,abstention_rate=.2))
            with patch("internalization.benchmarks.lawbench.checked_repo",return_value=root), \
                 patch("internalization.benchmarks.lawbench.subprocess.run",side_effect=run):
                result=OfficialLawBenchScorer(config).score({c:[{"prediction":"P","refr":"SECRET"}] for c in CATEGORIES},root/"grade")
            self.assertEqual(len(result),20)
            self.assertEqual(result["1-2"],{"score":.73,"abstention_rate":.2})

    def test_lawbench_native_single_response_has_no_tool_or_gold_leak(self):
        scorer=SimpleNamespace(score=lambda predictions,output:{"1-2":{"score":.5,"abstention_rate":0.}})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=make_config(root,"lawbench",[{"id":"1-2:0","split":"test","category":"1-2",
                "record":{"instruction":"I","question":"Q","answer":"SECRET"}}])
            env=LawBenchEnvironment(config,root/"episode",scorer=scorer)
            self.assertEqual(env.reset("1-2:0",0)["observation"],"I\nQ")
            result=env.step("Answer")
            self.assertEqual((result["tool_calls"],result["reward"],result["done"]),(0,.5,True))
            self.assertNotIn("SECRET",result["observation"])


class ContainerAdapterTests(unittest.TestCase):
    def test_pro_import_requires_image_lock_and_preserves_official_list_columns(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/"pro.jsonl"
            record={"instance_id":"instance_demo","repo":"A/B","base_commit":"a"*40,"problem_statement":"Issue",
                "fail_to_pass":["test_fix"],"pass_to_pass":["test_old"],"before_repo_set_cmd":"true",
                "selected_test_files_to_run":["tests.py"]}
            path.write_text(json.dumps(record)+"\n")
            with self.assertRaises(ValueError): import_tasks("swebench_pro",path)
            row=import_tasks("swebench_pro",path,image_lock={"instance_demo":{"tag":"official:tag","digest":"repo@sha256:"+"a"*64}})[0]
            self.assertEqual(row["record"]["fail_to_pass"],"['test_fix']")
            self.assertEqual(row["image_tag"],"official:tag")

    def test_official_pro_command_and_missing_grader_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=BenchmarkConfig("swebench_pro","unused",options={"evaluator_root":str(root),
                "evaluator_commit":"a"*40,"dockerhub_username":"official"})
            def run(argv,**kwargs):
                self.assertIn("--use_local_docker",argv); self.assertIn("--block_network",argv)
                prediction=json.loads(Path(argv[argv.index("--patch_path")+1]).read_text())
                self.assertEqual(prediction[0]["patch"],"STUDENT_PATCH")
                output=Path(argv[argv.index("--output_dir")+1]); (output/"instance_demo").mkdir(parents=True)
                (output/"eval_results.json").write_text(json.dumps({"instance_demo":True}))
                (output/"instance_demo/student_output.json").write_text(json.dumps({"tests":[{"name":"test_fix","status":"PASSED"}]}))
            with patch("internalization.benchmarks.swebench.checked_repo",return_value=root), \
                patch.object(OfficialProScorer,"assert_image") as image_check, \
                patch("internalization.benchmarks.swebench.subprocess.run",side_effect=run):
                scorer=OfficialProScorer(config)
                self.assertEqual(scorer.score({"instance_id":"instance_demo"},"STUDENT_PATCH",root/"good",image="digest"),1.)
                self.assertEqual(image_check.call_count,2)
            with patch("internalization.benchmarks.swebench.checked_repo",return_value=root), \
                patch.object(OfficialProScorer,"assert_image"), patch("internalization.benchmarks.swebench.subprocess.run"):
                with self.assertRaises(FileNotFoundError): scorer.score({"instance_id":"instance_demo"},"",root/"bad",image="digest")

    def test_tb2_environment_keeps_only_public_command_observations_at_step_limit(self):
        class Session:
            def __init__(self,*args): self.forced=[]
            async def start(self,path): return {"instruction":"PUBLIC_TASK"}
            async def step(self,action,force_finish):
                self.forced.append(force_finish)
                return {"observation":"PUBLIC_OUTPUT","finish":True,"terminal":True,"reward":1.,"tool_calls":1}
            async def close(self): pass
        async def exercise(root):
            tasks=root/"tasks"; task=tasks/"tb2-task"; (task/"tests").mkdir(parents=True)
            (task/"task.toml").write_text('version="1.0"'); (task/"instruction.md").write_text("PUBLIC_TASK")
            (task/"tests/test.sh").write_text("SECRET_TEST")
            config=make_config(root,"terminalbench2",import_tasks("terminalbench2",tasks))
            env=TerminalBench2Environment(replace(config,max_steps=1),root/"episode",session_factory=Session)
            self.assertNotIn("SECRET",(await env.reset("tb2-task",2))["observation"])
            result=await env.step('{"action":"exec","command":"pwd"}')
            self.assertEqual(env.session.forced,[True]); self.assertEqual(result["reward"],1.)
            self.assertNotIn("SECRET",result["observation"])
            await env.close()
        with tempfile.TemporaryDirectory() as d: asyncio.run(exercise(Path(d)))

    def pro(self, root, scorer=None):
        class Workspace:
            def __init__(self,*args): self.commands=[]; self.closed=False
            def execute(self,command):
                self.commands.append(command)
                return {"stdout":"PATCH" if "git diff" in command else "", "stderr":"","return_code":0}
            def close(self): self.closed=True
        self.workspace=Workspace()
        self.scored=[]
        def score(record,patch,output,**kwargs): self.scored.append(patch); return 1.
        scorer=scorer or SimpleNamespace(image_uri=lambda r:"official:tag",score=score)
        config=make_config(root,"swebench_pro",[{"id":"instance_demo","split":"test","image_tag":"official:tag",
            "image":"repo@sha256:"+"a"*64,"record":{"instance_id":"instance_demo","repo":"A/B",
            "base_commit":"a"*40,"problem_statement":"Fix the public issue","requirements":"Public requirements",
            "patch":"SECRET_GOLD_PATCH","fail_to_pass":"SECRET_TESTS"}}])
        return SWEBenchProEnvironment(config,root/"episode",workspace_factory=lambda *args:self.workspace,scorer=scorer)

    def test_pro_interaction_and_patch_verifier_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); env=self.pro(root)
            public=env.reset("instance_demo",2)["observation"]
            self.assertNotIn("SECRET",public)
            result=env.step('{"action":"exec","command":"touch solved"}')
            self.assertEqual(self.scored,[])
            self.assertEqual(result["tool_calls"],1)
            result=env.step('{"action":"final"}')
            self.assertEqual(self.scored,["PATCH"])
            self.assertEqual(result["reward"],1.)
            self.assertNotIn("PATCH",result["observation"])
            env.close(); self.assertTrue(self.workspace.closed)

    def test_pro_step_limit_still_scores_patch(self):
        with tempfile.TemporaryDirectory() as d:
            env=self.pro(Path(d)); env.config=replace(env.config,max_steps=1)
            env.reset("instance_demo",0)
            result=env.step("invalid")
            self.assertTrue(result["done"]); self.assertFalse(result["action_valid"])
            self.assertEqual(self.scored,["PATCH"])

    def test_pro_grader_fault_is_not_zero_reward(self):
        def fail(*args,**kwargs): raise RuntimeError("grader failed")
        with tempfile.TemporaryDirectory() as d:
            env=self.pro(Path(d),SimpleNamespace(image_uri=lambda r:"official:tag",score=fail))
            env.reset("instance_demo",0)
            with self.assertRaises(RuntimeError): env.step('{"action":"final"}')

    def test_docker_workspace_does_not_mount_host_or_run_commands_on_host(self):
        calls=[]
        class Container:
            def exec_run(self,argv,**kwargs): calls.append((argv,kwargs)); return SimpleNamespace(exit_code=0,output=(b"public",b""))
            def stop(self,**kwargs): pass
        def run(image,**kwargs):
            self.assertNotIn("volumes",kwargs); self.assertEqual(kwargs["network_mode"],"none")
            return Container()
        client=SimpleNamespace(images=SimpleNamespace(get=lambda image:None),containers=SimpleNamespace(run=run),close=lambda:None)
        workspace=DockerWorkspace("repo@sha256:"+"a"*64,12,client=client)
        self.assertEqual(workspace.execute("echo hello")["stdout"],"public")
        self.assertEqual(calls[0][0],['timeout','--kill-after=5','12','/bin/bash','-lc','echo hello'])
        workspace.close()
        with self.assertRaises(ValueError): DockerWorkspace("repo:latest",12,client=client)

    def test_harbor_real_queue_bridge_and_official_trial_result_contract(self):
        async def exercise(root):
            modules={"harbor":SimpleNamespace(),"harbor.agents":SimpleNamespace(),
                "harbor.agents.base":SimpleNamespace(BaseAgent=object)}
            with patch.dict(sys.modules,modules): agent=importlib.import_module("internalization.benchmarks.harbor_agent")
            env_calls=[]
            async def execute(command,**kwargs):
                env_calls.append((command,kwargs))
                return SimpleNamespace(stdout="PUBLIC_OUTPUT",stderr="",return_code=0)
            class Trial:
                @classmethod
                async def create(cls,config):
                    self.assertEqual(config["agent"]["import_path"],"internalization.benchmarks.harbor_agent:InternalizationAgent")
                    self.assertFalse(config["verifier"]["disable"])
                    return cls()
                async def run(self):
                    await agent.serve(agent.SESSION,"PUBLIC_TASK",SimpleNamespace(exec=execute))
                    return SimpleNamespace(exception_info=None,verifier_result=SimpleNamespace(rewards={"reward":.75}))
            modules.update({"harbor.models.trial.config":SimpleNamespace(TrialConfig=SimpleNamespace(model_validate=lambda x:x)),
                            "harbor.trial.trial":SimpleNamespace(Trial=Trial)})
            config=BenchmarkConfig("terminalbench2","unused")
            with patch.dict(sys.modules,modules),patch("internalization.benchmarks.terminalbench.version",return_value="0.23.0"):
                session=HarborSession(config,root)
                self.assertEqual((await session.start(root))["instruction"],"PUBLIC_TASK")
                first=await session.step('{"action":"exec","command":"pwd"}',False)
                self.assertIn("PUBLIC_OUTPUT",first["observation"])
                result=await session.step('{"action":"final"}',False)
                self.assertEqual(result["reward"],.75)
                self.assertTrue(result["terminal"])
                self.assertEqual(env_calls,[("pwd",{"timeout_sec":120})])
                await session.close()
        with tempfile.TemporaryDirectory() as d: asyncio.run(exercise(Path(d)))

    def test_harbor_trial_failure_cannot_be_reported_as_success(self):
        async def exercise():
            session=HarborSession(BenchmarkConfig("terminalbench2","unused"),Path("unused"))
            async def fail(): return SimpleNamespace(exception_info="infra failure")
            session.task=asyncio.create_task(fail())
            with self.assertRaises(RuntimeError): await session.receive()
        asyncio.run(exercise())

    def test_lawbench_category_aggregation_calls_whole_category_scorer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            records=[{"id":f"1-2:{i}","split":"test","category":"1-2",
                      "record":{"instruction":"I","question":str(i),"answer":"SECRET"}} for i in range(2)]
            config=make_config(root,"lawbench",records)
            for i in range(2):
                out=root/"environments"/str(i); out.mkdir(parents=True)
                (out/"grade.json").write_text(json.dumps({"task_id":f"1-2:{i}","seed":0,"category":"1-2",
                                                       "prediction":"P","metrics":{"score":.1}}))
            def run(argv,**kwargs):
                request=json.loads(Path(argv[argv.index("--request")+1]).read_text())
                self.assertEqual(len(request["predictions"]["1-2"]),2)
                Path(argv[argv.index("--response")+1]).write_text(json.dumps({"1-2":{"score":.7,"abstention_rate":.1}}))
            with patch("internalization.benchmarks.aggregate.subprocess.run",side_effect=run):
                result=aggregate(config,root,[EpisodeResult(f"1-2:{i}",0,.1,Cost()) for i in range(2)])
            self.assertEqual(result["macro_observed_categories"],.7)
            self.assertIsNone(result["official_20_category_macro"])
            self.assertFalse(result["full_dataset_verified"])


if __name__ == "__main__": unittest.main()
