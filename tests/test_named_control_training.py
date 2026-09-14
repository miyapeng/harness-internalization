"""Reuse the same CPU optimizer assertions with an independently retained named control."""
from unittest.mock import patch
import json

import test_revision_training as legacy
from internalization.harness.revision import InternalizationTarget


class NamedControlTrainingTests(legacy.RevisionTrainingTests):
    def run_training(self,root,**options):
        original=legacy.improvement
        def improvement(store,parent):
            candidate=original(store,parent)
            files=candidate.full_revision.files()
            files['controls/retained.py']='def run(api,payload): return {"suffix":"\\nRETAINED","selected":True}\n'
            files['config/harness.json']=json.dumps({'schema':2,'entrypoint':parent.config['entrypoint'],
                'composition':'independent_suffix','controls':[
                    {'id':'diagnosis_v1','entrypoint':'controls/diagnosis.py:augment','enabled':True},
                    {'id':'retained_v1','entrypoint':'controls/retained.py:run','enabled':True}]})
            from dataclasses import replace
            return replace(candidate,full_revision=store.snapshot(files))
        def reduction(store,full):
            return InternalizationTarget.from_control(store,full,'diagnosis_v1','diagnosis only')
        with patch.object(legacy,'improvement',improvement),patch.object(legacy,'reduction',reduction):
            policy=super().run_training(root,**options)
        # All recorded student prompts retain the other control; teacher target is never leaked.
        for path in (root/'training').glob('rollout_*/trajectories.jsonl'):
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if row['kind']=='transition': self.assertIn('RETAINED',row['student_prompt'])
        return policy
