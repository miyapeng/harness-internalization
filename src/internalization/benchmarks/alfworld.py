# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Formatting adapted from env_manager.py and memory/memory.py; environment
# behavior is provided entirely by the externally installed alfworld package.
from __future__ import annotations

from pathlib import Path
from .alfworld_prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
from .alfworld_projection import alfworld_projection
from ..training.rollout import EnvironmentStep


class AlfworldPrompt:
    def __init__(self, history_length=5):
        self.history_length = history_length
        self.history = []
        self.task = ""

    def reset(self, observation, admissible):
        self.history = []
        marker = "Your task is to: "
        if marker not in observation: raise ValueError("Task description missing from ALFWorld observation")
        self.task = observation.split(marker, 1)[1].strip()
        self.previous = observation
        return self.format(observation, admissible, initial=True)

    def advance(self, action, observation, admissible):
        self.history.append((self.previous, action))
        self.previous = observation
        return self.format(observation, admissible)

    def format(self, observation, admissible, initial=False):
        actions = "\n ".join(f"'{s}'" for s in admissible if s != "help")
        if initial or self.history_length <= 0:
            return ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=observation, admissible_actions=actions)
        recent = self.history[-self.history_length:]
        start = len(self.history) - len(recent)
        history = "\n".join(f"[Observation {start+i+1}: '{obs}', Action {start+i+1}: '{action}']"
                            for i, (obs, action) in enumerate(recent))
        return ALFWORLD_TEMPLATE.format(task_description=self.task, step_count=len(self.history),
            history_length=len(recent), action_history=history, current_step=len(self.history)+1,
            current_observation=observation, admissible_actions=actions)


class AlfworldEnvironment:
    def __init__(self, config_path, *, history_length=5, environment_class=None):
        self.config_path, self.environment_class = Path(config_path), environment_class
        self.prompt = AlfworldPrompt(history_length)
        self.env = None

    def reset(self, task_id, seed):
        import yaml
        environment_class = self.environment_class
        if environment_class is None:
            from alfworld.agents.environment import AlfredTWEnv
            environment_class = AlfredTWEnv
        task = Path(task_id).resolve(strict=True)
        if task.name != "game.tw-pddl": raise ValueError("Expected real ALFWorld game path")
        config = yaml.safe_load(self.config_path.read_text())
        config["general"]["use_cuda"] = False
        base = environment_class(config, train_eval="train")
        base.game_files, base.num_games = [str(task)], 1
        self.env = base.init_env(batch_size=1)
        self.env.seed(seed)
        observations, infos = self.env.reset()
        if Path(infos["extra.gamefile"][0]).resolve() != task: raise ValueError("Environment returned wrong task")
        self.admissible = infos["admissible_commands"][0]
        return self.prompt.reset(observations[0], self.admissible)

    def step(self, action):
        actions, valid = alfworld_projection([action], [self.admissible])
        observations, _, dones, infos = self.env.step(actions)
        self.admissible = infos["admissible_commands"][0]
        success = float(infos["won"][0])
        public = self.prompt.advance(actions[0], observations[0], self.admissible)
        return EnvironmentStep(public, 10.0*success, bool(dones[0]), success, bool(valid[0]))

    def close(self):
        if self.env is not None: self.env.close()
