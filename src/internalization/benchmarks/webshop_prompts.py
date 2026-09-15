# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Base template retained from pinned OPID; rendering below is project-owned.
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.\x20
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:\x20
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

def render_public_history(history, available):
    """Render once from full public page/action history and public legal actions.

    Keep all existing history, rather than introducing OPID's memory window.
    Never accept a goal, evaluator result, infos object, or catalog row here.
    """
    actions = []
    if available["has_search_bar"]:
        actions.append("search[<your query>]")
    actions.extend(f"click[{text}]" for text in available["clickables"])
    # The task is already present in the public page. Refer to it without
    # duplicating the page or reading the simulator's hidden goal object.
    return WEBSHOP_TEMPLATE_NO_HIS.format(
        task_description="follow the shopping instruction in the public page/history below",
        current_observation="\n".join(history),
        available_actions="\n".join(f"'{action}'," for action in actions))
