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

# Adapted from OPID agent_system/environments/env_package/webshop/projection.py.
# Keep legal-action validity in the existing adapter. Do not import language or
# think-tag penalties, or turn malformed prose into a last-20-character action.
import re


def webshop_action(response):
    """Extract one tagged search/click command; retain the bare-command interface.

    The caller retains the original response and response IDs for training.
    Malformed/multiple-tag text remains invalid under the adapter's existing
    legal-action check, instead of becoming an automatic action.
    """
    if re.fullmatch(r"(?:search|click)\[.+\]", response.strip(), re.DOTALL):
        return response.strip()
    lowered = response.lower()
    if lowered.count("<action>") == 1 and lowered.count("</action>") == 1:
        match = re.search(r"<action>\s*((?:search|click)\[.+\])\s*</action>", lowered, re.DOTALL)
        if match:
            return match[1].strip()
    return response.strip()
