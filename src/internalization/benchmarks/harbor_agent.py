"""Harbor's documented BaseAgent plugin. Imported only inside the Harbor worker."""
import asyncio
import json

from harbor.agents.base import BaseAgent
from .common import action_json

SESSION = None


async def serve(session, instruction, environment):
    await session.observations.put({"instruction":instruction})
    while True:
        action, force_finish = await session.actions.get()
        valid, calls, finish = True, 0, False
        try:
            item = action_json(action)
            if item["action"] == "exec":
                if not isinstance(item["command"], str): raise ValueError("Command must be a string")
                try:
                    result = await environment.exec(item["command"], timeout_sec=session.config.command_timeout_s)
                    observation = json.dumps({"stdout":result.stdout, "stderr":result.stderr,
                                              "return_code":result.return_code})
                except asyncio.TimeoutError:
                    observation = "Command timed out."
                calls = 1
            elif item["action"] == "final": finish, observation = True, "Task submitted."
            else: raise ValueError("Unknown action")
        except (ValueError, KeyError, TypeError) as exc:
            valid, observation = False, "Invalid action: "+str(exc)
        finish = finish or force_finish
        await session.observations.put({"observation":observation, "valid":valid,
                                        "tool_calls":calls, "finish":finish})
        if finish: return


class InternalizationAgent(BaseAgent):
    @staticmethod
    def name(): return "harness-internalization"

    def version(self): return "1"

    async def setup(self, environment): pass

    async def run(self, instruction, environment, context):
        if SESSION is None: raise RuntimeError("Missing internalization worker session")
        await serve(SESSION, instruction, environment)
