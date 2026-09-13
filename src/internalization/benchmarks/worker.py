"""Isolated native environments. JSON responses contain public observations only."""
import argparse
import asyncio
from contextlib import redirect_stdout
import inspect
import json
from pathlib import Path
import sys
import traceback

from .common import BenchmarkConfig, Catalog


def native_environment(config, output):
    Catalog(config.catalog, config.benchmark, config.options.get("catalog_hash"))
    if config.benchmark == "terminalbench2":
        from .terminalbench import TerminalBench2Environment
        return TerminalBench2Environment(config, output)
    if config.benchmark == "swebench_pro":
        from .swebench import SWEBenchProEnvironment
        return SWEBenchProEnvironment(config, output)
    if config.benchmark == "lawbench":
        from .lawbench import LawBenchEnvironment
        return LawBenchEnvironment(config, output)
    if config.benchmark == "hotpotqa":
        from .hotpotqa import HotpotQAEnvironment
        return HotpotQAEnvironment(config, output)
    raise ValueError("Unknown native environment")


async def invoke(fn, **kwargs):
    value = fn(**kwargs)
    return await value if inspect.isawaitable(value) else value


async def run(config, output):
    wire = sys.stdout
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
    with redirect_stdout(sys.stderr):
        env = native_environment(config, output)
        try:
            while line := await reader.readline():
                try:
                    request = json.loads(line)
                    operation = request.pop("op")
                    if operation == "reset":
                        env.training = request.pop("training")
                        result = await invoke(env.reset, **request)
                    elif operation == "step": result = await invoke(env.step, **request)
                    else: raise ValueError("Unknown worker operation")
                    message = {"ok":True, "result":result}
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    message = {"ok":False, "error":type(exc).__name__+": "+str(exc)}
                wire.write(json.dumps(message, ensure_ascii=False, allow_nan=False)+"\n")
                wire.flush()
        finally:
            transport.close()
            await invoke(env.close)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(BenchmarkConfig.load(args.config), args.output))


if __name__ == "__main__": main()
