"""Parent-side capability broker. Path validation is not the execution sandbox."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from .revision import validate_entrypoint


@dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: float = 30
    cpu_seconds: int = 5
    memory_mb: int = 256
    max_rpc_calls: int = 16
    max_output_bytes: int = 1048576

    def __post_init__(self):
        if any(value<=0 for value in asdict(self).values()): raise ValueError("Positive sandbox limits required")


class SandboxedCode:
    def __init__(self,limits=SandboxLimits()): self.limits=limits

    def call(self,revision,entrypoint,payload,broker,*,audit=None,hide_control_config=False):
        files=revision.files()
        validate_entrypoint(entrypoint,files,revision.policy)
        limits=self.limits
        request={"root":revision.path,"entrypoint":entrypoint,"payload":payload,
                 "memory_mb":limits.memory_mb,"cpu_s":limits.cpu_seconds}
        if hide_control_config:
            if entrypoint.split(":")[0]=="config/harness.json": raise ValueError("Invalid control entrypoint")
            request["readable_files"]=[p for p in files if p!="config/harness.json"]
        # -I/-S removes ambient PYTHONPATH/site customization. Do not pass credentials.
        process=subprocess.Popen([sys.executable,"-I","-S",str(Path(__file__).with_name("sandbox_worker.py"))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            env={"LANG":"C.UTF-8","PYTHONDONTWRITEBYTECODE":"1"},cwd=revision.path,close_fds=True)
        deadline=time.monotonic()+limits.wall_seconds
        pending=b"";used=0;calls=0;isolated=False;result=None;returned=False
        try:
            process.stdin.write((json.dumps(request)+"\n").encode());process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout,selectors.EVENT_READ,"stdout")
                selector.register(process.stderr,selectors.EVENT_READ,"stderr")
                while not returned:
                    left=deadline-time.monotonic()
                    if left<=0: raise TimeoutError("Harness candidate runtime timed out")
                    events=selector.select(min(left,1.))
                    if not events and process.poll() is not None: raise RuntimeError("Candidate worker exited before result")
                    for key,_ in events:
                        chunk=os.read(key.fileobj.fileno(),65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        used+=len(chunk)
                        if used>limits.max_output_bytes: raise ValueError("Candidate exceeded output budget")
                        if key.data=="stderr": continue
                        pending+=chunk
                        while b"\n" in pending:
                            line,pending=pending.split(b"\n",1)
                            message=json.loads(line)
                            if message["event"]=="isolated":
                                if isolated or message["landlock_abi"]<3 or message["seccomp"] is not True:
                                    raise RuntimeError("Required candidate isolation unavailable")
                                isolated=True
                                if audit: audit.append("candidate_isolation",revision=revision.version,**message)
                            elif message["event"]=="isolation_error":
                                raise RuntimeError(message["error"])
                            elif not isolated: raise RuntimeError("Candidate executed without required isolation")
                            elif message["event"]=="request":
                                calls+=1
                                if calls>limits.max_rpc_calls: raise ValueError("Candidate capability budget exceeded")
                                operation=message.pop("operation");message.pop("event")
                                # Broker, not candidate-reported costs/rewards, owns all effects.
                                value=broker(operation,message)
                                if time.monotonic()>deadline: raise TimeoutError("Candidate capability exceeded wall budget")
                                process.stdin.write((json.dumps({"ok":True,"result":value})+"\n").encode());process.stdin.flush()
                            elif message["event"]=="result": result,returned=message["result"],True
                            else: raise RuntimeError(message.get("error","Invalid candidate protocol"))
            revision.files()
            return result
        finally:
            if process.poll() is None: process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin,process.stdout,process.stderr): stream.close()
