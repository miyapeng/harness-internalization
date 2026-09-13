"""Trusted isolated-process bootstrap. Never import this file to execute a candidate in host.

Requires Linux Landlock ABI >=3 and libseccomp. All candidate execution happens only
after both irreversible kernel restrictions are installed. No fallback is provided.
"""
import ctypes
import errno
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import sysconfig


def confine(root, memory_mb, cpu_s):
    libc = ctypes.CDLL(None,use_errno=True)
    abi = libc.syscall(444,0,0,1)
    if abi < 3: raise RuntimeError("Required Landlock ABI >=3 unavailable")
    sec = ctypes.CDLL("libseccomp.so.2",use_errno=True)
    resource.setrlimit(resource.RLIMIT_AS,(memory_mb*1024*1024,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(cpu_s,cpu_s+1))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    class Rules(ctypes.Structure): _fields_=[("fs",ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_=1
        _fields_=[("access",ctypes.c_uint64),("fd",ctypes.c_int)]
    # Handle all filesystem rights through ABI3, including REFER and TRUNCATE.
    rules=Rules((1<<15)-1)
    fd=libc.syscall(444,ctypes.byref(rules),ctypes.sizeof(rules),0)
    if fd<0: raise OSError(ctypes.get_errno(),"landlock_create_ruleset")
    try:
        for path in {str(root),sysconfig.get_path("stdlib"),sysconfig.get_path("platstdlib")}:
            parent=os.open(path,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=PathRule((1<<2)|(1<<3),parent)
                if libc.syscall(445,fd,1,ctypes.byref(rule),0): raise OSError(ctypes.get_errno(),"landlock_add_rule")
            finally: os.close(parent)
        if libc.prctl(38,1,0,0,0) or libc.syscall(446,fd,0):
            raise OSError(ctypes.get_errno(),"landlock_restrict_self")
    finally: os.close(fd)
    sec.seccomp_init.argtypes=[ctypes.c_uint32];sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p];sec.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x00050000|errno.EPERM)
    if not ctx: raise RuntimeError("seccomp_init failed")
    try:
        allowed="read write close fstat newfstatat stat lstat statx lseek openat open readlink readlinkat getdents64 mmap mprotect munmap mremap brk rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack fcntl ioctl poll ppoll select pselect6 futex clock_gettime clock_nanosleep nanosleep getrandom getpid getppid gettid getuid geteuid getgid getegid getcwd uname getrlimit exit exit_group".split()
        for name in allowed:
            nr=sec.seccomp_syscall_resolve_name(name.encode())
            if nr>=0 and sec.seccomp_rule_add(ctx,0x7fff0000,nr,0): raise RuntimeError("seccomp allow rule failed")
        if sec.seccomp_load(ctx): raise RuntimeError("seccomp_load failed")
    finally: sec.seccomp_release(ctx)
    return {"landlock_abi":abi,"seccomp":True}


class API:
    def __init__(self,wire): self.wire=wire
    def request(self,operation,**payload):
        self.wire.write(json.dumps({"event":"request","operation":operation,**payload})+"\n");self.wire.flush()
        response=json.loads(sys.stdin.readline())
        if not response["ok"]: raise RuntimeError(response["error"])
        return response["result"]
    def model(self,prompt): return self.request("model",prompt=prompt)
    def environment(self,action): return self.request("environment",action=action)


def main():
    request=json.loads(sys.stdin.readline())
    root=Path(request["root"]).resolve(strict=True)
    # The only inherited descriptors are protocol stdin/stdout/stderr; no model/env handles.
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(root))
    wire=sys.stdout
    try:
        isolation=confine(root,request["memory_mb"],request["cpu_s"])
    except BaseException as exc:
        wire.write(json.dumps({"event":"isolation_error","error":"Required candidate isolation unavailable: "+str(exc)})+"\n")
        wire.flush()
        raise SystemExit(1)
    wire.write(json.dumps({"event":"isolated",**isolation})+"\n");wire.flush()
    try:
        filename,function=request["entrypoint"].split(":")
        spec=importlib.util.spec_from_file_location("candidate_entry",root/filename)
        module=importlib.util.module_from_spec(spec)
        sys.stdout=sys.stderr  # Ordinary candidate prints are not protocol messages.
        spec.loader.exec_module(module)
        result=getattr(module,function)(API(wire),request["payload"])
        wire.write(json.dumps({"event":"result","result":result},allow_nan=False)+"\n");wire.flush()
    except BaseException as exc:
        wire.write(json.dumps({"event":"error","error":type(exc).__name__+": "+str(exc)})+"\n");wire.flush()
        raise SystemExit(1)


if __name__=="__main__": main()
