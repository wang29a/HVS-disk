"""Bounded owned-process execution; interrupted points remain resumable."""
import os
import signal
import subprocess


class DeferredMeasurement(RuntimeError):
    pass


def stop_owned_group(child, grace=10):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Failure to reap is fatal; never start another experiment beside an
    # uninterruptible child whose ownership/resources are still outstanding.
    child.wait(timeout=grace)
