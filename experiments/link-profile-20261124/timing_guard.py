"""Admit and verify timed jobs on idle physical cores and their SMT siblings."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import threading
import time

from resume_scan import occupancy, siblings, ticks
from run import status


LIMIT = .10
WINDOW = 5


def snapshot(groups, seconds=WINDOW):
    cpus = set().union(*groups.values())
    initial = ticks(cpus)
    time.sleep(seconds)
    final = ticks(cpus)
    return {name: occupancy(initial, final, group) for name, group in groups.items() if group}


@contextmanager
def quiet_window(root, key, lock, cpus, max_busy=LIMIT):
    """Only hold the experiment lock once the requested CPUs are quiet."""
    groups = {"workers": cpus, "siblings": siblings(cpus)}
    if not groups["siblings"]:
        raise RuntimeError("expected SMT siblings for the timing CPUs")
    while True:
        before = snapshot(groups)
        if max(before.values()) <= max_busy:
            with lock.open("rb") as guard:
                try:
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    status(root, "WAITING_FOR_EXPERIMENT_LOCK", item=key)
                else:
                    before = snapshot(groups)
                    if max(before.values()) <= max_busy:
                        yield before
                        return
        else:
            status(root, "WAITING_FOR_QUIET_CPUS", item=key, preflight=before)
        time.sleep(15)


def guarded_call(cores, operation, admission):
    other = siblings(cores)
    stopped = threading.Event()
    samples = []
    problems = []

    def watch():
        try:
            begin = ticks(other)
            start = time.monotonic()
            while True:
                done = stopped.wait(WINDOW)
                finish = time.monotonic()
                end = ticks(other)
                if not done or finish - start >= 1:
                    samples.append(occupancy(begin, end, other))
                if done:
                    return
                begin, start = end, finish
        except Exception as error:
            problems.append(str(error))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        result = operation()
    finally:
        stopped.set()
        watcher.join()
    if problems or not samples:
        raise RuntimeError("missing SMT runtime measurements: " + ", ".join(problems))
    return result, dict(preflight=admission, siblings=sorted(other), samples=samples,
                        max_sibling_busy=max(samples), valid=max(samples) <= LIMIT)
