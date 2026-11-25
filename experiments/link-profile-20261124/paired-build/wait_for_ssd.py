"""Wait for the named topology campaign to finish SSD before starting Phase 0."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_status(path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {"state": "UNAVAILABLE"}


TERMINAL = {"PHASE0B_COMPLETE", "PHASE1_COMPLETE"}


def topology_active(session):
    return subprocess.run(["tmux", "has-session", "-t", session],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ready(status, session_active):
    return status.get("state") in TERMINAL and not session_active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--peer-session", required=True)
    parser.add_argument("--peer", required=True, type=Path)
    args, options = parser.parse_known_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    local_journal = root / "logs/topology-peer.jsonl"
    peer_journal = args.peer / "logs/paired-build-peer.jsonl"
    peer_status = args.peer / "status.json"
    previous = read_status(root / "status.json")

    def announce(state, other, seconds=None):
        event = dict(time=time.time(), paired_state=state, topology_state=other.get("state"),
                     topology_task=other.get("task"), topology_item=other.get("item"))
        if seconds is not None:
            event["wait_seconds"] = seconds
        line = json.dumps(event) + "\n"
        for journal in (local_journal, peer_journal):
            with journal.open("a", encoding="utf-8") as f:
                f.write(line)
        print(line, end="", flush=True)
        snapshot = dict(previous, state=state, time=event["time"], topology=other)
        temp = root / "status.waiting.partial"
        temp.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        temp.replace(root / "status.json")

    start = time.monotonic()
    other = read_status(peer_status)
    if not ready(other, topology_active(args.peer_session)):
        announce("WAITING_FOR_TOPOLOGY_SSD", other)
        while True:
            time.sleep(5)
            other = read_status(peer_status)
            if ready(other, topology_active(args.peer_session)):
                break
    announce("TOPOLOGY_SSD_COMPLETE", other, time.monotonic() - start)
    os.environ["HVS_PEER_STATUS"] = str(peer_status)
    runner = Path(__file__).with_name("run_phase0.py")
    os.execv(sys.executable, [sys.executable, "-B", "-u", str(runner), str(root), *options])


if __name__ == "__main__":
    main()
