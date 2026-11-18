"""Pause a running experiment between paid requests, retaining every receipt."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pid", type=int)
parser.add_argument("output", type=Path)
args = parser.parse_args()


terminal = set()

def active_calls():
    count = 0
    for p in args.output.glob("*/calls/*.json"):
        if p in terminal:
            continue
        state = json.loads(p.read_text()).get("state")
        if state == "in_flight":
            count += 1
        else:
            terminal.add(p)
    return count


deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    try:
        os.kill(args.pid, 0)
    except ProcessLookupError:
        print("Original process has already exited")
        break
    if not active_calls():
        os.kill(args.pid, signal.SIGSTOP)
        # Recheck with the process frozen to exclude the race before SIGSTOP.
        if not active_calls():
            os.kill(args.pid, signal.SIGINT)
            os.kill(args.pid, signal.SIGCONT)
            print("Interrupted at a checkpoint with no paid request in flight", flush=True)
            for _ in range(100):
                try:
                    os.kill(args.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Original process did not exit; do not start a duplicate")
            break
        os.kill(args.pid, signal.SIGCONT)
    time.sleep(0.001)
else:
    raise TimeoutError("No checkpoint reached; original process was left running")
