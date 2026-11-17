#!/usr/bin/env python3
"""Hold most free memory on one pre-checked CUDA device until the parent stops."""
import ctypes
import json
import os
import signal
import sys
import time

def main():
    parent = int(sys.argv[1])
    # Avoid leaking a reservation if the controller crashes or is killed.
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise RuntimeError('cannot configure parent-death signal')
    if os.getppid() != parent:
        raise RuntimeError('controller already exited')
    import torch
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    # Require a practically empty 80 GiB GPU after CUDA context creation.
    if total < 78000 * 1024**2 or free < total - 2 * 1024**3:
        raise RuntimeError('GPU no longer free')
    amount = free - 1536 * 1024**2
    hold = torch.empty(amount, dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize()
    print(json.dumps({'state': 'reserved', 'bytes': amount}), flush=True)
    while True:
        time.sleep(5)
        free, _ = torch.cuda.mem_get_info(0)
        if free < 512 * 1024**2:
            # A newcomer or external load consumed the headroom: release.
            print(json.dumps({'state': 'released_due_to_load'}), flush=True)
            return

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(json.dumps({'state': 'failed', 'type': type(e).__name__}), flush=True)
        sys.exit(1)
