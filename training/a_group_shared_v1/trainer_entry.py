#!/usr/bin/env python3
"""Run the unmodified author entry point and shut down only this Ray client."""
import signal

import ray
from verl.trainer.main_ppo import main


def stop(_signal, _frame):
    raise SystemExit('收到停止信号')


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    main()
finally:
    if ray.is_initialized():
        ray.shutdown()
