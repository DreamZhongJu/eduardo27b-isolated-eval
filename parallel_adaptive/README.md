# Two-student evaluation launcher

Independent five-GPU evaluator: three single-GPU teachers and two shared
student services. Per-source pre/control/post student sample requests are
dispatched to either local student service when it has a free slot. The frozen
dialogue student endpoint remains the primary service. Model, prompts, seeds,
source identities and existing grading results are preserved; concurrent
ordering and sampled outputs can differ. One source still has a total of four
concurrent sample requests, so speedup is not guaranteed.

**Not a live upgrade.** Do not run it with an existing writer using the same
results tree. The previous controller cannot drain in-flight sources; stopping
it can leave partial sources which the guard preserves and skips. The root and
stage locks, process guards and frozen-input checks must all pass. Automatic
regrading of historical unknown outcomes is not included in this launcher.

It waits for five unused A100 80GB devices and checks them again before
launch. GPU utilization and free memory are snapshots, not an atomic cross-user
reservation. It uses a synthetic service smoke and a real one-source pilot
per stage before bulk, recording code and routing hashes locally. The only
external grading remains the explicitly configured Agnes watcher; use the
approved grader hashes and keep credentials outside the code and logs.

The newer guard handles SIGUSR1 by finishing the current source and exiting
before attempting another, but the existing controller does not orchestrate
this signal. Do not send signals to an existing unrelated process. Docker exit
stops all work; the container must stay running. Real GPU, model compatibility,
throughput, and prior partial-source recovery still need target verification.

Run synthetic local tests in an environment with openai and httpx installed:
`python -m unittest discover -s parallel_adaptive -p 'test*.py'`.
