#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/workspace/eduardo27b-isolated-eval
PY=/workspace/eduardo-vllm018/bin/python
DATA=/workspace/data/math-transfer-test-499-49d3420/data/test.jsonl
ROOT=/workspace/results/own-test-499-v1
TEACHER=/workspace/models/Eduardo-27b-3bbf17e
STUDENT=/workspace/models/Meta-Llama-3.1-8B-Instruct
JUDGE="$REPO/agnes_eval_v2"
KEY=/workspace/.secrets/agnes_api_key
CURRENT=PRECHECK
WATCH_PID=

exec 9>/workspace/.own-test-499.lock
if ! flock -n 9; then
  echo '另一个同类控制器正在运行。'
  exit 1
fi
mkdir -p /workspace/results
if [[ -e "$ROOT" ]]; then
  if [[ "${1:-}" != "--resume" || ! -d "$ROOT" ]]; then
    echo '结果目录已存在；需显式使用 --resume 续跑。'
    exit 1
  fi
else
  if [[ "${1:-}" == "--resume" ]]; then echo '没有可续跑的结果目录。'; exit 1; fi
  mkdir "$ROOT"
fi
on_error() {
  printf 'FAILED_%s\n' "$CURRENT" > "$ROOT/STATUS"
  if [[ -n "$WATCH_PID" ]]; then kill -TERM "$WATCH_PID" 2>/dev/null || true; fi
}
on_terminate() {
  printf 'INTERRUPTED_%s\n' "$CURRENT" > "$ROOT/STATUS"
  if [[ -n "$WATCH_PID" ]]; then kill -TERM "$WATCH_PID" 2>/dev/null || true; fi
  exit 130
}
trap on_error ERR
trap on_terminate INT TERM
printf '%s\n' "$CURRENT" > "$ROOT/STATUS"

test -x "$PY"
test -f "$REPO/app/launch.py"
test -f "$TEACHER/config.json"
test -f "$STUDENT/config.json"
test -f "$JUDGE/watch_sources.py"
test -s "$KEY"
echo "5e5aa211549d221d2287529e5a4bac03006ba37d0dc7041c19b386f0eb4a5c01  $DATA" | sha256sum -c - >/dev/null
sha256sum --quiet -c - <<EOF
ddeb906bee82e415882d04790e045a961503ac86ec08e107010dffc18cd8c83b  $TEACHER/config.json
626abc50901718cd92703c081963ff16f42dff5ec45133d76f201999ec7ac4b4  $TEACHER/model-00001-of-00002.safetensors
7be58da8a22afb664ebdfdc9226c2ae30790ccd2a74ff076f8cfff92174d6cbf  $TEACHER/model-00002-of-00002.safetensors
06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523  $TEACHER/tokenizer.json
792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b  $TEACHER/tokenizer_config.json
2fbf099c2eb9284b19cc6188139e6f8aa840a93f4f9fcc937b3b89ae9b4fa837  $TEACHER/chat_template.jinja
29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e  $STUDENT/config.json
79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4  $STUDENT/tokenizer.json
177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424  $STUDENT/tokenizer_config.json
2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668  $STUDENT/model-00001-of-00004.safetensors
09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15  $STUDENT/model-00002-of-00004.safetensors
fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa  $STUDENT/model-00003-of-00004.safetensors
92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b  $STUDENT/model-00004-of-00004.safetensors
EOF

"$PY" - "$REPO" "$DATA" "$JUDGE" "$ROOT" <<'PY'
import hashlib, importlib.metadata as m, json, os, subprocess, sys, tempfile
from pathlib import Path
repo, data, judge, root = map(Path, sys.argv[1:])
for package in ('vllm', 'torch', 'transformers', 'openai', 'math-verify'):
    m.version(package)
if m.version('vllm') != '0.18.0':
    raise RuntimeError('unexpected_vllm_version')
files = [(p.relative_to(repo).as_posix(), p) for d in ('app', 'official')
         for p in (repo / d).rglob('*.py') if p.is_file()]
files += [('judge/' + name, judge / name) for name in
          ('run.sh', 'agnes_judge.py', 'watch_sources.py', 'status.py')]
binding = {'files': {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sorted(files)}}
commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                        capture_output=True, text=True, check=False)
binding['git_commit'] = commit.stdout.strip() if commit.returncode == 0 else None
path = root / 'CODE_BINDING.json'
if path.exists():
    if json.loads(path.read_text(encoding='utf-8')) != binding:
        raise ValueError('saved_code_changed')
else:
    fd, tmp = tempfile.mkstemp(prefix='.code-binding-', dir=root)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(binding, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
sys.path[:0] = [str(repo / 'app'), str(repo / 'official')]
from offline_eval import parse_sources
for difficulty in ('easy', 'medium', 'hard'):
    if len(parse_sources(data, difficulty)) != 499:
        raise ValueError('dataset_count_mismatch')
PY

CURRENT=API_CHECK
printf '%s\n' "$CURRENT" > "$ROOT/STATUS"
mkdir -p "$ROOT/AGNES/locks" "$ROOT/AGNES/health"
"$PY" - "$JUDGE" "$KEY" "$ROOT/AGNES" <<'PY'
from pathlib import Path
import sys, time
judge_dir, key_path, root = map(Path, sys.argv[1:])
sys.path.insert(0, str(judge_dir))
import agnes_judge as judge
row = {'source_id': 'synthetic-api-check', 'target': 'source', 'phase': 'check',
       'index': time.time_ns(), 'problem': 'What is 2+3?', 'reference_answer': '5',
       'student_response': 'Final answer: 5.'}
result = judge.grade_one(row, key_path.read_text().strip(), root / 'health',
                         root / 'locks', window=judge.AdaptiveWindow(5))
if result.get('status') != 'correct':
    raise RuntimeError('api_check_failed')
PY

resources_ready() {
  nvidia-smi --query-gpu=index,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '($1+0)==0 || ($1+0)==1 {count++; if (($3+0)<0.93*($2+0) || ($4+0)>=10) bad=1} END {exit !(count==2 && !bad)}' || return 1
  "$PY" - <<'PY'
import socket
for port in (8001, 8002):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError('local_port_in_use')
PY
}

wait_resources() {
  local attempt
  for attempt in {1..12}; do
    if resources_ready; then return 0; fi
    sleep 10
  done
  return 1
}

run_stage() {
  local label=$1 difficulty=$2 input=$3
  CURRENT=$label
  printf '%s\n' "$CURRENT" > "$ROOT/STATUS"
  if [[ -f "$ROOT/$label/STAGE_FINISHED" ]]; then return 0; fi
  wait_resources
  mkdir -p "$ROOT/$label"
  TEACHER_MODEL="$TEACHER" STUDENT_MODEL="$STUDENT" SOURCE_FILE="$input" \
    RESULTS_DIR="$ROOT/$label" "$PY" "$REPO/app/launch.py" \
    --teacher-gpus 1 --difficulty "$difficulty" >> "$ROOT/$label/controller.log" 2>&1
  : > "$ROOT/$label/STAGE_FINISHED"
}

"$PY" - "$DATA" "$ROOT" <<'PY'
from pathlib import Path
import sys
data, root = map(Path, sys.argv[1:])
lines = data.read_bytes().splitlines(keepends=True)
if len(lines) != 499:
    raise ValueError('dataset_count_mismatch')
for name, content in [('pilot.jsonl', lines[0]), ('easy-rest.jsonl', b''.join(lines[1:]))]:
    path = root / name
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('saved_input_changed')
    else:
        with path.open('xb') as f:
            f.write(content)
PY
run_stage PILOT easy "$ROOT/pilot.jsonl"
"$PY" - "$ROOT/PILOT" <<'PY'
from pathlib import Path
import json, sys
base = Path(sys.argv[1]) / 'sources'
groups = [p for p in base.iterdir() if p.is_dir()] if base.is_dir() else []
if len(groups) != 1:
    raise RuntimeError('pilot_source_count')
p = groups[0]
if (p / 'SOURCE_HELD.json').exists() or not (p / 'RAW_COMPLETE.json').is_file():
    raise RuntimeError('pilot_not_complete')
if json.loads((p / 'RAW_COMPLETE.json').read_text())['samples'] != 96:
    raise RuntimeError('pilot_sample_count')
if len(list((p / 'samples').glob('*/*/*/RESPONSE.json'))) != 96:
    raise RuntimeError('pilot_response_count')
PY

CURRENT=PILOT_GRADING
printf '%s\n' "$CURRENT" > "$ROOT/STATUS"
mkdir -p "$ROOT/AGNES/locks" "$ROOT/AGNES/PILOT" "$ROOT/AGNES/receipts/PILOT"
"$PY" "$JUDGE/watch_sources.py" --sources-dir "$ROOT/PILOT/sources" \
  --output-dir "$ROOT/AGNES/PILOT" --receipts-dir "$ROOT/AGNES/receipts/PILOT" \
  --lock-dir "$ROOT/AGNES/locks" --key-file "$KEY" --workers 5 --once \
  >> "$ROOT/AGNES/pilot.log" 2>&1
"$PY" - "$ROOT/AGNES/PILOT" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1]) / 'sources'
outputs = list(root.glob('*/RESULTS.json'))
if len(outputs) != 1 or list(root.glob('*/HELD.json')):
    raise RuntimeError('pilot_grading_missing')
results = json.loads(outputs[0].read_text())['results']
if len(results) != 96 or sum(x.get('status') in ('correct', 'incorrect') for x in results) < 48:
    raise RuntimeError('pilot_grading_not_working')
PY

CURRENT=START_GRADER
printf '%s\n' "$CURRENT" > "$ROOT/STATUS"
"$PY" "$JUDGE/watch_sources.py" --stages-root "$ROOT" \
  --output-dir "$ROOT/AGNES" --receipts-dir "$ROOT/AGNES/receipts" \
  --lock-dir "$ROOT/AGNES/locks" --key-file "$KEY" --workers 5 \
  >> "$ROOT/AGNES/watch.log" 2>&1 9>&- &
WATCH_PID=$!

run_stage EASY easy "$ROOT/easy-rest.jsonl"
run_stage MEDIUM medium "$DATA"
run_stage HARD hard "$DATA"
CURRENT=WAITING_FOR_GRADES
printf '%s\n' "$CURRENT" > "$ROOT/STATUS"
while ! "$PY" - "$ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for stage in ('EASY', 'MEDIUM', 'HARD'):
    for marker in (root / stage / 'sources').glob('*/RAW_COMPLETE.json'):
        dst = root / 'AGNES' / stage / 'sources' / marker.parent.name
        if not (dst / 'RESULTS.json').is_file() and not (dst / 'HELD.json').is_file():
            raise SystemExit(1)
PY
do
  kill -0 "$WATCH_PID"
  sleep 15
done
kill -TERM "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" || true
WATCH_PID=
printf 'FINISHED_CHECK_COUNTS\n' > "$ROOT/STATUS"
