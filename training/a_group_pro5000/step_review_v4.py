"""Per-token step review using the native veRL counters."""
from numbers import Real
import math

PHASES = ('gen', 'old_log_prob', 'update_actor')
TOKEN_KEY = 'perf/total_num_tokens'
MAX_RATIO = 1.3


def per_token(metrics):
    tokens = metrics.get(TOKEN_KEY)
    if isinstance(tokens, bool) or not isinstance(tokens, Real) or not math.isfinite(tokens) or tokens <= 0:
        raise ValueError('训练步缺少有效总 token 数')
    result = {}
    for name in PHASES:
        seconds = metrics.get(f'timing_s/{name}')
        if isinstance(seconds, bool) or not isinstance(seconds, Real) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('训练步缺少有效阶段计时')
        result[name] = {'minutes': seconds / 60, 'ms_per_token': seconds * 1000 / tokens}
    return result


def compare_steps(first, second):
    names = ('old_log_prob', 'update_actor')
    before, after = (per_token(row['metrics']) for row in (first, second))
    ratios = {name: after[name]['ms_per_token'] / before[name]['ms_per_token'] for name in names}
    return {'ratios': ratios, 'limit': MAX_RATIO, 'passed': all(ratio <= MAX_RATIO for ratio in ratios.values())}
