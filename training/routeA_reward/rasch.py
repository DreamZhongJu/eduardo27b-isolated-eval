"""Rasch MAP ability estimation for Route A ladder post-test rewards.

Given graded observations ``{(b_i, y_i)}_{i=1}^N`` with item difficulties ``b_i``
calibrated on the frozen untutored student (``b = ln((1 - p) / p)``,
``p = (n_correct + 0.5) / 17``), maximizes the Rasch log-posterior under a
standard normal prior ``N(0, 1)``:

    L(theta) = sum_i [ y_i * ln(sigma(theta - b_i))
                       + (1 - y_i) * ln(1 - sigma(theta - b_i)) ]
               - 0.5 * theta^2

Because ``L''(theta) <= -1`` everywhere, ``L`` is strictly strongly concave and
has a unique finite global maximum for every non-empty observation sequence,
including all-correct and all-wrong cases. Since difficulties are calibrated on
the frozen student, pre-tutoring ability is 0 and gain is ``G = theta_hat``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

RASCH_NEWTON_TOL: float = 1e-10
RASCH_NEWTON_MAX_ITER: int = 50


@dataclass(frozen=True)
class RaschEstimate:
    """Output of Rasch MAP estimation."""

    theta: float | None
    valid: bool
    n_obs: int
    n_correct: int
    all_correct: bool
    all_wrong: bool
    prior_pulled: bool
    iterations: int
    converged: bool
    log_posterior: float | None


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid ``1 / (1 + exp(-z))``."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def log_sigmoid(z: float) -> float:
    """Numerically stable ``ln(sigmoid(z))``."""
    if z >= 0.0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def log1m_sigmoid(z: float) -> float:
    """Numerically stable ``ln(1 - sigmoid(z))``."""
    if z >= 0.0:
        return -z - math.log1p(math.exp(-z))
    return -math.log1p(math.exp(z))


def difficulty_from_solve_count(n_correct: int, n_total: int = 16) -> tuple[float, float]:
    """Compute smoothed solve rate ``p`` and Rasch difficulty ``b``.

    For ``n_total = 16`` (T-023 pre-test): ``p = (n_correct + 0.5) / 17`` and
    ``b = ln((1 - p) / p)``.
    """
    if isinstance(n_correct, bool) or isinstance(n_total, bool):
        raise TypeError("n_correct and n_total must be integers, not bool")
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if not (0 <= n_correct <= n_total):
        raise ValueError(f"n_correct={n_correct} out of range [0, {n_total}]")
    p = (n_correct + 0.5) / (n_total + 1.0)
    b = math.log((1.0 - p) / p)
    return p, b


def difficulty_from_rate(p: float) -> float:
    """Compute Rasch item difficulty ``b = ln((1 - p) / p)`` for ``0 < p < 1``."""
    pf = float(p)
    if not (math.isfinite(pf) and 0.0 < pf < 1.0):
        raise ValueError(f"solve rate p must lie strictly in (0, 1), got {p!r}")
    return math.log((1.0 - pf) / pf)


def _validate_observations(
    observations: Sequence[tuple[float, int]],
) -> list[tuple[float, int]]:
    cleaned: list[tuple[float, int]] = []
    for item in observations:
        if len(item) != 2:
            raise ValueError("each observation must be a (difficulty_b, y) 2-tuple")
        b_raw, y_raw = item
        if isinstance(b_raw, bool):
            raise TypeError("difficulty b must be numeric, not bool")
        b = float(b_raw)
        if not math.isfinite(b):
            raise ValueError(f"difficulty b must be finite, got {b_raw!r}")
        if isinstance(y_raw, bool):
            y = int(y_raw)
        elif isinstance(y_raw, int) and y_raw in (0, 1):
            y = y_raw
        else:
            raise ValueError(f"outcome y must be 0 or 1, got {y_raw!r}")
        cleaned.append((b, y))
    return cleaned


def rasch_log_posterior(
    theta: float,
    observations: Sequence[tuple[float, int]],
) -> float:
    """Evaluate Rasch log-posterior (up to an additive constant) at ``theta``."""
    total = -0.5 * theta * theta
    for b, y in observations:
        z = theta - b
        total += log_sigmoid(z) if y == 1 else log1m_sigmoid(z)
    return total


def rasch_map_estimate(
    observations: Sequence[tuple[float, int]],
    *,
    tol: float = RASCH_NEWTON_TOL,
    max_iter: int = RASCH_NEWTON_MAX_ITER,
) -> RaschEstimate:
    """Compute the Rasch MAP ability estimate via Newton-Raphson.

    Returns an invalid ``RaschEstimate`` (``valid=False``, ``theta=None``) when
    ``observations`` is empty (e.g., every ladder attempt suffered a technical
    failure).
    """
    if tol <= 0.0 or not math.isfinite(tol):
        raise ValueError(f"tol must be finite and positive, got {tol!r}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter!r}")

    obs = _validate_observations(observations)
    n_obs = len(obs)
    if n_obs == 0:
        return RaschEstimate(
            theta=None,
            valid=False,
            n_obs=0,
            n_correct=0,
            all_correct=False,
            all_wrong=False,
            prior_pulled=False,
            iterations=0,
            converged=False,
            log_posterior=None,
        )

    theta = 0.0
    cur_lp = rasch_log_posterior(theta, obs)
    converged = False
    iters = 0

    for it in range(1, max_iter + 1):
        iters = it
        grad = -theta
        neg_hess = 1.0
        for b, y in obs:
            p = sigmoid(theta - b)
            grad += y - p
            neg_hess += p * (1.0 - p)

        step = grad / neg_hess
        if abs(grad) < tol and abs(step) < tol:
            converged = True
            break

        # Backtracking safeguard guarantees monotone ascent even for extreme inputs.
        alpha = 1.0
        next_theta = theta + step
        next_lp = rasch_log_posterior(next_theta, obs)
        while next_lp < cur_lp and alpha > 1e-12:
            alpha *= 0.5
            next_theta = theta + alpha * step
            next_lp = rasch_log_posterior(next_theta, obs)

        delta = next_theta - theta
        theta = next_theta
        cur_lp = next_lp

        grad_after = -theta + sum(y - sigmoid(theta - b) for b, y in obs)
        if abs(delta) < tol and abs(grad_after) < tol:
            converged = True
            break

    n_correct = sum(y for _, y in obs)
    all_correct = n_correct == n_obs
    all_wrong = n_correct == 0
    return RaschEstimate(
        theta=float(theta),
        valid=True,
        n_obs=n_obs,
        n_correct=n_correct,
        all_correct=all_correct,
        all_wrong=all_wrong,
        prior_pulled=all_correct or all_wrong,
        iterations=iters,
        converged=converged,
        log_posterior=float(cur_lp),
    )


def rasch_grid_search(
    observations: Sequence[tuple[float, int]],
    *,
    low: float = -12.0,
    high: float = 12.0,
    num_points: int = 24001,
) -> float:
    """Reference grid search for testing Newton-Raphson convergence."""
    obs = _validate_observations(observations)
    if not obs:
        raise ValueError("observations must be non-empty for grid search")
    if num_points < 2 or not (low < high):
        raise ValueError("invalid grid range")
    step = (high - low) / (num_points - 1)
    best_theta = low
    best_lp = rasch_log_posterior(low, obs)
    for idx in range(1, num_points):
        th = low + idx * step
        lp = rasch_log_posterior(th, obs)
        if lp > best_lp:
            best_lp = lp
            best_theta = th
    return best_theta
