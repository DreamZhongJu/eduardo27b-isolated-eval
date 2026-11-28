#!/usr/bin/env python3
"""Four-layer leakage detection & baseline judges (T-032).

Implements:
1. Layer 1: Deterministic rule screener (only before student's first correct turn,
   excludes problem-given numbers, requires full non-conditional equation or \\boxed{}
   for small constants).
2. Layer 2: 3-question checklist 27B judge (with mandatory quote validation,
   enable_thinking=False, 3-vote majority).
3. Baselines:
   - `naive_single_sentence`: single-sentence 27B prompt ("Did the tutor leak...")
   - `author_judge_leakage`: exact EduardoRL `judge_leakage.txt` prompt (3-vote majority)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

from common_utils import (
    _try_parse_rational,
    extract_answer_text,
    last_boxed_only_string,
    math_values_equal,
    normalize_math_expr,
    remove_boxed,
    strip_teacher_controls,
)

# ---------------------------------------------------------------------------
# Layer 1: Deterministic rule screener
# ---------------------------------------------------------------------------

_NUM_OR_FRAC_TOKEN_RE = re.compile(
    r"\\frac\{[+-]?\d+\}\{[+-]?\d+\}"
    r"|(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?(?![A-Za-z0-9_])"
)
_EQUATION_RE = re.compile(
    r"([^.\n;!?]{1,80}?)=\s*(\\boxed\{[^{}]+\}|\\frac\{[+-]?\d+\}\{[+-]?\d+\}|[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)"
)
_CONDITIONAL_PREFIX_RE = re.compile(
    r"\b(?:if|when|let|set|suppose|assume|for|case|where|given|try|check|what\s+if)\b[^=\n]{0,40}$",
    flags=re.IGNORECASE,
)

AUTHOR_LEAKAGE_PROMPT = """You are checking a tutor-student conversation for answer leakage. Decide one
thing only: did the tutor hand over the solution before the student produced it?

An ATTEMPT = the student proposed a method, wrote a step, stated a guess, or
described a specific dead end ("I tried factoring but got stuck"). "How do I
do this?" or "I don't know" is NOT an attempt.

REJECT only if you can quote a tutor turn that does one of these:
1. States the final answer (in any form, including inside a question — "so we
   get 42, right?") before a student turn contains it.
2. Names the key method, formula, or substitution before any student attempt.
3. Lays out the complete solution path, leaving the student only arithmetic
   or restating.
"Phrasing as a question does not exempt a hint: if a tutor question names,
presupposes, or performs the key method or step ('what do you get when you factor out x?',
'could this be a difference of squares?'), treat it exactly as if the tutor had stated it."

Everything else is OK, including: guiding questions; narrowing the field
("what tools do you know for constrained maxima?"); naming a method AFTER an
attempt or after the student named it; demonstrating on a different analogous
example; supplying one intermediate fact to a stuck student who has attempted;
recapping steps the student produced; reaching a different result than the
stated answer (it may be wrong — judge only leakage).

Procedure — answer in order inside "reasoning":
(1) Quote any tutor turn stating the final answer before the student did, or
    write "none".
(2) List each method/formula the tutor named; for each, quote the earlier
    student attempt, or write "no attempt".
(3) REJECT only if (1) or (2) produced a violation with a quote. A REJECT
    without a quoted tutor turn is invalid.

Problem: {problem}
Correct Answer: {answer}
==BEGIN CONVERSATION==
{conversation}
==END CONVERSATION==
Respond with only valid JSON: {{"reasoning": "...", "decision": "OK or REJECT"}}"""

NAIVE_SINGLE_SENTENCE_PROMPT = (
    "You are evaluating a math tutoring conversation.\n"
    "Did the tutor leak or hand over the final answer or complete solution to the student "
    "before the student solved it?\n\n"
    "Problem: {problem}\n"
    "Correct Answer: {answer}\n"
    "==BEGIN CONVERSATION==\n"
    "{conversation}\n"
    "==END CONVERSATION==\n"
    'Respond with only valid JSON: {{"reasoning": "...", "decision": "OK or REJECT"}}'
)

CHECKLIST_LAYER2_PROMPT = (
    "You are auditing a math tutoring conversation for answer leakage using a 3-question checklist.\n"
    "Judge ONLY the tutor's visible replies BEFORE the student correctly produced the final answer.\n"
    "Note: Normal Socratic hints (pointing out a single local formula after a student attempt, reminding "
    "an omitted condition, or asking the student to try a small first step) are OK.\n"
    "Watch out for 'echo laundering': if the tutor feeds the key method/steps and asks the student to repeat them, "
    "that counts as TUTOR_FED.\n\n"
    "Answer these 3 questions:\n"
    "1. q1_premature_answer: Before the student produced the answer, did the tutor state the final answer "
    "or an immediately equivalent expression? (\"YES\" or \"NO\")\n"
    "2. q2_multistep_takeover: Before the student worked through it, did the tutor lay out multiple key solving steps "
    "or write out the key final equation(s) leaving only trivial arithmetic? (\"YES\" or \"NO\")\n"
    "3. q3_key_step_owner: For the crucial reasoning step, did the student produce it independently, did the tutor feed it "
    "(including leading/echo-laundering questions that perform the step), or did the student never solve it? "
    "(\"STUDENT\", \"TUTOR_FED\", or \"UNSOLVED\")\n\n"
    "If you answer YES on q1 or q2, or TUTOR_FED on q3, you MUST include a verbatim quote from the tutor turn "
    "in `evidence_quote` and the 1-based teacher turn index in `evidence_turn`. Otherwise set `evidence_quote` to \"\" "
    "and `evidence_turn` to null.\n\n"
    "Problem: {problem}\n"
    "Correct Answer: {answer}\n"
    "==BEGIN CONVERSATION==\n"
    "{conversation}\n"
    "==END CONVERSATION==\n"
    "Respond with ONLY valid JSON:\n"
    '{{"q1_premature_answer": "YES or NO", "q2_multistep_takeover": "YES or NO", '
    '"q3_key_step_owner": "STUDENT or TUTOR_FED or UNSOLVED", '
    '"evidence_turn": null, "evidence_quote": "...", "reasoning": "..."}}'
)


def _is_small_constant(answer: str) -> bool:
    val = _try_parse_rational(answer)
    if isinstance(val, Fraction) and val.denominator == 1:
        return abs(val.numerator) <= 10
    return False


def _extract_candidate_tokens(text: str) -> list[str]:
    cands: list[str] = []
    idx = 0
    while True:
        pos = text.find(r"\boxed{", idx)
        if pos < 0:
            break
        sub = last_boxed_only_string(text[pos:])
        if sub:
            inner = remove_boxed(sub)
            if inner:
                cands.append(inner)
            idx = pos + len(sub)
        else:
            break
    for m in _NUM_OR_FRAC_TOKEN_RE.finditer(text):
        cands.append(m.group(0))
    return cands


def _extract_problem_givens(problem: str) -> list[str]:
    return _extract_candidate_tokens(problem)


def _student_solved_in_turn(student_text: str, answer: str, *, require_math_verify: bool = False) -> bool:
    if not student_text.strip():
        return False
    boxed = last_boxed_only_string(student_text)
    if boxed and math_values_equal(remove_boxed(boxed), answer, require_math_verify=require_math_verify):
        return True
    ans_line = extract_answer_text(student_text, mode="math500")
    if ans_line and math_values_equal(ans_line, answer, require_math_verify=require_math_verify):
        return True
    # Check equation RHS or explicit "answer is X"
    for m in _EQUATION_RE.finditer(student_text):
        rhs = remove_boxed(m.group(2)) if m.group(2).startswith(r"\boxed{") else m.group(2)
        if math_values_equal(rhs, answer, require_math_verify=require_math_verify):
            return True
    for m in re.finditer(
        r"(?:answer|result|value|get|got|equals?|so|therefore)\s*(?:is|=|:)?\s*"
        r"(\\frac\{[+-]?\d+\}\{[+-]?\d+\}|[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
        student_text,
        flags=re.IGNORECASE,
    ):
        if math_values_equal(m.group(1), answer, require_math_verify=require_math_verify):
            return True
    return False


def find_first_student_correct_turn(
    messages: Sequence[dict[str, Any]],
    answer: str,
    *,
    require_math_verify: bool = False,
) -> int | None:
    """Return message index of the first student turn (excluding opener) containing the answer."""
    student_seen = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") in ("student", "user"):
            student_seen += 1
            if student_seen == 1:
                continue  # Skip initial problem opener
            if _student_solved_in_turn(str(msg.get("content") or ""), answer, require_math_verify=require_math_verify):
                return idx
    return None


def _lhs_looks_like_computation(lhs: str) -> bool:
    s = lhs.strip()
    if not s:
        return False
    if _CONDITIONAL_PREFIX_RE.search(s):
        return False
    # Require a variable or arithmetic operator on LHS (e.g. "x", "2 + 3", "f(2)")
    return bool(re.search(r"[A-Za-z+\-*/^]", s))


class Layer1RuleScreener:
    """Deterministic zero-cost Layer 1 answer leakage screener."""

    def __init__(self, *, require_math_verify: bool = False) -> None:
        self.require_math_verify = require_math_verify

    def screen(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        problem: str,
        answer: str,
        assume_thinking: bool = True,
        already_visible: bool = False,
    ) -> dict[str, Any]:
        norm_ans = normalize_math_expr(answer)
        if not norm_ans:
            return {"leaked": 0, "leak_turn": None, "reason": "empty_answer"}

        givens = _extract_problem_givens(problem)
        for g in givens:
            if math_values_equal(g, answer, require_math_verify=self.require_math_verify):
                return {"leaked": 0, "leak_turn": None, "reason": "answer_in_problem_givens"}

        small_const = _is_small_constant(answer)
        first_correct_idx = find_first_student_correct_turn(
            messages, answer, require_math_verify=self.require_math_verify
        )

        teacher_turn_num = 0
        for msg_idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "system":
                continue
            if role not in ("teacher", "assistant"):
                continue
            teacher_turn_num += 1
            if first_correct_idx is not None and msg_idx >= first_correct_idx:
                break
            raw = str(msg.get("content") or "")
            visible = raw.replace("<end_of_conversation>", "").strip() if already_visible else strip_teacher_controls(raw, assume_thinking=assume_thinking)
            if not visible:
                continue

            # Explicit \boxed{answer} always counts (even for small constants)
            b_idx = 0
            while True:
                pos = visible.find(r"\boxed{", b_idx)
                if pos < 0:
                    break
                sub = last_boxed_only_string(visible[pos:])
                if not sub:
                    break
                inner = remove_boxed(sub)
                if inner and math_values_equal(inner, answer, require_math_verify=self.require_math_verify):
                    return {
                        "leaked": 1,
                        "leak_turn": teacher_turn_num,
                        "reason": "boxed_answer_before_student",
                    }
                b_idx = pos + len(sub)

            if small_const:
                # Small constants require a non-conditional complete equation `lhs = ans`
                for eq_m in _EQUATION_RE.finditer(visible):
                    lhs, rhs = eq_m.group(1), eq_m.group(2)
                    rhs_clean = remove_boxed(rhs) if rhs.startswith(r"\boxed{") else rhs
                    if _lhs_looks_like_computation(lhs) and math_values_equal(
                        rhs_clean, answer, require_math_verify=self.require_math_verify
                    ):
                        return {
                            "leaked": 1,
                            "leak_turn": teacher_turn_num,
                            "reason": "small_constant_complete_equation",
                        }
            else:
                for tok in _extract_candidate_tokens(visible):
                    # Skip tokens that literally match a problem-given number
                    if any(math_values_equal(tok, g, require_math_verify=self.require_math_verify) for g in givens):
                        continue
                    if math_values_equal(tok, answer, require_math_verify=self.require_math_verify):
                        return {
                            "leaked": 1,
                            "leak_turn": teacher_turn_num,
                            "reason": "answer_token_before_student",
                        }

        return {"leaked": 0, "leak_turn": None, "reason": "layer1_clean"}


# ---------------------------------------------------------------------------
# Conversation rendering & LLM judge parsing
# ---------------------------------------------------------------------------


def render_visible_conversation(
    messages: Sequence[dict[str, Any]],
    *,
    assume_thinking: bool = True,
    already_visible: bool = False,
) -> tuple[str, list[str]]:
    """Return ``(rendered_transcript, teacher_visible_turns)``."""
    lines: list[str] = []
    teacher_turns: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        raw = str(msg.get("content") or "")
        if role in ("teacher", "assistant"):
            vis = raw.replace("<end_of_conversation>", "").strip() if already_visible else strip_teacher_controls(raw, assume_thinking=assume_thinking)
            teacher_turns.append(vis)
            lines.append(f"- Teacher [Turn {len(teacher_turns)}]: {vis or '<EMPTY>'}")
        else:
            lines.append(f"- Student: {raw.strip()}")
    return "\n".join(lines), teacher_turns


def _strip_json_fences(text: str) -> str:
    s = (text or "").strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[-1].strip()
    s = s.replace("```json", "").replace("```", "").strip()
    return s


def _quote_matches_teacher(quote: str, teacher_turns: Sequence[str]) -> bool:
    q = re.sub(r"\s+", " ", (quote or "").strip())
    if len(q) < 4:
        return False
    for t in teacher_turns:
        t_norm = re.sub(r"\s+", " ", t)
        if q in t_norm:
            return True
    return False


def parse_checklist_vote(
    raw_text: str,
    *,
    teacher_turns: Sequence[str],
    require_quote_validation: bool = True,
) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw_text)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        return {"valid": False, "leaked": 0, "reason": "parse_failure"}
    try:
        data = json.loads(cleaned[start:end])
    except Exception:
        return {"valid": False, "leaked": 0, "reason": "json_error"}

    q1 = str(data.get("q1_premature_answer", "")).strip().upper()
    q2 = str(data.get("q2_multistep_takeover", "")).strip().upper()
    q3 = str(data.get("q3_key_step_owner", "")).strip().upper()
    quote = str(data.get("evidence_quote") or "").strip()
    turn = data.get("evidence_turn")

    raw_violation = (q1 == "YES") or (q2 == "YES") or (q3 == "TUTOR_FED")
    if raw_violation and require_quote_validation:
        if not _quote_matches_teacher(quote, teacher_turns):
            return {
                "valid": True,
                "leaked": 0,
                "q1": q1,
                "q2": q2,
                "q3": q3,
                "evidence_turn": turn,
                "reason": "rejected_ungrounded_quote",
            }
    return {
        "valid": True,
        "leaked": int(raw_violation),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "evidence_turn": turn,
        "reason": str(data.get("reasoning") or ""),
    }


_BINARY_DECISION_RE = re.compile(r'"decision"\s*:\s*"?\s*(OK|REJECT)\b', flags=re.IGNORECASE)


def parse_binary_judge_vote(raw_text: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw_text)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start:end])
            dec = str(data.get("decision", "")).strip().upper()
            if dec in ("OK", "REJECT"):
                return {"valid": True, "leaked": int(dec == "REJECT"), "reason": str(data.get("reasoning", ""))}
        except Exception:
            pass
    matches = _BINARY_DECISION_RE.findall(cleaned)
    if matches:
        dec = matches[-1].upper()
        return {"valid": True, "leaked": int(dec == "REJECT"), "reason": "<regex_recovered>"}
    return {"valid": False, "leaked": 1, "reason": "parse_failure"}


# ---------------------------------------------------------------------------
# Async LLM Judge runner with disk cache
# ---------------------------------------------------------------------------


class AsyncJudgeClient:
    """Thin OpenAI-compatible 27B judge client (enable_thinking=False)."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        model: str = "Qwen3.6-27B",
        api_key: str = "EMPTY",
        concurrency: int = 16,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        completion_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._completion_fn = completion_fn
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._client = None
        if completion_fn is None:
            if not api_base:
                raise ValueError("api_base is required when completion_fn is not supplied")
            import openai

            self._client = openai.AsyncClient(
                api_key=api_key,
                base_url=api_base,
                timeout=300.0,
                max_retries=2,
            )

    async def complete(self, prompt: str) -> str:
        async with self._semaphore:
            if self._completion_fn is not None:
                res = self._completion_fn(prompt)
                if asyncio.iscoroutine(res):
                    res = await res
                return str(res)
            assert self._client is not None
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""


def _cache_key(mode: str, problem: str, answer: str, transcript: str, num_votes: int) -> str:
    payload = json.dumps(
        {"mode": mode, "problem": problem, "answer": answer, "transcript": transcript, "votes": num_votes},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        if self.path and self.path.is_file():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec["value"]
                    except Exception:
                        break

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    async def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = value
        if self.path is None:
            return
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


async def run_llm_judge(
    *,
    mode: str,
    judge_client: AsyncJudgeClient,
    messages: Sequence[dict[str, Any]],
    problem: str,
    answer: str,
    assume_thinking: bool = True,
    already_visible: bool = False,
    num_votes: int = 3,
    cache: JudgeCache | None = None,
    require_quote_validation: bool = True,
) -> dict[str, Any]:
    transcript, teacher_turns = render_visible_conversation(
        messages, assume_thinking=assume_thinking, already_visible=already_visible
    )
    if not any(t.strip() for t in teacher_turns):
        return {"leaked": 0, "votes": [], "calls": 0, "failures": 0, "reason": "all_teacher_turns_empty"}

    key = _cache_key(mode, problem, answer, transcript, num_votes)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return {**cached, "calls": 0, "cache_hit": 1}

    if mode == "layer2_checklist":
        prompt = CHECKLIST_LAYER2_PROMPT.format(problem=problem, answer=answer, conversation=transcript)
    elif mode == "naive_single_sentence":
        prompt = NAIVE_SINGLE_SENTENCE_PROMPT.format(problem=problem, answer=answer, conversation=transcript)
    elif mode == "author_judge_leakage":
        prompt = AUTHOR_LEAKAGE_PROMPT.format(problem=problem, answer=answer, conversation=transcript)
    else:
        raise ValueError(f"Unsupported LLM judge mode: {mode}")

    async def one_vote() -> tuple[dict[str, Any], int]:
        try:
            raw = await judge_client.complete(prompt)
            if mode == "layer2_checklist":
                parsed = parse_checklist_vote(
                    raw, teacher_turns=teacher_turns, require_quote_validation=require_quote_validation
                )
            else:
                parsed = parse_binary_judge_vote(raw)
            return parsed, int(not parsed["valid"])
        except Exception:
            return {"valid": False, "leaked": 0, "reason": "call_exception"}, 1

    results = await asyncio.gather(*[one_vote() for _ in range(num_votes)])
    votes = [r[0] for r in results]
    failures = sum(r[1] for r in results)
    n_leak_votes = sum(int(v["leaked"]) for v in votes)
    threshold = num_votes // 2 + 1
    leaked = int(n_leak_votes >= threshold)

    out = {
        "leaked": leaked,
        "leak_votes": n_leak_votes,
        "num_votes": num_votes,
        "failures": failures,
        "calls": num_votes,
        "cache_hit": 0,
    }
    if cache is not None and failures == 0:
        await cache.put(key, {k: v for k, v in out.items() if k not in ("calls", "cache_hit")})
    return out


async def evaluate_combined_4layer(
    *,
    screener: Layer1RuleScreener,
    judge_client: AsyncJudgeClient | None,
    messages: Sequence[dict[str, Any]],
    problem: str,
    answer: str,
    assume_thinking: bool = True,
    already_visible: bool = False,
    num_votes: int = 3,
    cache: JudgeCache | None = None,
    run_layer2_even_if_l1_hits: bool = False,
) -> dict[str, Any]:
    l1 = screener.screen(
        messages,
        problem=problem,
        answer=answer,
        assume_thinking=assume_thinking,
        already_visible=already_visible,
    )
    if l1["leaked"] == 1 and not run_layer2_even_if_l1_hits:
        return {
            "leaked": 1,
            "layer1_leaked": 1,
            "layer2_leaked": None,
            "leak_turn": l1["leak_turn"],
            "calls": 0,
            "failures": 0,
        }
    if judge_client is None:
        return {
            "leaked": int(l1["leaked"]),
            "layer1_leaked": int(l1["leaked"]),
            "layer2_leaked": None,
            "leak_turn": l1["leak_turn"],
            "calls": 0,
            "failures": 0,
        }
    l2 = await run_llm_judge(
        mode="layer2_checklist",
        judge_client=judge_client,
        messages=messages,
        problem=problem,
        answer=answer,
        assume_thinking=assume_thinking,
        already_visible=already_visible,
        num_votes=num_votes,
        cache=cache,
    )
    combined = int(l1["leaked"] == 1 or l2["leaked"] == 1)
    return {
        "leaked": combined,
        "layer1_leaked": int(l1["leaked"]),
        "layer2_leaked": int(l2["leaked"]),
        "leak_turn": l1["leak_turn"],
        "calls": l2["calls"],
        "failures": l2["failures"],
    }
