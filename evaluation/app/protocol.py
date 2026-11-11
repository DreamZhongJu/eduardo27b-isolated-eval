"""Offline-only adaptation of selected EduardoRL evaluation primitives.

Derived from e8c89b7293e65e632149571b16a1c2d4e33c77fe; see LICENSE.author.
Network clients here explicitly reject non-loopback addresses.
"""
from __future__ import annotations
import asyncio, logging, os, re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
import openai

logger = logging.getLogger("offline-eduardo")
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / 'official' / 'eduardo' / 'configs' / 'prompts'
def load_prompt(name):
    return (_PROMPTS_DIR / (name + '.txt')).read_text()
_TEACHER_SYSTEM_PROMPT = load_prompt('teacher_prompt')
_STUDENT_SYSTEM_PROMPT = load_prompt('student_prompt')
_STUDENT_FINAL_PROMPT = load_prompt('student_final_prompt')
_STUDENT_FINAL_TRANSFER_PROMPT = load_prompt('student_final_transfer_prompt')
_STUDENT_OPENING_TEMPLATE = 'This is the problem I am trying to understand: {problem}'
_SOLVE_PROMPT = """Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\nProblem: {problem}"""
_END_OF_CONVERSATION = '<end_of_conversation>'
_OMITTED_TEACHER_TURN = '(hidden)'
_THINK_BLOCK_RE = re.compile(r'<think>(.*?)</think>', flags=re.DOTALL)
_THINK_RE = re.compile(r'<think>.*?</think>', flags=re.DOTALL)
def get_async_openai_client(base, key_env):
    p = urlparse(base)
    if p.scheme != 'http' or p.hostname not in ('localhost', '127.0.0.1') or p.username or p.password:
        raise ValueError('student endpoint must be loopback')
    return openai.AsyncClient(api_key=os.environ[key_env], base_url=base, timeout=300, max_retries=0)
def get_concurrency_semaphore(label, limit):
    return asyncio.Semaphore(limit)
def make_teacher(args):
    raise RuntimeError('must be installed by the local dialogue adapter')
async def _student_turn(*args):
    raise RuntimeError('must be installed by the local dialogue adapter')
def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last ``\\boxed{…}`` substring (including the braces).

    Handles arbitrary nesting depth via brace counting; ``None`` if not found.
    """
    if not string:
        return None
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    depth = 0
    right_idx: Optional[int] = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right_idx = i
                break
        i += 1
    return string[idx : right_idx + 1] if right_idx is not None else None

def _remove_boxed(s: str) -> str:
    left = r"\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return ""

def strip_teacher_controls(text: str, assume_thinking: bool = True) -> str:
    """Remove ``<end_of_conversation>`` and private thinking from a teacher message.

    The chat template pre-fills ``<think>``, so content usually arrives as
    ``REASONING</think>REPLY``. Text before an orphan ``</think>`` or after an
    orphan ``<think>`` is dropped. With ``assume_thinking``, content without any
    think tag means thinking hit the token cap, so ``""`` is returned.
    """
    if not text:
        return text
    if assume_thinking and "</think>" not in text and "<think>" not in text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    text = text.replace("<end_of_conversation>", "")
    return text.strip()

def split_thinking(text: str) -> tuple[str, str]:
    """Split a raw teacher message into ``(private reasoning, visible reply)``.

    Unlike the training version, tag-free content is treated as the visible
    reply (non-thinking models, API teachers). Handled forms:
      - balanced ``<think>…</think>`` blocks;
      - the Qwen3-style prefill ``REASONING</think>VISIBLE`` (orphan close,
        the opening tag lives in the prompt);
      - reasoning truncated by the per-turn token cap (orphan open → nothing
        after it is visible).
    ``<end_of_conversation>`` is dropped from the visible half.
    """
    if not text:
        return "", ""
    reasoning = [m.group(1) for m in _THINK_BLOCK_RE.finditer(text)]
    visible = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in visible:
        head, visible = visible.split("</think>", 1)
        reasoning.append(head.split("<think>", 1)[-1])
    if "<think>" in visible:
        visible, tail = visible.split("<think>", 1)
        reasoning.append(tail)
    return (
        "\n".join(p.strip() for p in reasoning if p.strip()),
        visible.replace(_END_OF_CONVERSATION, "").strip(),
    )

def thinking_text(text: str) -> str:
    """The private-reasoning half of a teacher message. Empty when the model
    did not think (or the provider dropped the reasoning)."""
    return split_thinking(text)[0]

def _eval_strip_teacher_controls(text):
    return split_thinking(text)[1]

class Conversation:
    """One teacher↔student rollout. ``messages`` uses neutral roles
    (``teacher`` / ``student``); perspective flipping + chat templating happen
    at generation time.
    """

    def __init__(self, problem: str, answer: str):
        self.problem = problem
        self.answer = answer
        self.messages: list[dict[str, str]] = [
            {
                "role": "student",
                "content": _STUDENT_OPENING_TEMPLATE.format(problem=problem),
            }
        ]
        self.done = False
        self.end_reason: str | None = None
        # One entry per teacher turn; source is "tokenizer", "api" or "estimate".
        self.thinking_tokens: list[int] = []
        self.thinking_tokens_source: str | None = None

    @property
    def num_teacher_turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "teacher")

    def finish(self, reason: str) -> None:
        self.done = True
        self.end_reason = reason

    def teacher_view(self) -> list[dict[str, str]]:
        """Teacher = assistant, student = user, plus the teacher system prompt."""
        out = [
            {
                "role": "system",
                "content": _TEACHER_SYSTEM_PROMPT.format(problem=self.problem),
            }
        ]
        for m in self.messages:
            role = "assistant" if m["role"] == "teacher" else "user"
            out.append({"role": role, "content": m["content"]})
        return out

    def student_view(self) -> list[dict[str, str]]:
        """Student = assistant, teacher = user (control tokens stripped)."""
        out = [
            {
                "role": "system",
                "content": _STUDENT_SYSTEM_PROMPT.format(problem=self.problem),
            }
        ]
        for m in self.messages:
            if m["role"] == "teacher":
                out.append({"role": "user", "content": _eval_strip_teacher_controls(m["content"])})
            else:
                out.append({"role": "assistant", "content": m["content"]})
        return out

    def scoring_messages(self) -> list[dict[str, str]]:
        """Transcript for RM scoring, teacher thinking/EoC tokens stripped
        (same as ``_hide_thinking`` in PedagogicalRL (TutorRL))."""
        return [
            {"role": m["role"], "content": _eval_strip_teacher_controls(m["content"])}
            for m in self.messages
        ]

    def to_record(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "answer": self.answer,
            "messages": self.scoring_messages(),
            "raw_messages": self.messages,
            "num_teacher_turns": self.num_teacher_turns,
            "thinking_tokens": self.thinking_tokens,
            "thinking_tokens_source": self.thinking_tokens_source,
            "end_reason": self.end_reason,
        }

def _has_think_prefill(teacher: Any, conv: "ValConversation", args) -> bool | None:
    """Whether the chat template pre-fills ``<think>`` into the prompt.

    Returns ``None`` for an API teacher (no local tokenizer).
    """
    tokenizer = getattr(teacher, "tokenizer", None)
    if tokenizer is None:
        return None
    rendered = tokenizer.apply_chat_template(
        conv.teacher_view(),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    # The prefill sits right after the assistant header at the end of the prompt.
    return "<think>" in rendered[-200:]

def resolve_assume_thinking(teacher: Any, conv: "ValConversation", args, texts: list[str]) -> bool:
    """Decide the ``assume_thinking`` rule for this run from the chat template.

    Response markers are only used to warn about contradictions, since models
    that think on some turns and not others produce mixed batches.
    """
    if args.assume_thinking in ("yes", "no"):
        assume = args.assume_thinking == "yes"
        logger.info("Thinking mode FORCED: assume_thinking=%s (--assume-thinking %s)",
                    assume, args.assume_thinking)
        return assume

    prefill = _has_think_prefill(teacher, conv, args)
    if prefill is None:
        logger.info(
            "Thinking mode AUTO: API teacher (no local template) -> "
            "assume_thinking=False; tag-free turns are the visible reply."
        )
        return False

    logger.info(
        "Thinking mode AUTO: chat template %s <think> prefill -> "
        "assume_thinking=%s (%s)",
        "HAS a" if prefill else "has NO", prefill,
        "tag-free turns are unterminated reasoning and get blanked"
        if prefill else
        "tag-free turns are kept as the visible reply; only <think> without "
        "</think> is blanked",
    )

    n_marked = sum(1 for t in texts if t and ("</think>" in t or "<think>" in t))
    if prefill and n_marked == 0 and texts:
        logger.warning(
            "Template prefills <think>, yet NONE of the %d first-batch turns "
            "closed it. Either every turn overran --max-tokens-per-turn=%d or "
            "the prefill detection is wrong; watch blanked_teacher_turn_rate "
            "and rerun with --assume-thinking no if it is high.",
            len(texts), args.max_tokens_per_turn,
        )
    elif not prefill and n_marked and n_marked < len(texts):
        logger.info(
            "%d/%d first-batch turns think explicitly; the rest are replies "
            "written without thinking and are kept (this mixed case is exactly "
            "what the old response-counting heuristic got wrong).",
            n_marked, len(texts),
        )
    return prefill

class ValConversation(Conversation):
    """One rollout over a ``test.parquet`` row, using the row's own teacher
    system prompt and the training student prompt.
    """

    def __init__(self, row: dict[str, Any], row_index: int = -1, sample_index: int = 0):
        extra = row["extra_info"]
        # Row index is the clustering unit for standard errors (compute_summary).
        self.row_index = row_index
        self.sample_index = sample_index
        super().__init__(extra["problem"], str(extra["answer"]))
        # Teacher system message and pre-seeded student opener, verbatim from the parquet.
        seed = list(row["prompt"])
        self.system_prompt = next(
            (m["content"] for m in seed if m["role"] == "system"),
            _TEACHER_SYSTEM_PROMPT.format(problem=self.problem),
        )
        opener = next((m["content"] for m in seed if m["role"] == "user"), None)
        if opener is None:
            raise ValueError("dataset row has no pre-seeded student opener")
        self.messages = [{"role": "student", "content": opener}]

        self.transfer_problem = str(extra.get("transfer_problem") or "")
        self.transfer_answer = str(extra.get("transfer_answer") or "")
        # Δ baselines, measured with the same student model at dataset creation.
        self.pre_solve_rate = float(extra.get("pre_solve_rate") or 0.0)
        self.pre_solve_rate_transfer = float(
            extra.get("pre_solve_rate_transfer_problem")
            if extra.get("pre_solve_rate_transfer_problem") is not None
            else extra.get("pre_solve_rate") or 0.0
        )
        self.eoc_used = False
        self.metrics: dict[str, Any] = {}
        # Set by resolve_assume_thinking after the first teacher batch.
        self.assume_thinking = True

    def teacher_view(self) -> list[dict[str, str]]:
        out = [{"role": "system", "content": self.system_prompt}]
        for m in self.messages:
            out.append(
                {
                    "role": "assistant" if m["role"] == "teacher" else "user",
                    "content": m["content"],
                }
            )
        return out

    def student_view(self) -> list[dict[str, str]]:
        out = [
            {
                "role": "system",
                "content": _STUDENT_SYSTEM_PROMPT.format(problem=self.problem),
            }
        ]
        for m in self.messages:
            if m["role"] == "teacher":
                out.append(
                    {
                        "role": "user",
                        "content": strip_teacher_controls(
                            m["content"], assume_thinking=self.assume_thinking
                        ),
                    }
                )
            else:
                out.append({"role": "assistant", "content": m["content"]})
        return out

    def reward_view(self) -> list[dict[str, str]]:
        """The transcript shape ``eduardo_reward_func`` receives: teacher =
        assistant, student = user, teacher content untouched (thinking intact).
        """
        return [
            {
                "role": "assistant" if m["role"] == "teacher" else "user",
                "content": m["content"],
            }
            for m in self.messages
        ]

    def judge_view(self) -> list[dict[str, str]]:
        """Student-visible transcript the judges score — teacher ``<think>``
        blocks and control tokens stripped (reward_function's ``judge_msgs``).
        """
        return [
            {
                **m,
                "content": strip_teacher_controls(
                    m["content"], assume_thinking=self.assume_thinking
                ),
            }
            if m["role"] == "assistant"
            else m
            for m in self.reward_view()
        ]

    def blanked_teacher_turns(self) -> int:
        """Teacher turns that strip to nothing in the student's/judge's view.

        Near 100% indicates a wrong ``assume_thinking`` rule.
        """
        return sum(
            1
            for m in self.messages
            if m["role"] == "teacher"
            and not strip_teacher_controls(
                m["content"], assume_thinking=self.assume_thinking
            ).strip()
        )

    def to_record(self) -> dict[str, Any]:
        rec = super().to_record()
        rec.update(
            {
                "transfer_problem": self.transfer_problem,
                "transfer_answer": self.transfer_answer,
                "pre_solve_rate": self.pre_solve_rate,
                "pre_solve_rate_transfer": self.pre_solve_rate_transfer,
                "system_prompt": self.system_prompt,
                "row_index": self.row_index,
                "sample_index": self.sample_index,
                "eoc_used": self.eoc_used,
                "assume_thinking": self.assume_thinking,
                "num_blanked_teacher_turns": self.blanked_teacher_turns(),
                "blanked_teacher_turn_rate": (
                    self.blanked_teacher_turns() / self.num_teacher_turns
                    if self.num_teacher_turns else 0.0
                ),
                **self.metrics,
            }
        )
        return rec

async def transfer_rollout(args, rows: list[dict[str, Any]]) -> list[ValConversation]:
    conversations = [
        ValConversation(row, row_index=i, sample_index=k)
        for i, row in enumerate(rows)
        for k in range(args.num_samples_per_problem)
    ]
    logger.info(
        "%d problems x %d samples = %d conversations",
        len(rows), args.num_samples_per_problem, len(conversations),
    )

    args.engine_seed = args.seed
    teacher = make_teacher(args)
    client = get_async_openai_client(args.student_api_base, args.student_api_key_env)
    # Same "student" concurrency budget as in training.
    semaphore = get_concurrency_semaphore("student", args.student_concurrency)

    round_idx = 0
    while True:
        active = [c for c in conversations if not c.done]
        if not active:
            break
        round_idx += 1
        logger.info("Round %d: teacher turn for %d conversations", round_idx, len(active))

        texts = await teacher.generate(active)
        if round_idx == 1:
            # Must be resolved before student_view() is first used.
            assume = resolve_assume_thinking(teacher, active[0], args, texts)
            for conv in conversations:
                conv.assume_thinking = assume
        for i, (conv, text) in enumerate(zip(active, texts)):
            conv.messages.append({"role": "teacher", "content": text})
            tokens, source = teacher.thinking_tokens(i, text)
            conv.thinking_tokens.append(tokens)
            conv.thinking_tokens_source = source
            if not text:
                conv.finish("teacher_failed")
                continue
            # As in training, EoC is ignored until min_teacher_turns_before_eoc
            # turns precede the tagged one.
            if (
                _END_OF_CONVERSATION in text
                and conv.num_teacher_turns - 1 >= args.min_teacher_turns_before_eoc
            ):
                conv.eoc_used = True
                conv.finish("end_of_conversation")
            elif conv.num_teacher_turns >= args.max_teacher_turns:
                conv.finish("max_teacher_turns")
            elif teacher.count_tokens(conv) > args.max_conversation_tokens:
                conv.finish("token_budget")

        active = [c for c in conversations if not c.done]
        if not active:
            break
        logger.info("Round %d: student turn for %d conversations", round_idx, len(active))
        await asyncio.gather(
            *(_student_turn(conv, client, args, semaphore) for conv in active)
        )
        for conv in active:
            if not conv.done and teacher.count_tokens(conv) > args.max_conversation_tokens:
                conv.finish("token_budget")

    teacher.close()
    return conversations

def _render_final_prompt(transfer_problem: str | None) -> str:
    """Final user turn: plain "now solve it" or the near-transfer variant.

    Uses ``str.replace`` since the prompt contains a literal ``\\boxed{...}``.
    """
    if not transfer_problem:
        return _STUDENT_FINAL_PROMPT
    return _STUDENT_FINAL_TRANSFER_PROMPT.replace(
        "{transfer_problem}", transfer_problem
    )

def _build_final_attempt_messages(
    messages: list[dict],
    problem: str,
    transfer_problem: str | None = None,
    omit_teacher_turns: bool = False,
) -> list[dict]:
    """Build the student-view chat history for a final-attempt call.

    Flips roles (teacher -> user, student -> assistant), strips teacher control
    tokens and thinking (or hides teacher turns if ``omit_teacher_turns``), and
    appends the final-attempt prompt. ``messages`` is not mutated.
    """
    out: list[dict] = [
        {"role": "system", "content": _STUDENT_SYSTEM_PROMPT.format(problem=problem)}
    ]
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            continue
        if role == "assistant":
            teacher_text = (
                _OMITTED_TEACHER_TURN
                if omit_teacher_turns
                else strip_teacher_controls(content)
            )
            out.append({"role": "user", "content": teacher_text})
        elif role == "user":
            out.append({"role": "assistant", "content": content})
    out.append({"role": "user", "content": _render_final_prompt(transfer_problem)})
    return out

def _row(
    problem: str,
    answer: str,
    solve_rate: float,
    transfer_problem: str,
    transfer_answer: str,
    transfer_solve_rate: float,
    split: str,
) -> dict:
    """One dataset row (schema consumed by veRL's RLHFDataset + ToolAgentLoop).

    ``ground_truth`` is the original answer (used by the leakage judge); the
    transfer answer is used only to grade the student's final attempts.
    """
    prompt = [
        {
            "role": "system",
            "content": _TEACHER_SYSTEM_PROMPT.format(problem=problem),
        },
        {
            "role": "user",
            "content": _STUDENT_OPENING_TEMPLATE.format(problem=problem),
        },
    ]
    return {
        "data_source": "bigmath",
        "prompt": prompt,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "problem": problem,
            "answer": answer,
            # Lets the reward function run the periodic-eval judge on val rows only.
            "split": split,
            # Original problem's rate; not the Δ_solve baseline.
            "solve_rate": float(solve_rate),
            "pre_solve_rate": float(solve_rate),
            "transfer_problem": transfer_problem,
            "transfer_answer": transfer_answer,
            # Δ_solve baseline: untutored rate on the transfer problem.
            "pre_solve_rate_transfer_problem": float(transfer_solve_rate),
            # The live dialogue covers only the original problem; the variant stays unseen.
            "interaction_kwargs": {
                "name": "eduardo",
                "problem": problem,
                "answer": answer,
                "ground_truth": answer,
            },
        },
        # EduardoAgentLoop exposes the transcript to the reward function.
        "agent_name": "eduardo_agent",
    }
