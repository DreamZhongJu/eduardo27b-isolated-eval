#!/usr/bin/env python3
"""Single-call Agnes API ping through the unmodified author judge client path.

Uses one completely synthetic dialogue and prints ONLY:
- HTTP status
- returned model name
- whether the response parses as the author's expected judge JSON

Never prints API keys, prompt text, or raw response reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _locate_author_repo() -> Path:
    candidates = []
    if os.environ.get("EDUARDO_REPO"):
        candidates.append(Path(os.environ["EDUARDO_REPO"]))
    candidates.extend([
        Path("/workspace/A_group_repro_20261117/source/EduardoRL"),
        Path(__file__).resolve().parents[2] / "EduardoRL_A组源码_20261117",
    ])
    for cand in candidates:
        if (cand / "eval" / "run_middleturn_eval.py").is_file():
            return cand
    raise SystemExit("ERROR: EduardoRL upstream repo not found; set EDUARDO_REPO")


def _strict_judge_json_ok(text: str) -> tuple[bool, str]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        return False, "no_json_object"
    try:
        data = json.loads(cleaned[start:end])
    except Exception as exc:
        return False, f"json_error:{type(exc).__name__}"
    if not isinstance(data, dict):
        return False, "not_dict"
    decision = str(data.get("decision", "")).strip().upper()
    if decision not in ("OK", "REJECT"):
        return False, "invalid_decision"
    if "reasoning" not in data:
        return False, "missing_reasoning"
    return True, decision


async def _run_ping(args: argparse.Namespace) -> int:
    repo = _locate_author_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import openai
    from eval.run_middleturn_eval import make_judge_client
    from eduardo.metrics import judge_quality

    if not os.environ.get(args.api_key_env):
        print(f"ERROR: environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2

    judge_args = SimpleNamespace(
        gemini=True,
        gemini_api_base=args.api_base,
        gemini_model=args.model,
        gemini_api_key_env=args.api_key_env,
        gemini_reasoning_effort=args.reasoning_effort,
        judge_model="",
        judge_api_base="",
    )
    client, model, extra_body = make_judge_client(judge_args)

    prompt = judge_quality._JUDGE_PROMPTS["no_leak"].format(
        problem="Solve 2x = 6.",
        answer="3",
        conversation=(
            "- Student: How do I solve 2x = 6?\n"
            "- Teacher: What operation undoes multiplying x by 2?"
        ),
    )

    try:
        raw_resp = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=16384,
            extra_body=extra_body,
        )
        http_status = raw_resp.status_code
        parsed_resp = raw_resp.parse()
        returned_model = getattr(parsed_resp, "model", None)
        content = parsed_resp.choices[0].message.content or ""
    except openai.APIStatusError as exc:
        print(f"http_status={exc.status_code}")
        print(f"error_type={type(exc).__name__}")
        print("json_parse_ok=False")
        return 1
    except Exception as exc:
        print("http_status=ERROR")
        print(f"error_type={type(exc).__name__}")
        print("json_parse_ok=False")
        return 1
    finally:
        await client.close()

    strict_ok, strict_detail = _strict_judge_json_ok(content)
    author_parsed = judge_quality._parse_judge_response(content)
    author_fallback_used = author_parsed["reasoning"] in ("<unparsed JSON>", "Parse failure")

    print(f"http_status={http_status}")
    print(f"returned_model={returned_model}")
    print(f"strict_json_ok={strict_ok}")
    print(f"strict_detail={strict_detail}")
    print(f"author_decision={author_parsed['decision']}")
    print(f"author_fallback_used={author_fallback_used}")
    return 0 if (http_status == 200 and strict_ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Ping Agnes once via author judge client path.")
    ap.add_argument("--api-base", default="https://api.agnes-ai.cn/v1")
    ap.add_argument("--model", default="agnes-3.0-flash")
    ap.add_argument("--api-key-env", default="AGNES_API_KEY")
    ap.add_argument("--reasoning-effort", default="high", choices=("low", "medium", "high"))
    args = ap.parse_args()
    return asyncio.run(_run_ping(args))


if __name__ == "__main__":
    raise SystemExit(main())
