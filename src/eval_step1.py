"""Prompt-only baseline: ask an LLM without any retrieved documents."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.evaluation import expected_decisions, infer_decision, ratio, reply_check
from src.llm import LLMClient

SYSTEM = (
    "You are a helpful customer support assistant for an online store. "
    "You do not have access to the company's documents, order records or tools. "
    "Answer from general knowledge only. Give your best concise answer; do not pad. "
    "Follow the language tag on every question: en = English, hi = Hindi, "
    "hi-en = Hinglish."
)

INSTRUCTIONS = (
    "Answer every question in at most two sentences. Return exactly one answer "
    "per question using this format and no other text:\n"
    "1. <answer>\n2. <answer>\n...\n\n"
)


def parse_answers(text: str, count: int) -> list[str]:
    answers = [""] * count
    current: int | None = None
    for raw_line in (text or "").splitlines():
        line = re.sub(r"^[\s#*_>-]+", "", raw_line).strip()
        match = re.match(r"^(?:Q)?(\d+)\s*(?:[.):-]\s*|\s+)(.*)$", line, re.IGNORECASE)
        if match and 1 <= int(match.group(1)) <= count:
            current = int(match.group(1)) - 1
            answers[current] = match.group(2).strip()
        elif current is not None and line:
            answers[current] = f"{answers[current]} {line}".strip()
    return answers


def ask_batch(llm: LLMClient, questions: list[str]) -> list[str]:
    block = "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    )
    response = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTIONS + block},
        ],
        max_tokens=max(600, len(questions) * 90),
        temperature=0.2,
        allow_fallback=False,
    )
    return parse_answers(response.content, len(questions))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--provider", default="llm7")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=13)
    parser.add_argument("--delay", type=float, default=55.0)
    parser.add_argument("--out", type=Path, default=Path("sample_outputs/eval_step1.json"))
    args = parser.parse_args()

    config.DATA_DIR = config.resolve_data_dir(args.data_dir)
    questions = pd.read_csv(
        config.data_path("practice/questions.csv"), dtype=str, keep_default_na=False
    )
    llm = LLMClient(provider=args.provider, model=args.model)
    llm.fallback = "none"

    prompts = [
        f"[reply language: {row['reply_language'] or 'en'}] {row['asr_text']}"
        for _, row in questions.iterrows()
    ]
    answers: list[str] = []
    errors: list[str] = []
    size = max(1, args.batch_size)
    for start in range(0, len(prompts), size):
        batch = prompts[start : start + size]
        try:
            answers.extend(ask_batch(llm, batch))
        except Exception as exc:  # noqa: BLE001 - provider errors are varied
            errors.append(f"batch {start // size + 1}: {exc}")
            answers.extend([""] * len(batch))
        if start + size < len(prompts):
            time.sleep(max(0.0, args.delay))

    rows: list[dict] = []
    for (_, row), reply in zip(questions.iterrows(), answers):
        decision = infer_decision(reply)
        decisions_ok = decision in expected_decisions(row["acceptable_decisions"])
        rows.append(
            {
                "id": row["id"],
                "question": row["asr_text"],
                "reply": reply,
                "decision": decision,
                "expected_decision": row["acceptable_decisions"],
                "decision_ok": decisions_ok,
                "must_include": row["must_include"],
                "reply_ok": reply_check(
                    reply, row["must_include"], row["must_not_include"]
                ),
                "retrieval_hit": None,
                "gold_evidence": row["gold_evidence"],
                "retrieved": [],
            }
        )

    reply_count, reply_total, reply_rate = ratio(row["reply_ok"] for row in rows)
    dec_count, dec_total, dec_rate = ratio(row["decision_ok"] for row in rows)
    joint_count = sum(
        1 for row in rows if row["reply_ok"] is not False and row["decision_ok"]
    )
    summary = {
        "step": 1,
        "questions": len(rows),
        "provider": args.provider,
        "model": args.model,
        "reply_ok": reply_count,
        "reply_total": reply_total,
        "reply_rate": round(reply_rate, 4),
        "decision_ok": dec_count,
        "decision_total": dec_total,
        "decision_rate": round(dec_rate, 4),
        "joint_ok": joint_count,
        "joint_rate": round(joint_count / len(rows), 4),
        "errors": errors,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
