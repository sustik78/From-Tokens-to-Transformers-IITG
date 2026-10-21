"""Evaluate the grounded single-turn answerer on practice questions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.answering import GroundedAnswerer
from src.evaluation import expected_decisions, ratio, reply_check, retrieval_hit
from src.llm import LLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("sample_outputs/eval_step2.json"))
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    config.DATA_DIR = config.resolve_data_dir(args.data_dir)
    questions = pd.read_csv(
        config.data_path("practice/questions.csv"), dtype=str, keep_default_na=False
    )
    llm = LLMClient(provider="none")
    llm.fallback = "none"
    answerer = GroundedAnswerer(llm)

    rows: list[dict] = []
    for _, row in questions.iterrows():
        answer = answerer.answer(row["asr_text"], language=row["reply_language"] or None)
        reply_ok = reply_check(answer.reply, row["must_include"], row["must_not_include"])
        decisions_ok = answer.decision in expected_decisions(row["acceptable_decisions"])
        hit = retrieval_hit([item.chunk.chunk_id for item in answer.results], row["gold_evidence"])
        rows.append(
            {
                "id": row["id"],
                "question": row["asr_text"],
                "reply": answer.reply,
                "decision": answer.decision,
                "expected_decision": row["acceptable_decisions"],
                "decision_ok": decisions_ok,
                "intent": answer.intent,
                "must_include": row["must_include"],
                "reply_ok": reply_ok,
                "retrieval_hit": hit,
                "gold_evidence": row["gold_evidence"],
                "retrieved": [item.to_dict() for item in answer.results],
            }
        )

    reply_count, reply_total, reply_rate = ratio(row["reply_ok"] for row in rows)
    hit_count, hit_total, hit_rate = ratio(row["retrieval_hit"] for row in rows)
    dec_count, dec_total, dec_rate = ratio(row["decision_ok"] for row in rows)
    joint_count = sum(
        1
        for row in rows
        if row["reply_ok"] is not False and row["decision_ok"] and row["retrieval_hit"] is not False
    )
    summary = {
        "step": 2,
        "questions": len(rows),
        "top_k": args.top_k,
        "reply_ok": reply_count,
        "reply_total": reply_total,
        "reply_rate": round(reply_rate, 4),
        "retrieval_hit": hit_count,
        "retrieval_total": hit_total,
        "retrieval_hit_rate": round(hit_rate, 4),
        "decision_ok": dec_count,
        "decision_total": dec_total,
        "decision_rate": round(dec_rate, 4),
        "joint_ok": joint_count,
        "joint_rate": round(joint_count / len(rows), 4),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["joint_rate"] >= 0.7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
