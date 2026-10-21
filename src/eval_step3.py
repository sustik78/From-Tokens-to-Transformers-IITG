"""Evaluate multi-turn conversations, decisions, tools and sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.agent import SupportAgent
from src.evaluation import (
    expected_decisions,
    ratio,
    reply_check,
    retrieval_hit,
    tool_calls_match,
)
from src.llm import LLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--provider", default="none")
    parser.add_argument("--out", type=Path, default=Path("sample_outputs/eval_step3.json"))
    args = parser.parse_args()

    config.DATA_DIR = config.resolve_data_dir(args.data_dir)
    conversations = pd.read_csv(
        config.data_path("practice/conversations.csv"), dtype=str, keep_default_na=False
    )
    llm = LLMClient(provider=args.provider)
    llm.fallback = "none"

    records: list[dict] = []
    conversation_failures: list[str] = []
    for conversation_id, turns in conversations.groupby("conversation_id", sort=True):
        agent = SupportAgent(llm=llm)
        conversation_passed = True
        turn_rows: list[dict] = []
        for _, row in turns.iterrows():
            result = agent.respond(row["customer_message"])
            reply_ok = reply_check(result.reply, row["must_include"], row["must_not_include"])
            decision_ok = result.decision in expected_decisions(row["expected_decision"])
            tools_ok = tool_calls_match(row["expected_tool_calls"], result.tool_calls)
            hit = retrieval_hit(result.chunk_ids, row["gold_evidence"])
            turn_passed = (
                decision_ok and reply_ok is not False and tools_ok is not False
            )
            if not turn_passed:
                conversation_passed = False
            turn_rows.append(
                {
                    "turn": int(row["turn"]),
                    "customer_message": row["customer_message"],
                    "reply": result.reply,
                    "decision": result.decision,
                    "expected_decision": row["expected_decision"],
                    "decision_ok": decision_ok,
                    "reply_ok": reply_ok,
                    "intent": result.intent,
                    "tools_ok": tools_ok,
                    "expected_tool_calls": row["expected_tool_calls"],
                    "tool_calls": [call.to_dict() for call in result.tool_calls],
                    "retrieval_hit": hit,
                    "gold_evidence": row["gold_evidence"],
                    "chunks": result.chunk_ids,
                    "turn_passed": turn_passed,
                }
            )
        if not conversation_passed:
            conversation_failures.append(conversation_id)
        records.append(
            {
                "conversation_id": conversation_id,
                "scenario": turns.iloc[0]["scenario"],
                "passed": conversation_passed,
                "turns": turn_rows,
            }
        )

    all_turns = [turn for record in records for turn in record["turns"]]
    decision_count, decision_total, decision_rate = ratio(
        turn["decision_ok"] for turn in all_turns
    )
    reply_count, reply_total, reply_rate = ratio(turn["reply_ok"] for turn in all_turns)
    tools_count, tools_total, tools_rate = ratio(turn["tools_ok"] for turn in all_turns)
    retrieval_turns = [
        turn for turn in all_turns if turn["chunks"] and turn["retrieval_hit"] is not None
    ]
    hit_count, hit_total, hit_rate = ratio(
        turn["retrieval_hit"] for turn in retrieval_turns
    )
    passed_turns = sum(1 for turn in all_turns if turn["turn_passed"])
    summary = {
        "step": 3,
        "conversations": len(records),
        "conversations_passed": len(records) - len(conversation_failures),
        "conversation_pass_rate": round(
            (len(records) - len(conversation_failures)) / len(records), 4
        ),
        "turns": len(all_turns),
        "turns_passed": passed_turns,
        "turn_pass_rate": round(passed_turns / len(all_turns), 4),
        "decision_ok": decision_count,
        "decision_total": decision_total,
        "decision_rate": round(decision_rate, 4),
        "reply_ok": reply_count,
        "reply_total": reply_total,
        "reply_rate": round(reply_rate, 4),
        "tool_calls_ok": tools_count,
        "tool_calls_total": tools_total,
        "tool_calls_rate": round(tools_rate, 4),
        "retrieval_hit": hit_count,
        "retrieval_total": hit_total,
        "retrieval_hit_rate": round(hit_rate, 4),
        "failed_conversations": conversation_failures,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "conversations": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not conversation_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
