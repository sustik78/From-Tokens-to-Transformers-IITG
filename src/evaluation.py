"""Small scoring helpers shared by the evaluation scripts."""
from __future__ import annotations

import re
from collections.abc import Iterable


def split_options(raw: str) -> list[str]:
    parts: list[str] = []
    for group in re.split(r"\s*/\s*|\s*;\s*", raw or ""):
        value = group.strip()
        if value:
            parts.append(value)
    return parts


def expected_decisions(raw: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"[/;]", raw or "")
        if value.strip()
    ]


def reply_check(reply: str, must_include: str, must_not_include: str) -> bool | None:
    lower = (reply or "").lower()
    required = split_options(must_include)
    forbidden = split_options(must_not_include)
    if any(value.lower() in lower for value in forbidden):
        return False
    if not required:
        return None
    return any(value.lower() in lower for value in required)


def infer_decision(reply: str) -> str:
    lower = (reply or "").lower()
    if any(
        marker in lower
        for marker in (
            "order id",
            "order number",
            "last four",
            "last 4",
            "pincode",
            "pin code",
        )
    ) and any(marker in lower for marker in ("share", "provide", "tell", "bata", "कृपया", "कripya")):
        return "clarify"
    if any(
        marker in lower
        for marker in (
            "connect you",
            "human",
            "team will",
            "escalat",
            "ticket",
            "hand over",
            "handover",
        )
    ):
        return "escalate"
    if any(
        marker in lower
        for marker in (
            "i can't",
            "i cannot",
            "unable to",
            "not allowed",
            "won't be able",
            "cannot share",
        )
    ):
        return "refuse"
    return "answer"


def gold_ids(raw: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"\s*;\s*", raw or "")
        if value.strip()
    ]


def retrieval_hit(chunk_ids: Iterable[str], gold: str) -> bool | None:
    expected = gold_ids(gold)
    if not expected:
        return None
    found = set(chunk_ids)
    return any(value in found for value in expected)


def parse_tool_expectation(raw: str) -> list[dict]:
    calls: list[dict] = []
    for part in split_options(raw):
        match = re.match(r"([A-Za-z_][\w]*)\((.*)\)$", part)
        if not match:
            continue
        args: dict[str, str] = {}
        for item in re.split(r",\s*(?=[A-Za-z_][\w]*=)", match.group(2)):
            if "=" in item:
                key, value = item.split("=", 1)
                args[key.strip()] = value.strip()
        calls.append({"name": match.group(1), "args": args})
    return calls


def tool_calls_match(expected: str, calls: list) -> bool | None:
    wanted = parse_tool_expectation(expected)
    if not wanted:
        return None
    if len(wanted) != len(calls):
        return False
    for expected_call, call in zip(wanted, calls):
        if expected_call["name"] != call.name:
            return False
        for key, value in expected_call["args"].items():
            actual = call.public_args.get(key)
            if actual is None:
                actual = call.private_args.get(key)
            if str(actual) != str(value):
                return False
    return True


def ratio(values: Iterable[bool | None]) -> tuple[int, int, float]:
    checked = [value for value in values if value is not None]
    passed = sum(1 for value in checked if value)
    return passed, len(checked), (passed / len(checked) if checked else 0.0)
