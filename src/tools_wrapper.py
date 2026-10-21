"""Thin wrapper around the dataset's mocked support tools.

The dataset's `tools/tools.py` owns the verification gate.  This module only
imports it from DATA_DIR, records an audit trail, and redacts verification
values from anything that may be written to disk.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from . import config


@dataclass
class ToolCall:
    name: str
    public_args: dict[str, Any] = field(default_factory=dict)
    private_args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    ok: bool = False

    def to_dict(self, include_private: bool = False) -> dict:
        payload = {
            "name": self.name,
            "args": self.public_args,
            "result": self.result,
            "ok": self.ok,
        }
        if include_private:
            payload["private_args"] = self.private_args
        return payload

    def signature(self) -> str:
        if not self.private_args:
            return f"{self.name}()"
        parts = ", ".join(
            f"{key}={value}" for key, value in self.private_args.items()
        )
        return f"{self.name}({parts})"


@lru_cache(maxsize=1)
def dataset_tools():
    data_dir = str(config.DATA_DIR)
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    import tools as dataset_tools_module  # type: ignore

    return dataset_tools_module


def _sanitize_lookup(result: dict) -> dict:
    if result.get("verified") is True:
        return {
            "found": True,
            "verified": True,
            "order_id": result.get("order_id"),
            "status": result.get("status"),
            "promised_date": result.get("promised_date"),
            "delivered_on": result.get("delivered_on"),
            "shipped_on": result.get("shipped_on"),
            "payment_method": result.get("payment_method"),
            "delivery_type": result.get("delivery_type"),
            "is_plus": result.get("is_plus"),
            "customer_first_name": result.get("customer_first_name"),
        }
    if result.get("verification_required"):
        return {
            "verification_required": True,
            "verified": result.get("verified", False),
            "found": result.get("found", False),
            "message": result.get("message"),
        }
    return {"verified": False, "found": False, "message": result.get("message")}


def lookup_order(
    order_id: str,
    verification_value: str | None = None,
) -> tuple[dict, ToolCall]:
    args = {"order_id": str(order_id)}
    private_args = {"order_id": str(order_id)}
    if verification_value not in (None, ""):
        private_args["verification_value"] = str(verification_value)
    try:
        result = dataset_tools().lookup_order(
            str(order_id),
            verification_value=(
                None if verification_value in (None, "") else str(verification_value)
            ),
        )
        ok = bool(result.get("verified") or result.get("verification_required"))
        call = ToolCall(
            name="lookup_order",
            public_args=args,
            private_args=private_args,
            result=_sanitize_lookup(result),
            ok=ok,
        )
        return result, call
    except Exception as exc:  # noqa: BLE001
        call = ToolCall(
            name="lookup_order",
            public_args=args,
            private_args=private_args,
            result={"error": str(exc)},
            ok=False,
        )
        return {"verified": False, "found": False, "error": str(exc)}, call


def create_ticket(
    summary: str,
    order_id: str | None = None,
    reason: str | None = None,
    intent: str | None = None,
) -> tuple[dict, ToolCall]:
    private_args = {
        "summary": summary,
        "order_id": order_id,
        "reason": reason,
        "intent": intent,
    }
    public_args = {
        "summary": summary,
        "order_id": order_id,
        "reason": reason,
        "intent": intent,
    }
    try:
        result = dataset_tools().create_ticket(
            summary=summary,
            order_id=order_id,
            reason=reason,
            intent=intent,
        )
        call = ToolCall(
            name="create_ticket",
            public_args=public_args,
            private_args=private_args,
            result={
                "created": bool(result.get("created")),
                "ticket_id": result.get("ticket_id"),
                "error": result.get("error"),
            },
            ok=bool(result.get("created")),
        )
        return result, call
    except Exception as exc:  # noqa: BLE001
        call = ToolCall(
            name="create_ticket",
            public_args=public_args,
            private_args=private_args,
            result={"created": False, "error": str(exc)},
            ok=False,
        )
        return {"created": False, "error": str(exc)}, call
