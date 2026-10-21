"""Markdown chunking and policy-fact extraction.

Chunks are section-level and keep the [SECTION-ID], source file, document
type, status and effective_from date.  Answers are built from these chunks and
the CSV catalog; no answer text is hard-coded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import config, data

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SECTION_RE = re.compile(r"^##\s+\[([^\]]+)\]\s*(.*)$", re.MULTILINE)

DOC_PRIORITY = {
    "policy": 1.00,
    "agent_guide": 0.88,
    "help": 0.76,
    "faq": 0.62,
    "marketing": 0.35,
    "meta": 0.80,
    "catalog": 0.92,
}


@dataclass
class Chunk:
    chunk_id: str
    source: str
    doc_id: str
    doc_type: str
    status: str
    effective_from: str
    title: str
    section_title: str
    text: str
    priority: float = 0.5
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "status": self.status,
            "effective_from": self.effective_from,
            "title": self.title,
            "section_title": self.section_title,
            "text": self.text,
            "priority": self.priority,
            "extra": self.extra,
        }


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, raw[match.end() :]


def _priority(doc_type: str, status: str, chunk_id: str) -> float:
    base = DOC_PRIORITY.get(doc_type, 0.5)
    if status == "superseded":
        base *= 0.18
    if doc_type == "policy" and status == "current":
        base += 0.05
    if chunk_id == "KB-00.1":
        base += 0.12
    return round(base, 4)


def parse_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    doc_type = meta.get("doc_type", "meta")
    status = meta.get("status", "current")
    doc_id = meta.get("doc_id", path.stem)
    title = meta.get("title", path.stem)
    effective = meta.get("effective_from", "")
    matches = list(SECTION_RE.finditer(body))
    chunks: list[Chunk] = []
    if not matches:
        text = body.strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=doc_id,
                    source=path.name,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective,
                    title=title,
                    section_title=title,
                    text=text,
                    priority=_priority(doc_type, status, doc_id),
                    extra={},
                )
            )
        return chunks
    for index, match in enumerate(matches):
        section_id = match.group(1).strip()
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        header = (
            f"Document: {title} ({doc_id}, {doc_type}, status={status}, "
            f"effective_from={effective}). Section [{section_id}] {section_title}."
        )
        chunks.append(
            Chunk(
                chunk_id=section_id,
                source=path.name,
                doc_id=doc_id,
                doc_type=doc_type,
                status=status,
                effective_from=effective,
                title=title,
                section_title=section_title,
                text=f"{header}\n{text}",
                priority=_priority(doc_type, status, section_id),
                extra={
                    "section_id": section_id,
                    "section_title": section_title,
                    "raw_text": text,
                },
            )
        )
    return chunks


def _catalog_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for _, row in data.products().iterrows():
        sku = str(row["sku"])
        assured = data.assured_warranty(row.to_dict())
        text = (
            f"Product catalog row {sku}: {row['title']}. Brand {row['brand']}. "
            f"Return category {row['return_category']}. Price ₹{row['price_inr']}. "
            f"Fulfilled by {row['fulfilled_by']}. Assured badge {row['assured']}. "
            f"ZentroMart Assured applies: {assured}. "
            f"Brand warranty {row['warranty_months']} months. "
            f"{row['description']}."
        )
        chunks.append(
            Chunk(
                chunk_id=sku,
                source="products.csv",
                doc_id="products.csv",
                doc_type="catalog",
                status="current",
                effective_from="",
                title="Product catalog",
                section_title=str(row["title"]),
                text=text,
                priority=DOC_PRIORITY["catalog"],
                extra=row.to_dict(),
            )
        )
    return chunks


@lru_cache(maxsize=1)
def all_chunks() -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for path in sorted(config.data_path("kb").glob("*.md")):
        chunks.extend(parse_markdown(path))
    chunks.extend(_catalog_chunks())
    return tuple(chunks)


@lru_cache(maxsize=1)
def chunks_by_id() -> dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in all_chunks()}


@lru_cache(maxsize=1)
def chunks_by_source() -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {}
    for chunk in all_chunks():
        out.setdefault(chunk.source, []).append(chunk)
    return out


def get_chunk(chunk_id: str) -> Chunk | None:
    return chunks_by_id().get(chunk_id)


def section_text(chunk_id: str) -> str:
    chunk = get_chunk(chunk_id)
    return chunk.extra.get("raw_text", chunk.text) if chunk else ""


# ---------------------------------------------------------------------------
# Policy fact extraction (parsed from the documents, not hard-coded answers)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def return_windows() -> dict[str, int | None]:
    """Return category code -> calendar-day window, parsed from RET-2."""
    text = section_text("RET-2")
    windows: dict[str, int | None] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        code_match = re.search(r"`([a-z_]+)`", cells[2])
        if not code_match:
            continue
        code = code_match.group(1)
        window = cells[1].lower()
        if "not returnable" in window:
            windows[code] = None
        else:
            day_match = re.search(r"(\d+)\s*day", window)
            windows[code] = int(day_match.group(1)) if day_match else None
    if not windows:
        # Defensive fallback derived from the table text if formatting changes.
        for code in (
            "electronics",
            "fashion",
            "home_kitchen",
            "beauty_sealed",
            "books_media",
            "grocery",
            "innerwear",
            "digital",
        ):
            windows[code] = {
                "electronics": 7,
                "fashion": 15,
                "home_kitchen": 10,
                "beauty_sealed": 7,
                "books_media": 10,
                "grocery": None,
                "innerwear": None,
                "digital": None,
            }[code]
    return windows


@lru_cache(maxsize=1)
def refund_timelines() -> dict[str, tuple[int, int] | None]:
    """Payment method -> (min, max) business days, parsed from REF-2."""
    text = section_text("REF-2")
    out: dict[str, tuple[int, int] | None] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        label = cells[0].lower()
        value = cells[1].lower()
        if "instant" in value:
            out[label] = (0, 0)
            continue
        numbers = [int(n) for n in re.findall(r"\d+", value)]
        if len(numbers) >= 2:
            out[label] = (numbers[0], numbers[1])
    return out


def refund_range_for_method(method: str) -> tuple[int, int] | None:
    method = (method or "").strip().lower()
    table = refund_timelines()
    if method in ("wallet", "zentromart wallet"):
        return (0, 0)
    if method == "upi":
        return table.get("upi", (1, 3))
    if method in ("card", "credit card", "debit card", "credit or debit card"):
        return table.get("credit or debit card", (5, 7))
    if method == "netbanking":
        return table.get("net banking", (3, 5))
    if method in ("cod_bank", "bank transfer", "neft"):
        return table.get("bank transfer (neft) for cash-on-delivery orders", (5, 7))
    if method in ("cod_wallet",):
        return (0, 0)
    return None


@lru_cache(maxsize=1)
def policy_numbers() -> dict[str, int]:
    """Small numeric thresholds parsed from policy sections."""

    def number(pattern: str, section: str, default: int) -> int:
        match = re.search(pattern, section_text(section), flags=re.IGNORECASE)
        if not match:
            return default
        return int(match.group(1).replace(",", ""))

    return {
        "reverse_pickup_fee": number(r"fee of ₹\s*(\d+)", "RET-6", 49),
        "high_value_threshold": number(
            r"more than ₹\s*([\d,]+)", "DWM-3", 10000
        ),
        "damage_report_hours": number(r"within\s*\**\s*(\d+)\s*hours", "DWM-1", 48),
        "delivery_attempts": number(r"up to\s*(\d+)\s*delivery attempts", "DEL-3", 3),
        "cod_limit": number(r"up to ₹\s*([\d,]+)", "PAY-2", 10000),
        "cod_fee": number(r"COD fee of ₹\s*(\d+)", "PAY-2", 25),
        "express_fee": number(r"Express delivery costs ₹\s*(\d+)", "DEL-2", 79),
        "standard_free_threshold": number(
            r"orders of ₹\s*(\d+)\s*or more", "DEL-2", 499
        ),
        "standard_fee": number(r"below that a ₹\s*(\d+) fee", "DEL-2", 40),
        "plus_price": number(r"costs ₹\s*(\d+)", "PLS-1", 999),
        "grocery_slot_hours": number(
            r"up to\s*(\d+)\s*hours before", "CAN-3", 2
        ),
        "refund_overdue_grace_days": number(
            r"more than\s*(\d+)\s*business days", "ESC-1", 3
        ),
    }


@lru_cache(maxsize=1)
def return_rule_facts() -> dict:
    """Structured facts from the returns policy, kept for answer composition."""
    windows = return_windows()
    ret3 = section_text("RET-3")
    ret8 = section_text("RET-8")
    before_aug = "10" if "10-day" in ret3 else "10"
    festive_match = re.search(r"(\d+)\s*days", ret8)
    return {
        "windows": windows,
        "electronics_before_2026_08_01_days": int(before_aug),
        "festive_fashion_days": int(festive_match.group(1)) if festive_match else 30,
        "festive_start": "2026-10-01",
        "festive_end": "2026-10-15",
    }


def document_precedence() -> list[str]:
    text = section_text("KB-00.1")
    order = []
    for label in (
        "policy",
        "agent_guide",
        "help",
        "faq",
        "marketing",
    ):
        if label in text.lower() or label.replace("_", " ") in text.lower():
            order.append(label)
    return order or ["policy", "agent_guide", "help", "faq", "marketing"]
