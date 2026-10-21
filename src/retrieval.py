"""Hybrid BM25 + multilingual-embedding retrieval over section chunks."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config, data, kb

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


SYNONYMS = {
    # Hindi (Devanagari) -> English retrieval terms
    "वापस": "return return window",
    "वापसी": "return return window",
    "रिटर्न": "return",
    "रिफ़ंड": "refund refund timeline",
    "रिफंड": "refund refund timeline",
    "रिफन्ड": "refund",
    "कैंसिल": "cancel cancellation",
    "रद्द": "cancel cancellation",
    "ऑर्डर": "order",
    "आर्डर": "order",
    "कितने": "how many",
    "कितना": "how much how many",
    "दिन": "days",
    "कब": "when",
    "तक": "until by",
    "कहाँ": "where",
    "कहां": "where",
    "पैसे": "money refund",
    "पैसा": "money refund",
    "रुपये": "rupees amount",
    "टूटा": "damaged broken",
    "टूट": "damaged broken",
    "खराब": "damaged defective faulty",
    "गलत": "wrong incorrect",
    "नहीं": "not cannot",
    "सकता": "can",
    "सकती": "can",
    "क्या": "what is",
    "कैसे": "how",
    "क्यों": "why",
    "सामान": "item product",
    "डिलीवरी": "delivery",
    "पिकअप": "pickup",
    "शिप": "ship shipped",
    "कूपन": "coupon",
    "वारंटी": "warranty",
    "गारंटी": "warranty",
    "असुरेंस": "assured",
    "सदस्य": "member plus",
    "सदस्यता": "membership plus",
    "बदल": "exchange",
    "पार्सल": "parcel",
    "फ़ीस": "fee",
    "फीस": "fee",
    "घंटे": "hours",
    "तुरंत": "instant",
    "कार्ड": "card",
    "बैंक": "bank",
    "वॉलेट": "wallet",
    "शिकायत": "complaint report",
    "फ़ोटो": "photos",
    "फोटो": "photos",
    "सील": "seal sealed",
    "खोल": "opened unopened",
    "बंद": "closed",
    "मना": "refuse decline",
    "शुरू": "started initiated",
    "कब तक": "by when how long",
    "सहयोगी": "colleague human agent",
    # Romanised Hindi / Hinglish -> English
    "wapas": "return return window",
    "vapas": "return return window",
    "lautana": "return",
    "lauta": "return",
    "kitne": "how many",
    "kitna": "how much how many",
    "din": "days",
    "kab": "when",
    "tak": "until by",
    "kahan": "where",
    "paisa": "money refund",
    "paise": "money refund",
    "rupaye": "rupees amount",
    "kharab": "damaged defective faulty",
    "toota": "damaged broken",
    "tuta": "damaged broken",
    "galat": "wrong incorrect",
    "nahi": "not cannot",
    "nahin": "not cannot",
    "chahiye": "want should",
    "kaise": "how",
    "kya": "what is",
    "mera": "my",
    "meri": "my",
    "aayega": "come arrive",
    "aana": "come arrive",
    "shuru": "started initiated",
    "hua": "happened",
    "parso": "delivered recently",
    "kal": "yesterday tomorrow",
    "sale": "festive sale",
    "kam": "less",
    "zyada": "more",
    "jaldi": "fast quickly",
    "batao": "tell",
    "bata": "tell",
    "samajh": "understand",
    "baat": "talk",
    "insaan": "human person",
    "aadmi": "human person",
    "ticket": "ticket handover",
    "voucher": "voucher compensation",
    "muft": "free",
    "free": "free no fee",
    "milega": "get",
    "milegi": "get",
    "karna": "do",
    "kar": "do",
    "kiya": "did",
    "tha": "was",
    "thi": "was",
    "hai": "is",
    "hain": "are",
}

INTENT_BOOSTS: list[tuple[tuple[str, ...], tuple[str, ...], float]] = [
    (
        (
            "refund",
            "money back",
            "refund status",
            "credited",
            "rrn",
            "reference number",
            "wallet",
            "upi",
            "net banking",
            "netbanking",
            "card",
            "bank transfer",
        ),
        ("REF-1", "REF-2", "REF-3", "REF-4", "REF-6", "RET-7", "KB-00.1", "KB-00.3"),
        0.055,
    ),
    (
        (
            "return",
            "send back",
            "send it back",
            "exchange",
            "window",
            "how many days",
            "eligible",
            "returnable",
            "pickup fee",
        ),
        ("RET-2", "RET-3", "RET-4", "RET-5", "RET-6", "RET-8", "RET-9", "RETv3-2"),
        0.06,
    ),
    (
        ("warranty", "assured", "guarantee", "repair"),
        ("WAR-1", "WAR-2", "WAR-3", "MKS-3"),
        0.08,
    ),
    (
        ("cancel", "cancellation", "stop the order", "refuse the parcel"),
        ("CAN-1", "CAN-2", "CAN-3", "CAN-4", "FAQH-1"),
        0.07,
    ),
    (
        (
            "damaged",
            "damage",
            "broken",
            "crack",
            "defective",
            "wrong item",
            "missing",
            "expired",
            "torn",
        ),
        ("DWM-1", "DWM-2", "DWM-3", "DWM-4", "DWM-5", "ESC-1"),
        0.08,
    ),
    (
        (
            "delivery",
            "late",
            "delay",
            "express",
            "attempt",
            "not home",
            "courier",
            "shipped",
            "tracking",
            "where is my order",
            "not arrived",
        ),
        ("DEL-1", "DEL-2", "DEL-3", "DEL-4", "DEL-5", "DEL-6", "TRK-1", "TRK-2"),
        0.06,
    ),
    (("coupon", "offer", "discount", "promo"), ("CPN-1", "CPN-2", "CPN-3"), 0.08),
    (
        ("plus", "membership", "member"),
        ("PLS-1", "PLS-2", "PLS-3", "MKT-1"),
        0.07,
    ),
    (
        ("human", "person", "agent", "colleague", "escalate", "complaint", "legal"),
        ("ESC-1", "ESC-2", "ESC-3", "ESC-4", "FAQ-4"),
        0.08,
    ),
    (
        ("troubleshoot", "not turning on", "not powering", "not syncing", "charging"),
        ("TRB-1", "TRB-2", "TRB-3", "TRB-4"),
        0.09,
    ),
    (
        ("verify", "verification", "phone", "pincode", "order number"),
        ("IDV-1", "IDV-2", "IDV-3", "IDV-4"),
        0.06,
    ),
    (
        ("ai", "assistant", "who are you", "hello", "hi"),
        ("STY-1", "STY-2", "STY-3", "STY-4", "STY-5"),
        0.04,
    ),
]

DEFAULT_FIXED_AGENT_IDS = ("STY-1", "IDV-2", "IDV-3", "ESC-1", "ESC-2")


@dataclass
class Retrieved:
    chunk: kb.Chunk
    score: float
    bm25: float
    dense: float
    boost: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "doc_type": self.chunk.doc_type,
            "status": self.chunk.status,
            "score": round(self.score, 6),
            "section_title": self.chunk.section_title,
        }


def normalize_query(query: str) -> str:
    text = data.normalize_digits(query).lower()
    text = re.sub(r"[₹]", " rupees ", text)
    # Expand multi-word phrases first, then individual tokens.
    for source, target in sorted(SYNONYMS.items(), key=lambda item: -len(item[0])):
        pattern = re.escape(source.lower())
        if " " in source:
            text = re.sub(pattern, f" {target} ", text)
        else:
            text = re.sub(rf"(?<![\w\u0900-\u097F]){pattern}(?![\w\u0900-\u097F])", f" {target} ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u0900-\u097F]+", text.lower())


def _apply_intent_boosts(query: str) -> dict[str, float]:
    lower = query.lower()
    boosts: dict[str, float] = {}
    for triggers, section_ids, weight in INTENT_BOOSTS:
        if any(trigger in lower for trigger in triggers):
            for section_id in section_ids:
                boosts[section_id] = max(boosts.get(section_id, 0.0), weight)

    if "assured" in lower:
        boosts["WAR-2"] = max(boosts.get("WAR-2", 0.0), 0.18)
    if any(
        phrase in lower
        for phrase in ("opened", "unopened", "seal", "sealed", "खोल", "सील")
    ):
        boosts["RET-4"] = max(boosts.get("RET-4", 0.0), 0.12)
    if any(
        phrase in lower
        for phrase in ("send back", "send it back", "how long do i get", "return window")
    ):
        for section_id in ("RET-2", "RET-8"):
            boosts[section_id] = max(boosts.get(section_id, 0.0), 0.16)
    if any(
        phrase in lower
        for phrase in ("doesnt fit", "doesn't fit", "not fit", "innerwear", "swimwear", "swimsuit")
    ):
        for section_id in ("RET-2", "RET-5"):
            boosts[section_id] = max(boosts.get(section_id, 0.0), 0.16)
    if "return" in lower and any(
        phrase in lower for phrase in ("kar", "karna", "sakta", "sakti")
    ):
        boosts["RET-2"] = max(boosts.get("RET-2", 0.0), 0.14)
    if any(
        phrase in lower
        for phrase in ("not at home", "not home", "nobody home", "ghar par nahi", "ghar pe nahi")
    ):
        boosts["DEL-3"] = max(boosts.get("DEL-3", 0.0), 0.22)
    if "refund" in lower and any(
        phrase in lower
        for phrase in (
            "when will",
            "how long",
            "how fast",
            "kitne din",
            "kab tak",
            "kab aayega",
            "कितने दिन",
            "कब तक",
            "कब आएगा",
            "कब आयेगा",
        )
    ):
        boosts["REF-2"] = max(boosts.get("REF-2", 0.0), 0.14)
    return boosts


class HybridIndex:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or config.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = list(kb.all_chunks())
        self.texts = [chunk.text for chunk in self.chunks]
        self.ids = [chunk.chunk_id for chunk in self.chunks]
        self._id_to_pos = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        self._bm25 = None
        self._embeddings: np.ndarray | None = None
        self._model = None
        self._build_or_load()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(config.EMBEDDING_MODEL.encode("utf-8"))
        for chunk in self.chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(chunk.text.encode("utf-8"))
        return digest.hexdigest()

    def _build_or_load(self) -> None:
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi([tokenize(text) for text in self.texts])
        else:
            self._bm25 = None

        embedding_file = self.index_dir / "embeddings.npy"
        meta_file = self.index_dir / "meta.json"
        fingerprint = self._fingerprint()
        if embedding_file.exists() and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if (
                    meta.get("fingerprint") == fingerprint
                    and meta.get("model") == config.EMBEDDING_MODEL
                    and meta.get("ids") == self.ids
                ):
                    matrix = np.load(embedding_file)
                    if matrix.shape[0] == len(self.chunks):
                        self._embeddings = matrix.astype(np.float32)
                        return
            except Exception:
                pass

        if config.EMBEDDING_BACKEND in ("auto", "sentence_transformers") and SentenceTransformer:
            try:
                self._model = SentenceTransformer(config.EMBEDDING_MODEL)
                matrix = self._model.encode(
                    self.texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
                self._embeddings = np.asarray(matrix, dtype=np.float32)
                np.save(embedding_file, self._embeddings)
                meta_file.write_text(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "model": config.EMBEDDING_MODEL,
                            "ids": self.ids,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                self._embeddings = None

    def _embed_query(self, query: str) -> np.ndarray | None:
        if self._embeddings is None:
            return None
        if self._model is None:
            try:
                self._model = SentenceTransformer(config.EMBEDDING_MODEL)
            except Exception:
                return None
        vector = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        fixed_ids: tuple[str, ...] | list[str] | None = None,
        include_catalog: bool = True,
    ) -> list[Retrieved]:
        top_k = top_k or config.RETRIEVAL_TOP_K
        expanded = normalize_query(query)
        tokens = tokenize(expanded)

        bm25_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self._bm25 is not None and tokens:
            raw = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
            if raw.max() > 0:
                bm25_scores = raw / raw.max()

        dense_scores = np.zeros(len(self.chunks), dtype=np.float32)
        query_vector = self._embed_query(expanded)
        if query_vector is not None and self._embeddings is not None:
            dense = self._embeddings @ query_vector
            dense = (dense + 1.0) / 2.0
            dense_scores = np.clip(dense, 0.0, 1.0).astype(np.float32)

        boosts = _apply_intent_boosts(expanded)
        results: list[Retrieved] = []
        for pos, chunk in enumerate(self.chunks):
            if not include_catalog and chunk.doc_type == "catalog":
                continue
            metadata_weight = 0.55 + 0.45 * chunk.priority
            base = 0.52 * float(bm25_scores[pos]) + 0.48 * float(dense_scores[pos])
            boost = boosts.get(chunk.chunk_id, 0.0)
            # A small exact-token bonus for the section's own title.
            if chunk.chunk_id.lower() in expanded.lower():
                boost += 0.08
            score = base * metadata_weight + boost
            results.append(
                Retrieved(
                    chunk=chunk,
                    score=score,
                    bm25=float(bm25_scores[pos]),
                    dense=float(dense_scores[pos]),
                    boost=boost,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        selected = results[:top_k]

        if fixed_ids:
            selected_ids = {item.chunk.chunk_id for item in selected}
            for fixed in fixed_ids:
                if fixed in selected_ids or fixed not in self._id_to_pos:
                    continue
                pos = self._id_to_pos[fixed]
                chunk = self.chunks[pos]
                selected.append(
                    Retrieved(
                        chunk=chunk,
                        score=0.0,
                        bm25=float(bm25_scores[pos]),
                        dense=float(dense_scores[pos]),
                        boost=0.0,
                    )
                )
                selected_ids.add(fixed)
        return selected


@lru_cache(maxsize=1)
def get_index() -> HybridIndex:
    return HybridIndex()


def search(
    query: str,
    top_k: int | None = None,
    fixed_ids: tuple[str, ...] | list[str] | None = None,
    include_catalog: bool = True,
) -> list[Retrieved]:
    return get_index().search(
        query,
        top_k=top_k,
        fixed_ids=fixed_ids,
        include_catalog=include_catalog,
    )


def context_block(results: list[Retrieved], max_chars: int = 9000) -> str:
    parts: list[str] = []
    used = 0
    for item in results:
        chunk = item.chunk
        block = (
            f"[{chunk.chunk_id}] source={chunk.source} type={chunk.doc_type} "
            f"status={chunk.status} effective_from={chunk.effective_from}\n"
            f"{chunk.text}"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def evidence(results: list[Retrieved]) -> list[dict]:
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "source": item.chunk.source,
            "doc_type": item.chunk.doc_type,
            "status": item.chunk.status,
        }
        for item in results
    ]
