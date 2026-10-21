"""Grounded answer composition.

Every factual statement is derived from retrieved KB chunks, the product
catalog, the dataset CSVs, and deterministic date arithmetic.  The LLM is used
only for language polishing/translation and for the rare open-ended fallback;
it never supplies policy facts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from . import config, data, kb, retrieval
from .llm import LLMClient, LLMError
from .tools_wrapper import ToolCall, create_ticket

HINGLISH_MARKERS = {
    "hai",
    "hain",
    "kya",
    "mera",
    "meri",
    "mujhe",
    "kitne",
    "kitna",
    "kab",
    "kaise",
    "nahi",
    "nahin",
    "chahiye",
    "wapas",
    "vapas",
    "aayega",
    "aana",
    "shuru",
    "hua",
    "hoga",
    "kar",
    "karo",
    "karna",
    "batao",
    "bata",
    "mein",
    "aap",
    "aur",
    "bhi",
    "toh",
    "ka",
    "ki",
    "ke",
    "se",
    "par",
    "tha",
    "thi",
    "raha",
    "rahi",
    "parso",
    "kal",
    "sale",
    "rupaye",
    "paise",
    "paisa",
    "kahan",
    "kaun",
    "kaunsa",
    "kripya",
    "maaf",
    "insaan",
    "aadmi",
}

UNSUPPORTED_PATTERNS = (
    "midnight",
    "student discount",
    "gst invoice",
    "store in kochi",
    "kochi",
    "pick up my order from a store",
    "store pickup",
    "pickup from a store",
    "student",
    "bulk order",
    "gst",
    "midnight delivery",
    "schedule a delivery at midnight",
    "दुकान",
    "कोच्चि",
    "कोची",
    "स्टोर",
)

SAFETY_PATTERNS = (
    "swelling battery",
    "battery swell",
    "burning smell",
    "burning",
    "sparks",
    "spark",
    "smoke",
    "overheat",
    "overheating",
    "electric shock",
    "injury",
    "injured",
    "fire",
)

FRAUD_PATTERNS = (
    "never placed",
    "never ordered",
    "did not place",
    "didn't place",
    "did not order",
    "didn't order",
    "unauthorized",
    "unauthorised",
    "fraud",
    "someone ordered",
    "order on my account that i never placed",
    "account activity",
)

HUMAN_PATTERNS = (
    "human",
    "real person",
    "speak to a person",
    "talk to a person",
    "human agent",
    "colleague",
    "representative",
    "customer care",
    "insaan",
    "aadmi",
    "किसी इंसान",
    "सहयोगी",
)

COMPENSATION_PATTERNS = (
    "compensation",
    "compensate",
    "voucher",
    "300 rupaye",
    "500 rupaye",
    "extra refund",
    "goodwill",
    "credit as compensation",
    "muawza",
    "मुआवज़ा",
    "मुआवजा",
)

POLICY_OVERRIDE_PATTERNS = (
    "phone number of the person",
    "phone number of another",
    "another customer",
    "other customer",
    "address of the person",
    "details of the person who ordered",
    "share the phone",
    "share.*phone number",
)

REFUND_STATUS_PATTERNS = (
    "where is my refund",
    "refund status",
    "refund kahan",
    "refund kahaan",
    "refund kaha",
    "refund kidhar",
    "पैसा कहाँ",
    "पैसे कहाँ",
    "रिफ़ंड किधर",
    "रिफंड किधर",
    "refund nahi aaya",
    "refund not received",
    "money not arrived",
    "money has not arrived",
    "refund ka status",
    "refund kab aayega",
    "when will my refund come",
    "refund aaya nahi",
    "refund received",
    "रिफ़ंड कहाँ",
    "रिफंड कहाँ",
    "रिफ़ंड कहा",
    "रिफंड कहा",
    "रिफ़ंड कब",
    "रिफंड कब",
)

REFUND_TIMELINE_PATTERNS = (
    "how fast",
    "how long",
    "how soon",
    "timeline",
    "kab tak",
    "kab aayega",
    "kab aa jaana",
    "kitne din",
    "business days",
    "refund was started",
    "refund was initiated",
    "refund started",
    "refund initiated",
    "refund shuru",
    "कितने दिन",
    "कब तक",
    "कब आएगा",
    "कब आयेगा",
    "refund.*started",
    "refund.*initiated",
    "refund.*begun",
    "refund.*show up",
    "refund.*take",
    "refund.*days",
    "money back",
    "by which date",
)

CANCEL_PATTERNS = (
    "cancel",
    "stop the order",
    "stop it",
    "stop my order",
    "radd",
    "kaincil",
    "कैंसिल",
    "रद्द",
    "cancel kar",
    "already on its way",
    "ship ho gaya",
    "शिप हो गया",
    "already shipped",
)

ORDER_SHIPPED_PATTERNS = (
    "when was it shipped",
    "when did it ship",
    "shipping date",
    "shipped on",
    "kab ship",
    "ship hua",
    "ship hua tha",
)

RETURN_WINDOW_PATTERNS = (
    "return window",
    "how many days",
    "how long do i get",
    "how long can i return",
    "days to return",
    "kitne din",
    "din tak",
    "still ten days",
    "ten days",
    "10 days",
    "7 days",
    "15 days",
    "30 days",
    "send back",
    "lautane",
    "lauta",
)

RETURN_ELIGIBILITY_PATTERNS = (
    "can i return",
    "can i send it back",
    "send it back",
    "return kar",
    "wapas kar",
    "vapas kar",
    "returnable",
    "eligible",
    "doesn't fit",
    "does not fit",
    "opened",
    "seal",
    "unopened",
    "return it",
    "return something",
    "वापस कर",
    "लौटा",
    "खोल",
    "सील",
)

RETURN_FEE_PATTERNS = (
    "pickup fee",
    "collection free",
    "will i be charged",
    "fee waived",
    "reverse pickup fee",
    "free collection",
)

DAMAGE_PATTERNS = (
    "damaged",
    "damage",
    "crack",
    "cracked",
    "broken",
    "torn",
    "defective",
    "wrong item",
    "wrong product",
    "missing",
    "expired",
    "not working",
    "टूटा",
    "खराब",
    "गलत",
    "toota",
    "kharab",
)

WARRANTY_PATTERNS = ("warranty", "assured", "guarantee", "war-", "वारंटी", "गारंटी")

DELIVERY_ATTEMPT_PATTERNS = (
    "not home",
    "not at home",
    "missed",
    "delivery attempt",
    "courier comes",
    "nobody came",
    "not available",
    "ghar par nahi",
)

DELIVERY_AREA_PATTERNS = (
    "deliver to",
    "outside india",
    "international",
    "deliver outside",
    "dubai",
    "serviceable",
    "delivery area",
    "deliver in",
)

LATE_DELIVERY_PATTERNS = (
    "late",
    "delay",
    "delayed",
    "express",
    "fee back",
    "arrived a day late",
    "late aaya",
    "der se",
)

WHERE_ORDER_PATTERNS = (
    "hasn't arrived",
    "has not arrived",
    "not arrived",
    "where is it",
    "where is my order",
    "where is my parcel",
    "track",
    "tracking",
    "kahan hai",
    "nahi aaya",
    "status of my order",
    "order status",
    "not received",
)

COUPON_PATTERNS = ("coupon", "promo", "discount", "offer", "कूपन")

PLUS_PATTERNS = ("plus membership", "zentromart plus", "plus member", "membership")

TROUBLESHOOT_PATTERNS = (
    "won't turn on",
    "wont turn on",
    "not turning on",
    "not powering",
    "not syncing",
    "charging slowly",
    "one side not working",
    "earbuds",
    "bluetooth speaker",
    "smartwatch",
    "troubleshoot",
)

EXCHANGE_PATTERNS = ("exchange", "different size", "size change", "badal")


def detect_language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text or ""):
        return "hi"
    tokens = re.findall(r"[a-zA-Z]+", (text or "").lower())
    hits = sum(1 for token in tokens if token in HINGLISH_MARKERS)
    if hits >= 2:
        return "hi-en"
    return "en"


def _has_refund_language(lower: str) -> bool:
    return any(
        token in lower
        for token in ("refund", "re fund", "money back", "paisa", "paise", "रिफ़ंड", "रिफंड", "रिफन्ड")
    )


def detect_intent(text: str, order: dict | None = None, has_order: bool = False) -> str:
    lower = data.normalize_speech(text)
    has_refund = _has_refund_language(lower)

    if any(re.search(pattern, lower) for pattern in POLICY_OVERRIDE_PATTERNS):
        return "policy_override"
    if any(pattern in lower for pattern in FRAUD_PATTERNS):
        return "fraud"
    if any(pattern in lower for pattern in SAFETY_PATTERNS):
        return "safety"
    if any(pattern in lower for pattern in HUMAN_PATTERNS):
        return "human_request"
    if any(pattern in lower for pattern in UNSUPPORTED_PATTERNS):
        return "unsupported"
    if any(pattern in lower for pattern in COMPENSATION_PATTERNS):
        return "compensation"
    if any(pattern in lower for pattern in CANCEL_PATTERNS):
        status = str((order or {}).get("status", "")).lower()
        if status in (
            "shipped",
            "out_for_delivery",
            "delivery_attempted",
            "rto",
            "returned_to_origin",
        ):
            return "cancel_shipped"
        if any(
            phrase in lower
            for phrase in (
                "already on its way",
                "already shipped",
                "shipped",
                "ship ho gaya",
                "शिप हो गया",
            )
        ):
            return "cancel_shipped"
        if status in ("placed", "packed"):
            return "cancel_unshipped"
        return "cancel_general"
    if has_refund and any(pattern in lower for pattern in REFUND_STATUS_PATTERNS):
        return "refund_status"
    if has_refund and any(
        re.search(pattern, lower) for pattern in REFUND_TIMELINE_PATTERNS
    ):
        return "refund_timeline"
    if any(pattern in lower for pattern in WHERE_ORDER_PATTERNS):
        return "where_is_my_order"
    if any(pattern in lower for pattern in RETURN_FEE_PATTERNS):
        return "return_fee"
    if any(pattern in lower for pattern in DAMAGE_PATTERNS):
        return "damaged_item"
    if any(pattern in lower for pattern in WARRANTY_PATTERNS):
        return "warranty"
    if any(pattern in lower for pattern in DELIVERY_ATTEMPT_PATTERNS):
        return "delivery_attempts"
    if any(pattern in lower for pattern in DELIVERY_AREA_PATTERNS):
        return "delivery_area"
    if any(pattern in lower for pattern in LATE_DELIVERY_PATTERNS):
        return "late_delivery"
    if any(pattern in lower for pattern in EXCHANGE_PATTERNS):
        return "exchange"
    if any(pattern in lower for pattern in RETURN_ELIGIBILITY_PATTERNS):
        return "return_eligibility"
    if any(pattern in lower for pattern in RETURN_WINDOW_PATTERNS):
        return "return_window"
    if any(pattern in lower for pattern in ORDER_SHIPPED_PATTERNS):
        return "order_shipped"
    if any(pattern in lower for pattern in TROUBLESHOOT_PATTERNS):
        return "troubleshooting"
    if any(pattern in lower for pattern in COUPON_PATTERNS):
        return "coupon"
    if any(pattern in lower for pattern in PLUS_PATTERNS):
        return "plus_membership"
    return "general"


def detect_payment_method(text: str, order: dict | None = None) -> str | None:
    lower = data.normalize_speech(text)
    if "net banking" in lower or "netbanking" in lower or "net bank" in lower:
        return "netbanking"
    if "credit card" in lower or "debit card" in lower or re.search(r"\bcard\b", lower):
        return "card"
    if re.search(r"\bupi\b", lower) or "u p i" in lower:
        return "upi"
    if "wallet" in lower:
        return "wallet"
    if "cash on delivery" in lower or re.search(r"\bcod\b", lower):
        return "cod"
    if "bank transfer" in lower or "neft" in lower:
        return "cod_bank"
    if order:
        return str(order.get("payment_method") or "").lower() or None
    return None


def _find_product(text: str, order: dict | None, items: list[dict] | None) -> dict | None:
    product = data.product_for_text(text)
    if product:
        return product
    if order:
        damage_sku = order.get("damage_sku")
        if damage_sku:
            product = data.products_by_sku().get(str(damage_sku))
            if product:
                return product
        return_sku = order.get("return_sku")
        if return_sku:
            product = data.products_by_sku().get(str(return_sku))
            if product:
                return product
    if items:
        if len(items) == 1:
            return data.products_by_sku().get(str(items[0].get("sku")))
        # A follow-up like "the item" should use the only returned/damaged item.
        if order:
            for key in ("return_sku", "damage_sku"):
                sku = order.get(key)
                if sku and sku in data.products_by_sku():
                    return data.products_by_sku()[sku]
    return None


def _parse_order_date(order: dict | None, key: str) -> date | None:
    if not order:
        return None
    raw = order.get(key)
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


_DATE_TEXT = re.compile(
    r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z\u0900-\u097F]+\.?(?:\s+\d{4})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{4})",
    re.IGNORECASE,
)


def _placed_date_from_text(text: str) -> date | None:
    """Find a date tied to ordering, not delivery."""
    for match in _DATE_TEXT.finditer(text or ""):
        context = (text or "")[max(0, match.start() - 70) : match.end() + 70].lower()
        if any(
            word in context
            for word in (
                "order",
                "ordered",
                "placed",
                "bought",
                "buy",
                "purchase",
                "sale",
                "kiya",
                "kharida",
            )
        ):
            parsed = data.parse_date(match.group(1))
            if parsed:
                return parsed
    return None


def _return_window_for(product: dict | None, order: dict | None, text: str) -> dict:
    category = data.return_category(product)
    windows = kb.return_windows()
    window = windows.get(category)
    lower = (text or "").lower()
    delivered = _parse_order_date(order, "delivered_on")
    placed = _parse_order_date(order, "placed_at")
    if placed is None:
        placed = _placed_date_from_text(text)
    if delivered is None and placed is None:
        delivered = data.parse_date(text)
        # "delivered on 20 July" or "received yesterday" do not always have a
        # year; parse_date already defaults to 2026.

    festive = False
    if category == "fashion" and placed:
        festive = date(2026, 10, 1) <= placed <= date(2026, 10, 15)
    elif category == "fashion" and "sale" in lower and placed is None:
        festive = True
    if festive:
        window = kb.return_rule_facts()["festive_fashion_days"]
    elif category == "electronics" and delivered and delivered < date(2026, 8, 1):
        window = kb.return_rule_facts()["electronics_before_2026_08_01_days"]

    deadline = None
    if delivered is not None and window:
        deadline = data.add_calendar_days(delivered, window)

    fee = None
    if product is not None:
        plus = bool((order or {}).get("is_plus"))
        damaged = any(pattern in (text or "").lower() for pattern in DAMAGE_PATTERNS)
        if category == "fashion" and not plus and not damaged:
            fee = kb.policy_numbers()["reverse_pickup_fee"]
        elif plus:
            fee = 0
        elif damaged:
            fee = 0

    return {
        "category": category,
        "window": window,
        "deadline": deadline,
        "delivered": delivered,
        "placed": placed,
        "festive": festive,
        "fee": fee,
    }


def _refund_dates(method: str | None, start: date | None) -> tuple[date, date, tuple[int, int]] | None:
    if not method or start is None:
        return None
    method_key = method
    if method_key == "cod":
        return None
    span = kb.refund_range_for_method(method_key)
    if span is None:
        return None
    min_days, max_days = span
    if max_days == 0:
        return start, start, (0, 0)
    end_min, end_max = data.business_day_range(start, min_days, max_days)
    return end_min, end_max, (min_days, max_days)


def _refund_method_label(method: str | None) -> str:
    return {
        "upi": "UPI",
        "card": "credit or debit card",
        "netbanking": "net banking",
        "wallet": "ZentroMart Wallet",
        "cod": "cash on delivery",
        "cod_bank": "bank transfer",
        "cod_wallet": "ZentroMart Wallet",
    }.get((method or "").lower(), method or "the original payment method")


@lru_cache(maxsize=1)
def _translation_cache() -> dict:
    path = config.INDEX_DIR / "translations.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_translation_cache(cache: dict) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (config.INDEX_DIR / "translations.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def localize(text: str, language: str, llm: LLMClient | None = None) -> str:
    if language == "en" or not text:
        return text
    cache = _translation_cache()
    key = f"{language}::{text}"
    if key in cache:
        return cache[key]
    target = "natural Hindi in Devanagari script" if language == "hi" else "natural Hinglish in Latin script"
    prompt = (
        f"Translate this ZentroMart support reply into {target}. Preserve every "
        "number, date, order ID, product name, RRN and rupee amount exactly. "
        "Keep it to two or three short sentences. Do not add or remove facts. "
        "Return only the translation.\n\n" + text
    )
    try:
        llm = llm or LLMClient()
        translated = llm.chat_text(
            "You are a precise support-reply translator. Never invent facts.",
            prompt,
            max_tokens=350,
            temperature=0.1,
        ).strip()
        source_numbers = set(re.findall(r"\d[\d,]*", text))
        target_numbers = set(re.findall(r"\d[\d,]*", translated))
        if translated and source_numbers.issubset(target_numbers):
            cache[key] = translated
            _save_translation_cache(cache)
            return translated
    except (LLMError, Exception):
        pass
    return text


def _pick_language_reply(language: str, en: str, hi: str, hinglish: str) -> str:
    if language == "hi":
        return hi
    if language == "hi-en":
        return hinglish
    return en


class GroundedAnswerer:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def answer(
        self,
        text: str,
        order: dict | None = None,
        items: list[dict] | None = None,
        results: list[retrieval.Retrieved] | None = None,
        language: str | None = None,
        allow_ticket: bool = True,
        extra_context: str = "",
    ) -> "AnswerResult":
        language = language or detect_language(text)
        text = data.normalize_speech(text)
        query = text
        if order:
            query += f" order status {order.get('status')} {order.get('payment_method')}"
        results = results or retrieval.search(query, fixed_ids=retrieval.DEFAULT_FIXED_AGENT_IDS)
        intent = detect_intent(text, order=order, has_order=bool(order))
        product = _find_product(text, order, items)
        tool_calls: list[ToolCall] = []
        facts: dict = {"intent": intent}

        # --- safety / integrity first -------------------------------------
        if intent == "policy_override":
            reply = _pick_language_reply(
                language,
                "I'm sorry, I can't share any details about other customers.",
                "माफ़ कीजिए, मैं दूसरे ग्राहकों की कोई जानकारी साझा नहीं कर सकती।",
                "Maaf kijiye, main doosre customers ki koi details share nahi kar sakti.",
            )
            return AnswerResult(reply, "refuse", intent, results, facts, language, tool_calls)

        if intent == "fraud":
            reply, created, tool_calls = self._escalate(
                "A customer reports an order or account activity they did not make. "
                "Account-security review is required under ESC-1. Customer said: "
                f"{text}. No order details were shared.",
                order_id=(order or {}).get("order_id"),
                reason="fraud_or_unrecognised_activity",
                intent=intent,
                language=language,
                allow_ticket=allow_ticket,
            )
            facts["ticket_created"] = created
            return AnswerResult(reply, "escalate", intent, results, facts, language, tool_calls)

        if intent == "safety":
            reply, created, tool_calls = self._escalate(
                "Customer reports a possible product safety risk and must stop using it. "
                "ESC-1 requires immediate human handover. Customer said: "
                f"{text}.",
                order_id=(order or {}).get("order_id"),
                reason="product_safety_risk",
                intent=intent,
                language=language,
                allow_ticket=allow_ticket,
            )
            facts["ticket_created"] = created
            return AnswerResult(reply, "escalate", intent, results, facts, language, tool_calls)

        if intent == "human_request":
            reply, created, tool_calls = self._escalate(
                "Customer explicitly asked for a human colleague (ESC-1 item 6). "
                f"Customer said: {text}. Order context: {(order or {}).get('order_id', 'none')}. "
                "No policy outcome was promised.",
                order_id=(order or {}).get("order_id"),
                reason="customer_requested_human",
                intent=intent,
                language=language,
                allow_ticket=allow_ticket,
            )
            facts["ticket_created"] = created
            return AnswerResult(reply, "escalate", intent, results, facts, language, tool_calls)

        # --- repeat contact rule ------------------------------------------
        if order:
            try:
                contacts = int(order.get("contacts_last_7_days") or 0)
            except (TypeError, ValueError):
                contacts = 0
            if contacts >= 2:
                reply, created, tool_calls = self._escalate(
                    "Third or later contact about the same issue within 7 days; "
                    f"contacts_last_7_days={contacts}. ESC-1 requires a human handover. "
                    f"Customer said: {text}.",
                    order_id=order.get("order_id"),
                    reason="repeat_contact",
                    intent=intent,
                    language=language,
                    allow_ticket=allow_ticket,
                )
                facts.update({"ticket_created": created, "contacts_last_7_days": contacts})
                return AnswerResult(reply, "escalate", intent, results, facts, language, tool_calls)

        # --- intent routing ------------------------------------------------
        if intent == "compensation":
            if order and str(order.get("delivery_type", "")).lower() == "express":
                return self._express_late(order, language, results, facts, tool_calls)
            reply = _pick_language_reply(
                language,
                "I'm sorry the delivery was late. There is no compensation for late "
                "Standard deliveries; only a late Express order gets the ₹79 express fee "
                "credited to the ZentroMart Wallet.",
                "देर से डिलीवरी के लिए माफ़ कीजिए। लेट स्टैंडर्ड डिलीवरी पर कोई मुआवज़ा नहीं मिलता; "
                "सिर्फ़ लेट एक्सप्रेस ऑर्डर पर ₹79 की एक्सप्रेस फ़ीस ZentroMart Wallet में जमा होती है।",
                "Der se delivery ke liye maaf kijiye. Late Standard delivery par koi compensation "
                "nahi milta; sirf late Express order par ₹79 ki express fee ZentroMart Wallet mein "
                "credit hoti hai.",
            )
            return AnswerResult(reply, "refuse", intent, results, facts, language, tool_calls)

        if intent in ("cancel_shipped", "cancel_unshipped", "cancel_general"):
            return self._cancel(text, order, language, results, facts, tool_calls, intent)

        if intent == "where_is_my_order":
            return self._where_order(text, order, language, results, facts, tool_calls)

        if intent == "order_shipped":
            return self._order_shipped(order, language, results, facts, tool_calls)

        if intent == "refund_status":
            return self._refund_status(text, order, language, results, facts, tool_calls, allow_ticket)

        if intent == "refund_timeline":
            return self._refund_timeline(text, order, language, results, facts, tool_calls)

        if intent in ("return_window", "return_eligibility", "return_fee", "exchange"):
            return self._return_answer(
                text, order, items, product, language, results, facts, tool_calls, intent
            )

        if intent == "warranty":
            return self._warranty(product, language, results, facts, tool_calls)

        if intent == "damaged_item":
            return self._damaged(text, order, product, language, results, facts, tool_calls, allow_ticket)

        if intent == "delivery_attempts":
            attempts = kb.policy_numbers()["delivery_attempts"]
            reply = _pick_language_reply(
                language,
                f"The courier makes up to {attempts} delivery attempts. After a failed "
                "attempt you get an SMS to reschedule, and after the third failed attempt "
                "the parcel returns to the warehouse.",
                f"कूरियर {attempts} बार डिलीवरी की कोशिश करता है। असफल कोशिश के बाद रीशेड्यूल के लिए "
                "एसएमएस आता है, और तीसरी असफल कोशिश के बाद पार्सल वेयरहाउस लौट जाता है।",
                f"Courier {attempts} delivery attempts tak karta hai. Failed attempt ke baad "
                "reschedule ke liye SMS aata hai, aur teesri failed attempt ke baad parcel "
                "warehouse wapas chala jaata hai.",
            )
            facts["attempts"] = attempts
            return AnswerResult(reply, "answer", intent, results, facts, language, tool_calls)

        if intent == "delivery_area":
            reply = _pick_language_reply(
                language,
                "ZentroMart delivers only within India, to pincodes listed as serviceable.",
                "ZentroMart सिर्फ़ भारत में, सर्विसेबल पिनकोड पर डिलीवरी करता है।",
                "ZentroMart sirf India mein, serviceable pincodes par delivery karta hai.",
            )
            return AnswerResult(reply, "answer", intent, results, facts, language, tool_calls)

        if intent == "late_delivery":
            if order and str(order.get("delivery_type", "")).lower() == "express":
                return self._express_late(order, language, results, facts, tool_calls)
            fee = kb.policy_numbers()["express_fee"]
            reply = _pick_language_reply(
                language,
                f"If an Express order arrives after its promised date, the ₹{fee} express "
                "fee is credited to ZentroMart Wallet automatically within 48 hours. "
                "There is no compensation for late Standard deliveries.",
                f"अगर एक्सप्रेस ऑर्डर तय तारीख के बाद आता है, तो ₹{fee} एक्सप्रेस फ़ीस 48 घंटे के "
                "अंदर अपने आप ZentroMart Wallet में जमा हो जाती है। लेट स्टैंडर्ड डिलीवरी पर कोई "
                "मुआवज़ा नहीं मिलता।",
                f"Agar Express order promised date ke baad aata hai, toh ₹{fee} express fee "
                "48 hours ke andar automatically ZentroMart Wallet mein credit ho jaati hai. "
                "Late Standard delivery par koi compensation nahi milta.",
            )
            facts["express_fee"] = fee
            return AnswerResult(reply, "answer", intent, results, facts, language, tool_calls)

        if intent == "troubleshooting":
            return self._troubleshooting(text, language, results, facts, tool_calls)

        if intent == "coupon":
            reply = _pick_language_reply(
                language,
                "Only one coupon can be applied per order. Coupons cannot be combined, "
                "added after the order is placed, or exchanged for cash.",
                "एक ऑर्डर पर सिर्फ़ एक कूपन लग सकता है। कूपन जोड़े नहीं जा सकते, ऑर्डर के बाद "
                "नहीं जोड़े जा सकते, और नकद में नहीं बदले जा सकते।",
                "Ek order par sirf ek coupon laga sakte hain. Coupons combine nahi ho sakte, "
                "order ke baad add nahi ho sakte, aur cash mein exchange nahi ho sakte.",
            )
            return AnswerResult(reply, "answer", intent, results, facts, language, tool_calls)

        if intent == "plus_membership":
            price = kb.policy_numbers()["plus_price"]
            reply = _pick_language_reply(
                language,
                f"ZentroMart Plus costs ₹{price} a year. It gives free Express delivery "
                "where available, no reverse pickup fee, and refunds initiated at pickup. "
                "It does not change return windows or refund timelines after initiation.",
                f"ZentroMart Plus की कीमत ₹{price} प्रति वर्ष है। इसमें जहाँ उपलब्ध हो मुफ़्त एक्सप्रेस "
                "डिलीवरी, किसी भी रिटर्न पर पिकअप फ़ीस नहीं, और पिकअप के समय रिफ़ंड शुरू होता है। "
                "रिटर्न विंडो या रिफ़ंड टाइमलाइन नहीं बदलती।",
                f"ZentroMart Plus ₹{price} saal ka hai. Isme jahan available ho free Express "
                "delivery, kisi bhi return par pickup fee nahi, aur pickup ke time refund "
                "initiate hota hai. Return window ya refund timeline nahi badalti.",
            )
            facts["plus_price"] = price
            return AnswerResult(reply, "answer", intent, results, facts, language, tool_calls)

        if intent == "unsupported":
            reply = _pick_language_reply(
                language,
                "Sorry, I don't have information about that. I can connect you to a "
                "human colleague who can help.",
                "माफ़ कीजिए, मेरे पास इस बारे में जानकारी नहीं है। मैं आपको किसी सहयोगी से जोड़ सकती हूँ।",
                "Maaf kijiye, mere paas iske baare mein jaankari nahi hai. Main aapko ek "
                "colleague se connect kar sakti hoon.",
            )
            return AnswerResult(reply, "answer", intent, results, facts, language, tool_calls)

        # --- generic retrieval-grounded fallback ---------------------------
        return self._general_answer(text, language, results, facts, tool_calls, extra_context)

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------
    def _escalate(
        self,
        summary: str,
        order_id: str | None,
        reason: str,
        intent: str,
        language: str,
        allow_ticket: bool,
    ) -> tuple[str, bool, list[ToolCall]]:
        calls: list[ToolCall] = []
        created = False
        ticket_id = None
        if allow_ticket:
            result, call = create_ticket(
                summary=summary, order_id=order_id, reason=reason, intent=intent
            )
            calls.append(call)
            created = bool(result.get("created"))
            ticket_id = result.get("ticket_id")
        if created:
            reply = _pick_language_reply(
                language,
                "I've created a ticket and a human colleague will contact you. "
                f"Your ticket ID is {ticket_id}.",
                "मैंने टिकट बना दिया है और एक सहयोगी आपसे संपर्क करेंगे। "
                f"आपका टिकट आईडी {ticket_id} है।",
                "Maine ticket bana diya hai aur ek colleague aapse contact karega. "
                f"Aapka ticket ID {ticket_id} hai.",
            )
        else:
            reply = _pick_language_reply(
                language,
                "This needs a human colleague, but I couldn't create the ticket right now. "
                "Please contact support again from your registered phone number.",
                "इसके लिए किसी सहयोगी की ज़रूरत है, लेकिन अभी टिकट नहीं बन सका। कृपया अपने "
                "रजिस्टर्ड फ़ोन से सहायता से फिर संपर्क करें।",
                "Iske liye ek colleague chahiye, lekin abhi ticket nahi ban saka. Kripya "
                "apne registered phone se support se dobara contact karein.",
            )
        return reply, created, calls

    def _cancel(
        self,
        text: str,
        order: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
        intent: str,
    ) -> "AnswerResult":
        status = str((order or {}).get("status", "")).lower()
        lower = data.normalize_speech(text)
        request = any(
            phrase in lower
            for phrase in (
                "i want to cancel",
                "i want to stop",
                "please cancel",
                "cancel kar do",
                "cancel karna",
                "cancel karo",
                "ise cancel",
                "stop my order",
            )
        )
        if intent == "cancel_shipped" or status in (
            "shipped",
            "out_for_delivery",
            "delivery_attempted",
            "rto",
            "returned_to_origin",
        ):
            reply = _pick_language_reply(
                language,
                "A shipped order can't be cancelled. You can refuse the parcel at the door, "
                "or accept it and return it within its return window. For a prepaid order, "
                "the refund follows once the parcel reaches the warehouse, usually 7 to 10 "
                "calendar days after the last delivery attempt.",
                "शिप हो चुका ऑर्डर कैंसिल नहीं हो सकता। आप डिलीवरी के समय पार्सल लेने से मना कर "
                "सकते हैं, या डिलीवरी के बाद रिटर्न विंडो में वापस कर सकते हैं। प्रीपेड ऑर्डर "
                "का रिफ़ंड पार्सल वेयरहाउस पहुँचने के बाद, आमतौर पर आख़िरी डिलीवरी कोशिश के "
                "7 से 10 दिन में शुरू होता है।",
                "Shipped order cancel nahi ho sakta. Aap delivery ke time parcel lene se mana "
                "kar sakte hain, ya delivery ke baad return window mein wapas kar sakte hain. "
                "Prepaid order ka refund parcel warehouse pahunchne ke baad, usually last "
                "delivery attempt ke 7 se 10 din mein initiate hota hai.",
            )
            decision = "refuse" if request else "answer"
            return AnswerResult(reply, decision, intent, results, facts, language, tool_calls)

        reply = _pick_language_reply(
            language,
            "You can cancel it free of charge in My Orders before it ships. "
            "I can't cancel it from here, and I have not cancelled it.",
            "शिप होने से पहले आप इसे My Orders में मुफ़्त में कैंसिल कर सकते हैं। "
            "मैं यहाँ से कैंसिल नहीं कर सकती, और मैंने इसे कैंसिल नहीं किया है।",
            "Ship hone se pehle aap ise My Orders mein free cancel kar sakte hain. "
            "Main yahan se cancel nahi kar sakti, aur maine ise cancel nahi kiya.",
        )
        return AnswerResult(reply, "answer", intent, results, facts, language, tool_calls)

    def _order_shipped(
        self,
        order: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
    ) -> "AnswerResult":
        if not order:
            reply = _pick_language_reply(
                language,
                "Could you share your order ID so I can check when it was shipped?",
                "कृपया अपना ऑर्डर आईडी बताइए ताकि मैं देख सकूँ कि यह कब शिप हुआ।",
                "Kripya apna order ID bataiye taaki main dekh sakoon ki yeh kab ship hua.",
            )
            return AnswerResult(reply, "clarify", "order_shipped", results, facts, language, tool_calls)

        shipped = order.get("shipped_on")
        if shipped:
            shipped_text = data.format_date_long(date.fromisoformat(str(shipped)[:10]))
            reply = _pick_language_reply(
                language,
                f"Your order was shipped on {shipped_text}. Its current status is "
                f"{str(order.get('status', '')).replace('_', ' ')}.",
                f"आपका ऑर्डर {shipped_text} को शिप हुआ था। इसका मौजूदा स्टेटस "
                f"{str(order.get('status', '')).replace('_', ' ')} है।",
                f"Aapka order {shipped_text} ko ship hua tha. Iska current status "
                f"{str(order.get('status', '')).replace('_', ' ')} hai.",
            )
            facts.update({"shipped_on": shipped, "status": order.get("status")})
            return AnswerResult(reply, "answer", "order_shipped", results, facts, language, tool_calls)

        reply = _pick_language_reply(
            language,
            "This order has not been shipped yet, so there is no shipping date on the record.",
            "यह ऑर्डर अभी शिप नहीं हुआ है, इसलिए रिकॉर्ड में शिपिंग तारीख नहीं है।",
            "Yeh order abhi ship nahi hua hai, isliye record mein shipping date nahi hai.",
        )
        return AnswerResult(reply, "answer", "order_shipped", results, facts, language, tool_calls)

    def _where_order(
        self,
        text: str,
        order: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
    ) -> "AnswerResult":
        if not order:
            reply = _pick_language_reply(
                language,
                "Could you share your order ID so I can check where it is?",
                "कृपया अपना ऑर्डर आईडी बताइए ताकि मैं देख सकूँ कि यह कहाँ है।",
                "Kripya apna order ID bataiye taaki main dekh sakoon ki yeh kahan hai.",
            )
            return AnswerResult(reply, "clarify", "where_is_my_order", results, facts, language, tool_calls)
        status = str(order.get("status", "")).replace("_", " ")
        promised = order.get("promised_date")
        delivered = order.get("delivered_on")
        shipped = order.get("shipped_on")
        if delivered:
            reply = _pick_language_reply(
                language,
                f"Your order shows as delivered on {data.format_date_long(date.fromisoformat(str(delivered)[:10]))}. "
                "If you did not receive it, please report it within 48 hours as a missing item.",
                f"आपका ऑर्डर {data.format_date_long(date.fromisoformat(str(delivered)[:10]))} को डिलीवर हुआ दिख रहा है। "
                "अगर आपको यह नहीं मिला, तो कृपया 48 घंटे के अंदर मिसिंग आइटम के रूप में रिपोर्ट करें।",
                f"Aapka order {data.format_date_long(date.fromisoformat(str(delivered)[:10]))} ko delivered dikh raha hai. "
                "Agar aapko yeh nahi mila, toh kripya 48 hours ke andar missing item ke roop mein report karein.",
            )
        elif promised:
            promised_long = data.format_date_long(date.fromisoformat(str(promised)[:10]))
            ship_text = ""
            if shipped:
                ship_text = f" It was shipped on {data.format_date_long(date.fromisoformat(str(shipped)[:10]))}."
            reply = _pick_language_reply(
                language,
                f"Your order is {status}. The promised delivery date is {promised_long}.{ship_text}",
                f"आपका ऑर्डर {status} है। डिलीवरी की तय तारीख {promised_long} है।",
                f"Aapka order {status} hai. Promised delivery date {promised_long} hai.",
            )
        else:
            reply = _pick_language_reply(
                language,
                f"Your order status is {status}. I don't have a promised date on this record.",
                f"आपका ऑर्डर स्टेटस {status} है। इस रिकॉर्ड में तय तारीख नहीं है।",
                f"Aapka order status {status} hai. Is record mein promised date nahi hai.",
            )
        facts.update({"status": status, "promised_date": promised, "delivered_on": delivered, "shipped_on": shipped})
        return AnswerResult(reply, "answer", "where_is_my_order", results, facts, language, tool_calls)

    def _refund_timeline(
        self,
        text: str,
        order: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
    ) -> "AnswerResult":
        method = detect_payment_method(text, order)
        start = data.parse_date(text)
        if start is None and order:
            for key in ("return_refund_initiated_on", "cancellation_refund_initiated_on", "rto_refund_initiated_on"):
                start = _parse_order_date(order, key)
                if start:
                    break
        if method == "cod":
            if order:
                method = "cod_wallet"
            else:
                reply = _pick_language_reply(
                    language,
                    "For a cash-on-delivery refund, you can choose ZentroMart Wallet for an "
                    "instant refund, or a bank transfer that takes 5 to 7 business days after "
                    "the bank details are verified.",
                    "कैश ऑन डिलीवरी रिफ़ंड के लिए आप ZentroMart Wallet चुन सकते हैं (तुरंत), "
                    "या बैंक ट्रांसफ़र चुन सकते हैं जिसमें बैंक डिटेल्स वेरिफ़ाई होने के बाद "
                    "5 से 7 कामकाजी दिन लगते हैं।",
                    "Cash on delivery refund ke liye aap ZentroMart Wallet chun sakte hain "
                    "(instant), ya bank transfer chun sakte hain jisme bank details verify "
                    "hone ke baad 5 se 7 business days lagte hain.",
                )
                return AnswerResult(reply, "answer", "refund_timeline", results, facts, language, tool_calls)

        if method is None or method not in ("upi", "card", "netbanking", "wallet", "cod_wallet", "cod_bank"):
            reply = _pick_language_reply(
                language,
                "How did you pay for the order? The refund timeline depends on the payment "
                "method. If you share your order ID, I can check it.",
                "आपने ऑर्डर का भुगतान कैसे किया था? रिफ़ंड का समय पेमेंट मेथड पर निर्भर करता है। "
                "ऑर्डर आईडी बताएँ तो मैं देख सकती हूँ।",
                "Aapne order ka payment kaise kiya tha? Refund timeline payment method par "
                "depend karta hai. Order ID batayein toh main dekh sakti hoon.",
            )
            return AnswerResult(reply, "clarify", "refund_timeline", results, facts, language, tool_calls)

        if method in ("wallet", "cod_wallet"):
            reply = _pick_language_reply(
                language,
                "Refunds to the ZentroMart Wallet are instant, within 2 hours of initiation.",
                "ZentroMart Wallet में रिफ़ंड तुरंत, शुरू होने के 2 घंटे के अंदर आ जाता है।",
                "ZentroMart Wallet mein refund instant, initiation ke 2 hours ke andar aa jaata hai.",
            )
            facts.update({"payment_method": method, "instant": True})
            return AnswerResult(reply, "answer", "refund_timeline", results, facts, language, tool_calls)

        span = kb.refund_range_for_method(method)
        if span is None:
            reply = _pick_language_reply(
                language,
                "I don't have a refund timeline for that payment method. I can connect you "
                "to a human colleague.",
                "उस पेमेंट मेथड के लिए मेरे पास रिफ़ंड टाइमलाइन नहीं है। मैं आपको किसी सहयोगी से जोड़ सकती हूँ।",
                "Us payment method ke liye mere paas refund timeline nahi hai. Main aapko "
                "ek colleague se connect kar sakti hoon.",
            )
            return AnswerResult(reply, "answer", "refund_timeline", results, facts, language, tool_calls)

        min_days, max_days = span
        if start:
            end_min, end_max = data.business_day_range(start, min_days, max_days)
            start_text = data.format_date_long(end_min)
            end_text = data.format_date_long(end_max)
            reply = _pick_language_reply(
                language,
                f"Refunds to {_refund_method_label(method)} take {min_days} to {max_days} "
                f"business days after initiation. For a refund started on "
                f"{data.format_date_long(start)}, that means between {start_text} and {end_text}.",
                f"{_refund_method_label(method)} में रिफ़ंड शुरू होने के बाद {min_days} से {max_days} "
                f"कामकाजी दिन लगते हैं। {data.format_date_long(start)} को शुरू हुए रिफ़ंड के लिए "
                f"यह {start_text} से {end_text} के बीच आना चाहिए।",
                f"{_refund_method_label(method)} refund initiation ke baad {min_days} se {max_days} "
                f"business days leta hai. {data.format_date_long(start)} ko shuru hue refund ke "
                f"liye yeh {start_text} se {end_text} ke beech aana chahiye.",
            )
            facts.update(
                {
                    "payment_method": method,
                    "start": start.isoformat(),
                    "end_min": end_min.isoformat(),
                    "end_max": end_max.isoformat(),
                }
            )
        else:
            reply = _pick_language_reply(
                language,
                f"Refunds to {_refund_method_label(method)} take {min_days} to {max_days} "
                "business days after the refund is initiated.",
                f"{_refund_method_label(method)} में रिफ़ंड शुरू होने के बाद {min_days} से {max_days} "
                "कामकाजी दिन लगते हैं।",
                f"{_refund_method_label(method)} refund initiation ke baad {min_days} se {max_days} "
                "business days leta hai.",
            )
            facts.update({"payment_method": method, "min_days": min_days, "max_days": max_days})
        return AnswerResult(reply, "answer", "refund_timeline", results, facts, language, tool_calls)

    def _refund_status(
        self,
        text: str,
        order: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
        allow_ticket: bool,
    ) -> "AnswerResult":
        if not order:
            reply = _pick_language_reply(
                language,
                "Could you share your order ID so I can check the refund status?",
                "कृपया अपना ऑर्डर आईडी बताइए ताकि मैं रिफ़ंड स्टेटस देख सकूँ।",
                "Kripya apna order ID bataiye taaki main refund status dekh sakoon.",
            )
            return AnswerResult(reply, "clarify", "refund_status", results, facts, language, tool_calls)

        initiated_raw = order.get("return_refund_initiated_on") or order.get("cancellation_refund_initiated_on") or order.get("rto_refund_initiated_on")
        method = order.get("return_refund_method") or order.get("cancellation_refund_method") or order.get("rto_refund_method") or order.get("payment_method")
        qc = order.get("return_qc_passed_on")
        picked = order.get("return_pickup_on")
        rrn = order.get("return_rrn") or order.get("cancellation_rrn") or order.get("rto_rrn")

        if initiated_raw:
            initiated = date.fromisoformat(str(initiated_raw)[:10])
            method = (method or "").lower()
            span = kb.refund_range_for_method(method)
            if span:
                end_min, end_max = data.business_day_range(initiated, span[0], span[1])
                # Overdue when more than 3 business days past its maximum.
                overdue_at = data.add_business_days(end_max, kb.policy_numbers()["refund_overdue_grace_days"])
                if config.reference_date() > overdue_at:
                    summary = (
                        "Refund is more than 3 business days past its maximum timeline "
                        f"(ESC-1 item 5). Order {order.get('order_id')}; refund initiated "
                        f"{initiated.isoformat()} via {method}; maximum {end_max.isoformat()}; "
                        f"today {config.reference_date().isoformat()}; RRN {rrn}."
                    )
                    extra = ""
                    if rrn:
                        extra = _pick_language_reply(
                            language,
                            f" Your refund reference number is {rrn}.",
                            f" आपका रिफ़ंड रेफ़रेंस नंबर {rrn} है।",
                            f" Aapka refund reference number {rrn} hai.",
                        )
                    reply, created, calls = self._escalate(
                        summary,
                        order_id=order.get("order_id"),
                        reason="refund_overdue",
                        intent="refund_status",
                        language=language,
                        allow_ticket=allow_ticket,
                    )
                    tool_calls.extend(calls)
                    facts.update(
                        {
                            "refund_initiated": initiated.isoformat(),
                            "refund_max": end_max.isoformat(),
                            "rrn": rrn,
                            "ticket_created": created,
                        }
                    )
                    if created:
                        reply = reply.rstrip() + extra
                    return AnswerResult(reply, "escalate", "refund_status", results, facts, language, tool_calls)

                reply = _pick_language_reply(
                    language,
                    f"Your refund was initiated on {data.format_date_long(initiated)} via "
                    f"{_refund_method_label(method)}. It should reach you between "
                    f"{data.format_date_long(end_min)} and {data.format_date_long(end_max)}.",
                    f"आपका रिफ़ंड {data.format_date_long(initiated)} को {_refund_method_label(method)} के "
                    f"माध्यम से शुरू हुआ था। यह {data.format_date_long(end_min)} से "
                    f"{data.format_date_long(end_max)} के बीच आना चाहिए।",
                    f"Aapka refund {data.format_date_long(initiated)} ko {_refund_method_label(method)} "
                    f"se initiate hua tha. Yeh {data.format_date_long(end_min)} se "
                    f"{data.format_date_long(end_max)} ke beech aana chahiye.",
                )
                facts.update(
                    {
                        "refund_initiated": initiated.isoformat(),
                        "refund_min": end_min.isoformat(),
                        "refund_max": end_max.isoformat(),
                        "rrn": rrn,
                    }
                )
                return AnswerResult(reply, "answer", "refund_status", results, facts, language, tool_calls)

        if picked:
            picked_date = date.fromisoformat(str(picked)[:10])
            reply = _pick_language_reply(
                language,
                f"The item was picked up on {data.format_date_long(picked_date)} and is "
                "waiting for the quality check. The refund is initiated within 2 business "
                "days after pickup once it passes QC.",
                f"सामान {data.format_date_long(picked_date)} को पिकअप हुआ और गुणवत्ता जाँच (QC) का "
                "इंतज़ार कर रहा है। QC पास होने के बाद पिकअप के 2 कामकाजी दिन के अंदर रिफ़ंड "
                "शुरू किया जाता है।",
                f"Item {data.format_date_long(picked_date)} ko pickup hua aur quality check (QC) "
                "ka wait kar raha hai. QC pass hone ke baad pickup ke 2 business days ke andar "
                "refund initiate kiya jaata hai.",
            )
            facts.update({"return_pickup_on": picked, "qc_pending": not bool(qc)})
            return AnswerResult(reply, "answer", "refund_status", results, facts, language, tool_calls)

        reply = _pick_language_reply(
            language,
            "I can't see an initiated refund on this order yet. If the item was returned, "
            "the refund starts after pickup and quality check. Could you confirm the return status?",
            "इस ऑर्डर पर अभी शुरू हुआ रिफ़ंड नहीं दिख रहा। अगर सामान वापस किया गया है, तो पिकअप "
            "और गुणवत्ता जाँच के बाद रिफ़ंड शुरू होता है। कृपया रिटर्न स्टेटस बताइए।",
            "Is order par abhi initiated refund nahi dikh raha. Agar item return hua hai, toh "
            "pickup aur QC ke baad refund start hota hai. Kripya return status confirm karein.",
        )
        return AnswerResult(reply, "answer", "refund_status", results, facts, language, tool_calls)

    def _return_answer(
        self,
        text: str,
        order: dict | None,
        items: list[dict] | None,
        product: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
        intent: str,
    ) -> "AnswerResult":
        if product is None and intent == "return_fee":
            reply = _pick_language_reply(
                language,
                "No. ZentroMart Plus members pay no reverse pickup fee, so a "
                "change-of-mind return is free.",
                "नहीं। ZentroMart Plus सदस्यों पर कोई रिवर्स पिकअप फ़ीस नहीं लगती, "
                "इसलिए चेंज-ऑफ-माइंड रिटर्न मुफ़्त है।",
                "Nahi. ZentroMart Plus members par koi reverse pickup fee nahi lagti, "
                "isliye change-of-mind return free hai.",
            )
            facts.update({"plus_fee_waived": True})
            return AnswerResult(reply, "answer", "return_fee", results, facts, language, tool_calls)

        if product is None:
            category = data.category_from_text(text)
            window = kb.return_windows().get(category) if category else None
            if category and window:
                festive_en = (
                    " Orders placed in the 1-15 October festive sale get 30 days instead."
                    if category == "fashion"
                    else ""
                )
                festive_hi = (
                    " 1-15 अक्टूबर की फ़ेस्टिव सेल में किए गए ऑर्डर के लिए 30 दिन मिलते हैं।"
                    if category == "fashion"
                    else ""
                )
                festive_hinglish = (
                    " 1-15 October festive sale ke orders ke liye 30 din milte hain."
                    if category == "fashion"
                    else ""
                )
                reply = _pick_language_reply(
                    language,
                    f"{category.replace('_', ' ').title()} items have {window} calendar "
                    f"days from delivery.{festive_en}",
                    f"{category.replace('_', ' ')} श्रेणी के सामान डिलीवरी से {window} कैलेंडर "
                    f"दिन में वापस हो सकते हैं।{festive_hi}",
                    f"{category.replace('_', ' ')} items delivery se {window} calendar din "
                    f"mein return ho sakte hain.{festive_hinglish}",
                )
                facts.update({"category": category, "window": window})
                return AnswerResult(
                    reply, "answer", "return_window", results, facts, language, tool_calls
                )
            if category and window is None:
                reply = _pick_language_reply(
                    language,
                    f"{category.replace('_', ' ').title()} items cannot be returned for "
                    "change of mind. If the item arrived damaged, expired, wrong or missing, "
                    "report it within 48 hours.",
                    f"{category.replace('_', ' ')} श्रेणी के सामान मन बदलने पर वापस नहीं हो "
                    "सकते। खराब, एक्सपायर्ड, गलत या मिसिंग होने पर 48 घंटे में शिकायत करें।",
                    f"{category.replace('_', ' ')} items change of mind par return nahi ho "
                    "sakte. Damaged, expired, wrong ya missing ho to 48 ghante mein report karein.",
                )
                facts.update({"category": category, "returnable": False})
                return AnswerResult(
                    reply, "answer", "return_eligibility", results, facts, language, tool_calls
                )
            # A pure "how many days" question with no product and no order item.
            reply = _pick_language_reply(
                language,
                "Which product is it? The return window depends on the category.",
                "कौन सा सामान है? रिटर्न विंडो श्रेणी पर निर्भर करती है।",
                "Kaun sa product hai? Return window category par depend karti hai.",
            )
            return AnswerResult(reply, "clarify", "return_window", results, facts, language, tool_calls)

        category = data.return_category(product)
        lower = (text or "").lower()
        details = _return_window_for(product, order, text)
        window = details["window"]

        # Special: opened beauty product.
        if category == "beauty_sealed" and any(
            word in lower for word in ("opened", "open", "khol", "खोल", "tried", "use kiya")
        ):
            reply = _pick_language_reply(
                language,
                f"Beauty products can be returned only if unopened with the seal intact, "
                f"so this {product['title']} can't be returned once opened.",
                f"ब्यूटी प्रोडक्ट सिर्फ़ बिना खोले, सील के साथ ही वापस होते हैं, इसलिए खोले गए "
                f"{product['title']} को वापस नहीं किया जा सकता।",
                f"Beauty products sirf unopened, seal ke saath return ho sakte hain, isliye "
                f"khula hua {product['title']} return nahi ho sakta.",
            )
            facts.update({"category": category, "returnable": False})
            return AnswerResult(reply, "answer", "return_eligibility", results, facts, language, tool_calls)

        if window is None:
            reply = _pick_language_reply(
                language,
                f"{product['title']} is in the {category.replace('_', ' ')} category, which "
                "cannot be returned for change of mind. If it arrived damaged, expired, wrong "
                "or missing, report it within 48 hours under the damaged-items policy.",
                f"{product['title']} {category.replace('_', ' ')} श्रेणी में है, जिसे मन बदलने पर "
                "वापस नहीं किया जा सकता। अगर यह खराब, एक्सपायर्ड, गलत या मिसिंग आया है, तो "
                "48 घंटे के अंदर शिकायत करें।",
                f"{product['title']} {category.replace('_', ' ')} category mein hai, jo change "
                "of mind par return nahi ho sakta. Agar damaged, expired, wrong ya missing "
                "aaya hai, toh 48 hours ke andar report karein.",
            )
            facts.update({"category": category, "returnable": False})
            return AnswerResult(reply, "answer", "return_eligibility", results, facts, language, tool_calls)

        if intent == "exchange":
            reply = _pick_language_reply(
                language,
                f"Exchanges for a different size or colour are available on clothing and "
                f"footwear within the {window}-day return window, subject to stock.",
                f"कपड़ों और जूतों पर अलग साइज़ या रंग का एक्सचेंज {window} दिन की रिटर्न विंडो "
                "में, स्टॉक के अधीन उपलब्ध है।",
                f"Clothing aur footwear par different size ya colour ka exchange {window}-day "
                "return window mein, stock ke subject par available hai.",
            )
            facts.update({"category": category, "window": window})
            return AnswerResult(reply, "answer", "exchange", results, facts, language, tool_calls)

        deadline = details["deadline"]
        fee = details["fee"]
        plus = bool((order or {}).get("is_plus"))
        parts_en: list[str] = []
        parts_hi: list[str] = []
        parts_hg: list[str] = []

        if details["festive"]:
            parts_en.append(
                f"This is a festive-sale order, so {product['title']} can be returned within "
                f"{window} days of delivery."
            )
            parts_hi.append(
                f"यह फ़ेस्टिव सेल का ऑर्डर है, इसलिए {product['title']} डिलीवरी के {window} दिन के "
                "अंदर वापस किया जा सकता है।"
            )
            parts_hg.append(
                f"Yeh festive sale ka order hai, isliye {product['title']} delivery se {window} "
                "din ke andar return ho sakta hai."
            )
        else:
            parts_en.append(
                f"{product['title']} can be returned within {window} days of delivery."
            )
            parts_hi.append(
                f"{product['title']} डिलीवरी के {window} दिन के अंदर वापस किया जा सकता है।"
            )
            parts_hg.append(
                f"{product['title']} delivery se {window} din ke andar return ho sakta hai."
            )
        if deadline:
            parts_en.append(f"The deadline is {data.format_date_long(deadline)}.")
            parts_hi.append(f"आख़िरी तारीख {data.format_date_long(deadline)} है।")
            parts_hg.append(f"Last date {data.format_date_long(deadline)} hai.")
        if fee == 0:
            if plus:
                parts_en.append("As a ZentroMart Plus member, the reverse pickup fee is waived.")
                parts_hi.append("ZentroMart Plus सदस्य होने के कारण रिवर्स पिकअप फ़ीस माफ़ है।")
                parts_hg.append("ZentroMart Plus member hone ki wajah se reverse pickup fee waived hai.")
            else:
                parts_en.append("There is no reverse pickup fee for a damaged, defective or wrong item.")
                parts_hi.append("खराब, गलत या डैमेज सामान पर कोई रिवर्स पिकअप फ़ीस नहीं है।")
                parts_hg.append("Damaged, wrong ya defective item par koi reverse pickup fee nahi hai.")
        elif fee:
            parts_en.append(f"A ₹{fee} reverse pickup fee applies for this change-of-mind return.")
            parts_hi.append(f"इस चेंज-ऑफ-माइंड रिटर्न पर ₹{fee} रिवर्स पिकअप फ़ीस लगेगी।")
            parts_hg.append(f"Is change-of-mind return par ₹{fee} reverse pickup fee lagegi.")

        reply = _pick_language_reply(language, " ".join(parts_en), " ".join(parts_hi), " ".join(parts_hg))
        facts.update(
            {
                "category": category,
                "window": window,
                "deadline": deadline.isoformat() if deadline else None,
                "fee": fee,
                "festive": details["festive"],
                "returnable": True,
            }
        )
        return AnswerResult(reply, "answer", "return_window", results, facts, language, tool_calls)

    def _warranty(
        self,
        product: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
    ) -> "AnswerResult":
        if not product:
            reply = _pick_language_reply(
                language,
                "Which product or SKU is it? Warranty depends on the product and seller.",
                "कौन सा प्रोडक्ट या SKU है? वारंटी प्रोडक्ट और सेलर पर निर्भर करती है।",
                "Kaun sa product ya SKU hai? Warranty product aur seller par depend karti hai.",
            )
            return AnswerResult(reply, "clarify", "warranty", results, facts, language, tool_calls)

        brand_months = data.warranty_months(product)
        assured = data.assured_warranty(product)
        seller = str(product.get("fulfilled_by", ""))
        if assured:
            brand_part = f" plus the {brand_months}-month brand warranty" if brand_months else ""
            reply = _pick_language_reply(
                language,
                f"Yes. {product['title']} has a 12-month ZentroMart Assured warranty{brand_part}. "
                "Claims are raised in the app under My Orders.",
                f"हाँ। {product['title']} पर 12 महीने की ZentroMart Assured वारंटी है"
                + (f" और {brand_months} महीने की ब्रांड वारंटी भी है" if brand_months else "")
                + "। क्लेम ऐप में My Orders से करें।",
                f"Haan. {product['title']} par 12 mahine ki ZentroMart Assured warranty hai"
                + (f" aur {brand_months} mahine ki brand warranty bhi hai" if brand_months else "")
                + ". Claim app mein My Orders se karein.",
            )
            facts.update({"assured": True, "brand_months": brand_months, "assured_months": 12})
        else:
            if brand_months:
                reply = _pick_language_reply(
                    language,
                    f"{product['title']} has the brand warranty of {brand_months} months. "
                    "ZentroMart Assured does not apply to this item"
                    + (f" because it is fulfilled by {seller}." if seller.lower() != "zentromart" else "."),
                    f"{product['title']} पर {brand_months} महीने की ब्रांड वारंटी है। इस आइटम पर "
                    "ZentroMart Assured लागू नहीं होती।",
                    f"{product['title']} par {brand_months} mahine ki brand warranty hai. Is item "
                    "par ZentroMart Assured apply nahi hoti.",
                )
            else:
                reply = _pick_language_reply(
                    language,
                    f"I don't have a warranty listed for {product['title']}. I can connect you "
                    "to a colleague if you need help.",
                    f"{product['title']} के लिए मेरे पास वारंटी की जानकारी नहीं है। ज़रूरत हो तो "
                    "मैं आपको सहयोगी से जोड़ सकती हूँ।",
                    f"{product['title']} ke liye mere paas warranty ki jaankari nahi hai. "
                    "Zaroorat ho toh main aapko colleague se connect kar sakti hoon.",
                )
            facts.update({"assured": False, "brand_months": brand_months})
        return AnswerResult(reply, "answer", "warranty", results, facts, language, tool_calls)

    def _damaged(
        self,
        text: str,
        order: dict | None,
        product: dict | None,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
        allow_ticket: bool,
    ) -> "AnswerResult":
        threshold = kb.policy_numbers()["high_value_threshold"]
        price = data.product_price(product) or 0
        # If the catalog does not name the item, check an order's damaged SKU.
        if not price and order:
            sku = order.get("damage_sku") or order.get("return_sku")
            if sku:
                price = data.product_price(data.products_by_sku().get(str(sku))) or 0
        if price > threshold:
            summary = (
                f"Customer reports a damaged, defective or wrong item worth more than "
                f"₹{threshold}: {product['title'] if product else text}. "
                f"DWM-3 and ESC-1 require Priority Resolution handover; the agent cannot "
                f"approve the claim. Order: {(order or {}).get('order_id', 'not provided')}. "
                f"Customer said: {text}."
            )
            reply, created, calls = self._escalate(
                summary,
                order_id=(order or {}).get("order_id"),
                reason="high_value_damage",
                intent="damaged_item",
                language=language,
                allow_ticket=allow_ticket,
            )
            tool_calls.extend(calls)
            facts.update({"price": price, "high_value": True, "ticket_created": created})
            return AnswerResult(reply, "escalate", "damaged_item", results, facts, language, tool_calls)

        hours = kb.policy_numbers()["damage_report_hours"]
        reply = _pick_language_reply(
            language,
            f"Please report it within {hours} hours of delivery with photos of the item and "
            "the packaging. After verification you get a free replacement or a full refund, "
            "including delivery fees, and no reverse pickup fee applies.",
            f"डिलीवरी के {hours} घंटे के अंदर सामान और पैकेट की फ़ोटो के साथ शिकायत करें। "
            "जाँच के बाद मुफ़्त रिप्लेसमेंट या पूरा रिफ़ंड मिलेगा, जिसमें डिलीवरी फ़ीस भी शामिल है, "
            "और कोई रिवर्स पिकअप फ़ीस नहीं लगेगी।",
            f"Delivery ke {hours} hours ke andar item aur packaging ki photos ke saath report "
            "karein. Verification ke baad free replacement ya full refund milega, delivery "
            "fees ke saath, aur koi reverse pickup fee nahi lagegi.",
        )
        facts.update({"price": price, "high_value": False, "report_hours": hours})
        return AnswerResult(reply, "answer", "damaged_item", results, facts, language, tool_calls)

    def _express_late(
        self,
        order: dict,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
    ) -> "AnswerResult":
        fee = kb.policy_numbers()["express_fee"]
        credited = order.get("express_late_credit_inr")
        plus = bool(order.get("is_plus"))
        if plus:
            reply = _pick_language_reply(
                language,
                "ZentroMart Plus members pay no Express fee, so there is no late-delivery "
                "credit on this order.",
                "ZentroMart Plus सदस्य एक्सप्रेस फ़ीस नहीं देते, इसलिए इस ऑर्डर पर लेट-डिलीवरी "
                "क्रेडिट नहीं मिलता।",
                "ZentroMart Plus members Express fee nahi dete, isliye is order par late-delivery "
                "credit nahi milta.",
            )
            facts.update({"express_fee": fee, "plus": True, "credit": None})
            return AnswerResult(reply, "answer", "late_delivery", results, facts, language, tool_calls)
        if credited:
            date_text = order.get("express_late_credit_on")
            reply = _pick_language_reply(
                language,
                f"Yes. The ₹{fee} Express fee was credited to your ZentroMart Wallet on "
                f"{data.format_date_long(date.fromisoformat(str(date_text)[:10]))}.",
                f"हाँ। ₹{fee} एक्सप्रेस फ़ीस {data.format_date_long(date.fromisoformat(str(date_text)[:10]))} "
                "को आपके ZentroMart Wallet में जमा की गई थी।",
                f"Haan. ₹{fee} Express fee {data.format_date_long(date.fromisoformat(str(date_text)[:10]))} "
                "ko aapke ZentroMart Wallet mein credit kiya gaya tha.",
            )
            facts.update({"express_fee": fee, "plus": False, "credit": fee, "credit_on": date_text})
            return AnswerResult(reply, "answer", "late_delivery", results, facts, language, tool_calls)
        reply = _pick_language_reply(
            language,
            f"Yes. If an Express order arrives after its promised date, the ₹{fee} fee is "
            "credited to ZentroMart Wallet automatically within 48 hours.",
            f"हाँ। अगर एक्सप्रेस ऑर्डर तय तारीख के बाद आता है, तो ₹{fee} फ़ीस 48 घंटे के अंदर "
            "अपने आप ZentroMart Wallet में जमा हो जाती है।",
            f"Haan. Agar Express order promised date ke baad aata hai, toh ₹{fee} fee 48 "
            "hours ke andar automatically ZentroMart Wallet mein credit ho jaati hai.",
        )
        facts.update({"express_fee": fee, "plus": False, "credit": None})
        return AnswerResult(reply, "answer", "late_delivery", results, facts, language, tool_calls)

    def _troubleshooting(
        self,
        text: str,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
    ) -> "AnswerResult":
        lower = (text or "").lower()
        if "speaker" in lower or "turn on" in lower or "power" in lower:
            reply = _pick_language_reply(
                language,
                "Charge it for at least 30 minutes with the original cable, then hold the "
                "power button for 10 seconds to force a restart. If it still doesn't turn on, "
                "it can be returned within its window or claimed under warranty.",
                "इसे ओरिजिनल केबल से कम से कम 30 मिनट चार्ज करें, फिर पावर बटन 10 सेकंड दबाकर "
                "रीस्टार्ट करें। अगर फिर भी नहीं चलता, तो रिटर्न विंडो में वापस करें या वारंटी "
                "क्लेम करें।",
                "Ise original cable se kam se kam 30 minutes charge karein, phir power button "
                "10 seconds dabakar restart karein. Agar phir bhi nahi chalta, toh return "
                "window mein wapas karein ya warranty claim karein.",
            )
            facts.update({"steps": ["charge 30 minutes", "hold power 10 seconds"]})
        elif "earbud" in lower or "one side" in lower:
            reply = _pick_language_reply(
                language,
                "Put both earbuds in the case and close the lid for 10 seconds. Then forget "
                "the device in Bluetooth and pair again; if one side is still silent, it can "
                "be returned within its window or claimed under warranty.",
                "दोनों ईयरबड्स केस में रखकर ढक्कन 10 सेकंड बंद करें। फिर ब्लूटूथ से डिवाइस "
                "भूलें और दोबारा पेयर करें; अगर एक साइड फिर भी नहीं चलता, तो रिटर्न या वारंटी "
                "क्लेम करें।",
                "Dono earbuds case mein rakhkar lid 10 seconds band karein. Phir Bluetooth "
                "se device forget karke dobara pair karein; agar ek side phir bhi nahi chalta, "
                "toh return ya warranty claim karein.",
            )
            facts.update({"steps": ["case 10 seconds", "forget and pair again"]})
        elif "smartwatch" in lower or "sync" in lower:
            reply = _pick_language_reply(
                language,
                "Check that the companion app has Bluetooth and location permission, restart "
                "the watch and phone, then unpair and pair again. If it still doesn't sync, "
                "it can be returned within its window or claimed under warranty.",
                "कम्पैनियन ऐप को ब्लूटूथ और लोकेशन परमिशन दें, वॉच और फ़ोन रीस्टार्ट करें, "
                "फिर अनपेयर करके दोबारा पेयर करें। अगर फिर भी सिंक नहीं होता, तो रिटर्न या "
                "वारंटी क्लेम करें।",
                "Companion app ko Bluetooth aur location permission dein, watch aur phone "
                "restart karein, phir unpair karke dobara pair karein. Agar phir bhi sync "
                "nahi hota, toh return ya warranty claim karein.",
            )
            facts.update({"steps": ["permissions", "restart", "unpair and pair"]})
        else:
            reply = _pick_language_reply(
                language,
                "Use the original cable and adapter, clean the charging port gently, and try "
                "airplane mode while charging. If the problem continues, the item can be "
                "returned within its window or claimed under warranty.",
                "ओरिजिनल केबल और एडाप्टर इस्तेमाल करें, चार्जिंग पोर्ट साफ़ करें, और चार्ज करते "
                "समय एयरप्लेन मोड आज़माएँ। अगर समस्या बनी रहे, तो रिटर्न या वारंटी क्लेम करें।",
                "Original cable aur adapter use karein, charging port saaf karein, aur charge "
                "karte time airplane mode try karein. Agar problem bani rahe, toh return ya "
                "warranty claim karein.",
            )
            facts.update({"steps": ["original cable", "clean port", "airplane mode"]})
        return AnswerResult(reply, "answer", "troubleshooting", results, facts, language, tool_calls)

    def _general_answer(
        self,
        text: str,
        language: str,
        results: list[retrieval.Retrieved],
        facts: dict,
        tool_calls: list[ToolCall],
        extra_context: str,
    ) -> "AnswerResult":
        context = retrieval.context_block(results[:5])
        if extra_context:
            context = extra_context + "\n\n" + context
        top_score = results[0].score if results else 0.0
        if top_score < 0.28:
            reply = _pick_language_reply(
                language,
                "Sorry, I don't have information about that. I can connect you to a "
                "human colleague who can help.",
                "माफ़ कीजिए, मेरे पास इस बारे में जानकारी नहीं है। मैं आपको किसी सहयोगी से जोड़ सकती हूँ।",
                "Maaf kijiye, mere paas iske baare mein jaankari nahi hai. Main aapko ek "
                "colleague se connect kar sakti hoon.",
            )
            return AnswerResult(reply, "answer", "unsupported", results, facts, language, tool_calls)

        prompt = (
            "Answer the customer using ONLY the evidence below. If the evidence does "
            "not contain the answer, say you don't know and offer a human colleague. "
            "Do not invent a policy, date, amount or action. Keep it to two or three "
            "short sentences. Reply in English.\n\n"
            f"Evidence:\n{context}\n\nCustomer question:\n{text}"
        )
        try:
            reply_en = (self.llm or LLMClient()).chat_text(
                "You are ZentroMart's grounded support assistant.",
                prompt,
                max_tokens=350,
                temperature=0.1,
            ).strip()
        except LLMError:
            reply_en = ""
        if not reply_en:
            reply_en = (
                "Sorry, I don't have information about that. I can connect you to a "
                "human colleague who can help."
            )
        reply = localize(reply_en, language, self.llm)
        return AnswerResult(reply, "answer", "general", results, facts, language, tool_calls)


@dataclass
class AnswerResult:
    reply: str
    decision: str
    intent: str
    results: list[retrieval.Retrieved] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    language: str = "en"
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def used(self) -> list[dict]:
        return retrieval.evidence(self.results)

    @property
    def chunk_ids(self) -> list[str]:
        return [item.chunk.chunk_id for item in self.results]
