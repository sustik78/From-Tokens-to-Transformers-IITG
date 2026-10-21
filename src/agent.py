"""Conversation state, identity verification and grounded answer routing."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import data, retrieval
from .answering import GroundedAnswerer, detect_intent, detect_language
from .llm import LLMClient
from .tools_wrapper import ToolCall, lookup_order

ORDER_RECORD_INTENTS = {
    "where_is_my_order",
    "refund_status",
    "order_shipped",
}

CONVERSATION_ORDER_MARKERS = (
    "my order",
    "mera order",
    "mere order",
    "order id",
    "order number",
    "order ke baare",
    "order ke bare",
    "this order",
    "is order",
    "ise",
    "help",
    "madad",
)


@dataclass
class AgentResult:
    reply: str
    decision: str
    intent: str
    language: str = "en"
    order_id: str | None = None
    verified: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[retrieval.Retrieved] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [item.chunk.chunk_id for item in self.results]

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "decision": self.decision,
            "intent": self.intent,
            "language": self.language,
            "order_id": self.order_id,
            "verified": self.verified,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "retrieved": [item.to_dict() for item in self.results],
            "facts": self.facts,
        }


def _pick(language: str, english: str, hindi: str, hinglish: str) -> str:
    if language == "hi":
        return hindi
    if language == "hi-en":
        return hinglish
    return english


def _has_disclosure(text: str) -> bool:
    lower = (text or "").lower()
    return any(
        marker in lower
        for marker in ("ai assistant", "artificial intelligence", "एआई", "कृत्रिम बुद्धिमत्ता")
    )


class SupportAgent:
    """A small state machine for one support conversation.

    Order details are fetched only through the supplied verification-gated
    tools.  The agent never stores verification values in its public result.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        answerer: GroundedAnswerer | None = None,
        standalone: bool = False,
    ):
        self.llm = llm or LLMClient()
        self.answerer = answerer or GroundedAnswerer(self.llm)
        self.standalone = standalone

        self.history: list[dict] = []
        self.order_id: str | None = None
        self.order: dict | None = None
        self.items: list[dict] = []
        self.verified = False
        self.verification_attempts = 0
        self.verification_locked = False
        self.pending_question: str | None = None
        self.pending_intent: str | None = None
        self.first_reply_sent = False
        self.tool_calls: list[ToolCall] = []
        self._turn_start = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.history.clear()
        self.order_id = None
        self.order = None
        self.items = []
        self.verified = False
        self.verification_attempts = 0
        self.verification_locked = False
        self.pending_question = None
        self.pending_intent = None
        self.first_reply_sent = False
        self.tool_calls = []
        self._turn_start = 0

    def respond(
        self,
        text: str,
        language: str | None = None,
        allow_ticket: bool = True,
    ) -> AgentResult:
        original = (text or "").strip()
        language = language or detect_language(original)
        normalized = data.normalize_speech(original)
        self._turn_start = len(self.tool_calls)
        self.history.append({"role": "customer", "text": original})

        if self.verification_locked:
            result = self._locked_result(language)
            self._finish_turn(result)
            return result

        intent = detect_intent(normalized, self.order, bool(self.order))

        # Safety, integrity, privacy and explicit human handover do not need
        # order details and take priority over the normal verification flow.
        if intent in ("policy_override", "fraud", "safety", "human_request"):
            result = self._from_answerer(
                original, language, allow_ticket, force_intent=None
            )
            result = self._maybe_disclose(result, language)
            self._finish_turn(result)
            return result

        if not self.order_id:
            order_id = data.extract_order_id(normalized)
            if order_id:
                return self._start_lookup(
                    order_id=order_id,
                    original=original,
                    normalized=normalized,
                    intent=intent,
                    language=language,
                    allow_ticket=allow_ticket,
                )

            if self._should_ask_for_order(intent, normalized):
                if not self.first_reply_sent and intent not in ("general", "unsupported"):
                    self.pending_question = original
                    self.pending_intent = intent
                reply = _pick(
                    language,
                    "I'm ZentroMart's AI assistant. Please share your order ID so I can check.",
                    "मैं ZentroMart का AI सहायक हूँ। कृपया अपना ऑर्डर आईडी बताइए ताकि मैं देख सकूँ।",
                    "Main ZentroMart ka AI assistant hoon. Kripya apna order ID batayein taaki main check kar sakoon.",
                )
                result = AgentResult(reply, "clarify", intent, language)
                self._finish_turn(result)
                return result

            result = self._from_answerer(original, language, allow_ticket)
            result = self._maybe_disclose(result, language)
            self._finish_turn(result)
            return result

        if not self.verified:
            return self._handle_verification(
                original=original,
                normalized=normalized,
                language=language,
                allow_ticket=allow_ticket,
            )

        return self._answer_with_order(
            original, language, allow_ticket, start_new=True
        )

    # ------------------------------------------------------------------
    # Identity verification
    # ------------------------------------------------------------------
    def _start_lookup(
        self,
        order_id: str,
        original: str,
        normalized: str,
        intent: str,
        language: str,
        allow_ticket: bool,
    ) -> AgentResult:
        self.order_id = order_id
        if (
            not self.pending_question
            and not self.first_reply_sent
            and intent not in ("general", "unsupported")
        ):
            self.pending_question = original
            self.pending_intent = intent

        result, call = lookup_order(order_id)
        self.tool_calls.append(call)

        if result.get("verified"):
            return self._complete_verification(result, language, allow_ticket)

        if not result.get("verification_required"):
            reply = _pick(
                language,
                "I couldn't find an order with that ID. Could you check the number and share it again?",
                "उस आईडी से कोई ऑर्डर नहीं मिला। कृपया नंबर जाँचकर दोबारा बताइए।",
                "Us ID se koi order nahi mila. Kripya number check karke dobara batayein.",
            )
            result_obj = AgentResult(reply, "clarify", intent, language, order_id)
            self.order_id = None
            self._finish_turn(result_obj)
            return result_obj

        value = data.extract_verification_value(normalized)
        if value:
            return self._verify(
                value=value,
                original=original,
                language=language,
                allow_ticket=allow_ticket,
            )

        reply = _pick(
            language,
            "Thanks. To protect your privacy, please share the last four digits of "
            "your registered phone number or your delivery pincode.",
            "धन्यवाद। आपकी गोपनीयता के लिए, कृपया अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी चार अंक "
            "या अपना डिलीवरी पिनकोड बताइए।",
            "Thanks. Privacy ke liye, kripya apne registered phone number ke last four digits "
            "ya apna delivery pincode batayein.",
        )
        result_obj = AgentResult(reply, "clarify", "verification", language, order_id)
        self._finish_turn(result_obj)
        return result_obj

    def _handle_verification(
        self,
        original: str,
        normalized: str,
        language: str,
        allow_ticket: bool,
    ) -> AgentResult:
        value = data.extract_verification_value(normalized)
        if value:
            return self._verify(
                value=value,
                original=original,
                language=language,
                allow_ticket=allow_ticket,
            )

        reply = _pick(
            language,
            "To protect your privacy, please share the last four digits of your "
            "registered phone number or your delivery pincode.",
            "आपकी गोपनीयता के लिए, कृपया अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी चार अंक या "
            "अपना डिलीवरी पिनकोड बताइए।",
            "Privacy ke liye, kripya apne registered phone number ke last four digits ya "
            "apna delivery pincode batayein.",
        )
        result = AgentResult(reply, "clarify", "verification", language, self.order_id)
        self._finish_turn(result)
        return result

    def _verify(
        self,
        value: str,
        original: str,
        language: str,
        allow_ticket: bool,
    ) -> AgentResult:
        assert self.order_id
        result, call = lookup_order(self.order_id, verification_value=value)
        self.tool_calls.append(call)
        self.verification_attempts += 1

        if result.get("verified"):
            return self._complete_verification(result, language, allow_ticket)

        if self.verification_attempts < 2:
            reply = _pick(
                language,
                "That didn't match. Please try once more with the last four digits "
                "of your registered phone number or your delivery pincode.",
                "यह मेल नहीं खाया। कृपया एक बार और अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी चार "
                "अंक या अपना डिलीवरी पिनकोड बताइए।",
                "Yeh match nahi hua. Kripya ek baar aur apne registered phone number ke "
                "last four digits ya apna delivery pincode batayein.",
            )
            result_obj = AgentResult(
                reply, "clarify", "verification", language, self.order_id
            )
            self._finish_turn(result_obj)
            return result_obj

        self.verification_locked = True
        reply = _pick(
            language,
            "I'm sorry, those details don't match, so I can't share order information. "
            "Please contact support from your registered phone number.",
            "माफ़ कीजिए, ये जानकारी मेल नहीं खाई, इसलिए मैं ऑर्डर की जानकारी साझा नहीं कर सकती। "
            "कृपया अपने रजिस्टर्ड फ़ोन नंबर से सहायता से संपर्क करें।",
            "Maaf kijiye, yeh details match nahi hui, isliye main order ki information share "
            "nahi kar sakti. Kripya apne registered phone number se support se contact karein.",
        )
        result_obj = AgentResult(reply, "refuse", "verification", language, self.order_id)
        self._finish_turn(result_obj)
        return result_obj

    def _complete_verification(
        self, result: dict, language: str, allow_ticket: bool
    ) -> AgentResult:
        assert self.order_id
        self._set_order(result)

        pending = self.pending_question
        pending_intent = self.pending_intent or (
            detect_intent(pending or "", self.order, bool(self.order)) if pending else None
        )
        if pending and pending_intent not in (None, "general", "unsupported", "verification"):
            answer = self._from_answerer(pending, language, allow_ticket)
            prefix = _pick(
                language,
                "Thanks, your order is verified.",
                "धन्यवाद, आपका ऑर्डर सत्यापित हो गया है।",
                "Thanks, aapka order verify ho gaya hai.",
            )
            answer.reply = f"{prefix} {answer.reply}"
            self._finish_turn(answer)
            return answer

        reply = _pick(
            language,
            "Thanks, your order is verified. How can I help with it?",
            "धन्यवाद, आपका ऑर्डर सत्यापित हो गया है। मैं इसमें कैसे मदद कर सकती हूँ?",
            "Thanks, aapka order verify ho gaya hai. Main isme kaise help kar sakti hoon?",
        )
        result_obj = AgentResult(
            reply, "answer", "verified", language, self.order_id, True
        )
        self._finish_turn(result_obj)
        return result_obj

    def _set_order(self, result: dict) -> None:
        self.order = result
        self.order_id = str(result.get("order_id") or self.order_id)
        self.items = list(result.get("items") or [])
        self.verified = True

    # ------------------------------------------------------------------
    # Normal answers
    # ------------------------------------------------------------------
    def _answer_with_order(
        self,
        text: str,
        language: str,
        allow_ticket: bool,
        start_new: bool,
    ) -> AgentResult:
        result = self._from_answerer(text, language, allow_ticket)
        result.verified = True
        result.order_id = self.order_id
        self._finish_turn(result)
        return result

    def _from_answerer(
        self,
        text: str,
        language: str,
        allow_ticket: bool,
        force_intent: str | None = None,
    ) -> AgentResult:
        answer = self.answerer.answer(
            text,
            order=self.order,
            items=self.items,
            language=language,
            allow_ticket=allow_ticket,
        )
        self.tool_calls.extend(answer.tool_calls)
        return AgentResult(
            reply=answer.reply,
            decision=answer.decision,
            intent=force_intent or answer.intent,
            language=answer.language,
            order_id=self.order_id,
            verified=self.verified,
            tool_calls=list(self.tool_calls[self._turn_start :]),
            results=list(answer.results),
            facts=dict(answer.facts),
        )

    def _should_ask_for_order(self, intent: str, normalized: str) -> bool:
        if self.standalone:
            return False
        if intent in ORDER_RECORD_INTENTS:
            return True
        if intent.startswith("cancel") and any(
            marker in normalized for marker in ("my order", "mera order", "ise", " it")
        ):
            return True
        if intent == "general":
            return True
        if any(marker in normalized for marker in CONVERSATION_ORDER_MARKERS):
            return intent not in (
                "return_window",
                "return_eligibility",
                "return_fee",
                "refund_timeline",
                "warranty",
                "delivery_attempts",
                "delivery_area",
                "coupon",
                "plus_membership",
                "troubleshooting",
                "unsupported",
            )
        return False

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _maybe_disclose(self, result: AgentResult, language: str) -> AgentResult:
        if self.first_reply_sent or _has_disclosure(result.reply):
            return result
        disclosure = _pick(
            language,
            "I'm ZentroMart's AI assistant. ",
            "मैं ZentroMart का AI सहायक हूँ। ",
            "Main ZentroMart ka AI assistant hoon. ",
        )
        result.reply = disclosure + result.reply
        return result

    def _locked_result(self, language: str) -> AgentResult:
        reply = _pick(
            language,
            "I can't share order details because verification did not pass. "
            "Please contact support from your registered phone number.",
            "सत्यापन नहीं होने के कारण मैं ऑर्डर की जानकारी साझा नहीं कर सकती। "
            "कृपया अपने रजिस्टर्ड फ़ोन नंबर से सहायता से संपर्क करें।",
            "Verification pass nahi hui, isliye main order details share nahi kar sakti. "
            "Kripya apne registered phone number se support se contact karein.",
        )
        return AgentResult(reply, "refuse", "verification", language, self.order_id)

    def _finish_turn(self, result: AgentResult) -> None:
        result.tool_calls = list(self.tool_calls[self._turn_start :])
        self.first_reply_sent = True
        self.history.append({"role": "agent", "text": result.reply})
