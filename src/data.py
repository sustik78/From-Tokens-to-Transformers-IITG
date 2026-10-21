"""Dataset-backed date math, catalog lookup and small helpers."""
from __future__ import annotations

import re
from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

import pandas as pd

from . import config

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
    "जनवरी": 1,
    "फ़रवरी": 2,
    "फरवरी": 2,
    "मार्च": 3,
    "अप्रैल": 4,
    "मई": 5,
    "जून": 6,
    "जुलाई": 7,
    "अगस्त": 8,
    "सितंबर": 9,
    "अक्टूबर": 10,
    "नवंबर": 11,
    "नवम्बर": 11,
    "दिसंबर": 12,
}

MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

SPEECH_NUMBERS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

# "to" is a very common STT rendering of "two".  It is only treated as a
# digit inside a run such as "to zero to six" or "to five six".
_SPEECH_WORD = "|".join(sorted(SPEECH_NUMBERS, key=len, reverse=True))
_SPEECH_RUN = re.compile(
    rf"(?<![a-z])((?:{_SPEECH_WORD}|to)(?:[\s-]+(?:{_SPEECH_WORD}|to))*(?![a-z]))",
    re.IGNORECASE,
)


def _speech_run_to_digits(run: str) -> str:
    words = [word for word in re.split(r"[\s-]+", run.lower()) if word]
    values: list[int] = []
    had_digit = False
    for word in words:
        if word == "to":
            values.append(2)
            continue
        value = SPEECH_NUMBERS.get(word)
        if value is None:
            continue
        values.append(value)
        had_digit = True
    # A single small word is left alone so product names such as "one-piece"
    # and phrases like "one coupon" survive unchanged.
    if not had_digit:
        return run
    if len(values) == 1 and values[0] < 10:
        return run
    if any(value >= 10 for value in values):
        return " ".join(str(value) for value in values)
    return "".join(str(value) for value in values)


def normalize_speech(text: str) -> str:
    """Normalise the word-digit forms produced by speech-to-text.

    Examples: ``ele-zero zero one to`` -> ``ele 0012``,
    ``october to zero to six`` -> ``october 2026``,
    ``to five six gb`` -> ``256 gb``.
    """
    clean = normalize_digits(str(text)).lower()
    clean = re.sub(r"\bre[\s-]+fund\b", "refund", clean)
    clean = re.sub(r"\bu\s+p\s+i\b", "upi", clean)
    clean = re.sub(r"\bc\s+o\s+d\b", "cod", clean)
    clean = _SPEECH_RUN.sub(lambda match: _speech_run_to_digits(match.group(1)), clean)
    return re.sub(r"\s+", " ", clean).strip()


def normalize_digits(text: str) -> str:
    return str(text).translate(_DEVANAGARI_DIGITS)


@lru_cache(maxsize=1)
def holidays() -> frozenset[date]:
    frame = config.load_csv("data/holidays.csv")
    out: set[date] = set()
    for raw in frame.get("date", []):
        try:
            out.add(date.fromisoformat(str(raw)))
        except ValueError:
            continue
    return frozenset(out)


def is_business_day(day: date) -> bool:
    return day.weekday() != 6 and day not in holidays()  # Monday=0, Sunday=6


def add_business_days(start: date, days: int) -> date:
    """Add N business days, not counting the start day."""
    if days < 0:
        raise ValueError("days must be non-negative")
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if is_business_day(current):
            added += 1
    return current


def business_day_range(start: date, min_days: int, max_days: int) -> tuple[date, date]:
    return add_business_days(start, min_days), add_business_days(start, max_days)


def add_calendar_days(start: date, days: int) -> date:
    return start + timedelta(days=days)


def parse_date(text: str, default_year: int | None = None) -> date | None:
    """Parse the date styles used by the practice set and support chats."""
    if not text:
        return None
    clean = normalize_digits(text)
    default_year = default_year or config.reference_date().year

    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", clean)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None

    dmy = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", clean)
    if dmy:
        try:
            return date(int(dmy.group(3)), int(dmy.group(2)), int(dmy.group(1)))
        except ValueError:
            return None

    named = re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"([A-Za-z\u0900-\u097F]+)\.?"
        r"(?:\s+(\d{4}))?\b",
        clean,
        flags=re.IGNORECASE,
    )
    for match in named:
        month = MONTHS.get(match.group(2).lower())
        if not month:
            continue
        try:
            return date(
                int(match.group(3) or default_year), month, int(match.group(1))
            )
        except ValueError:
            continue

    month_first = re.finditer(
        r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b",
        clean,
        flags=re.IGNORECASE,
    )
    for match in month_first:
        month = MONTHS.get(match.group(1).lower())
        if not month:
            continue
        try:
            return date(
                int(match.group(3) or default_year),
                month,
                int(match.group(2)),
            )
        except ValueError:
            continue
    return None


def format_date_long(day: date) -> str:
    return f"{day.day} {MONTH_NAMES[day.month]} {day.year}"


def date_variants(day: date) -> list[str]:
    return [
        day.isoformat(),
        format_date_long(day),
        f"{day.day} {MONTH_NAMES[day.month]}",
        f"{day.day:02d}/{day.month:02d}/{day.year}",
        f"{day.day:02d}-{day.month:02d}-{day.year}",
    ]


@lru_cache(maxsize=1)
def products() -> pd.DataFrame:
    return config.load_csv("data/products.csv")


@lru_cache(maxsize=1)
def products_by_sku() -> dict[str, dict]:
    frame = products()
    return {str(row["sku"]): row.to_dict() for _, row in frame.iterrows()}


@lru_cache(maxsize=1)
def order_items() -> pd.DataFrame:
    return config.load_csv("data/order_items.csv")


@lru_cache(maxsize=1)
def customers_by_id() -> dict[str, dict]:
    frame = config.load_csv("data/customers.csv")
    return {str(row["customer_id"]): row.to_dict() for _, row in frame.iterrows()}


@lru_cache(maxsize=1)
def pincodes_by_code() -> dict[str, dict]:
    frame = config.load_csv("data/pincodes.csv")
    return {str(row["pincode"]): row.to_dict() for _, row in frame.iterrows()}


def items_for_order(order_id: str) -> list[dict]:
    frame = order_items()
    rows = frame[frame["order_id"].astype(str) == str(order_id)]
    return [row.to_dict() for _, row in rows.iterrows()]


def product_for_text(text: str) -> dict | None:
    """Find the product named in a customer message.

    SKU wins.  Otherwise we require a strong token overlap so generic words
    ("speaker", "jeans") do not accidentally select a catalog row.
    """
    if not text:
        return None
    clean = normalize_speech(text)
    sku_match = re.search(
        r"\b([a-z]{2,4})[\s-]+(\d{4})\b", clean, flags=re.IGNORECASE
    )
    if sku_match:
        found = products_by_sku().get(
            f"{sku_match.group(1).upper()}-{sku_match.group(2)}"
        )
        if found:
            return found

    lower = clean.lower()
    frame = products()
    exact = frame[frame["title"].str.lower().apply(lambda title: title in lower)]
    if len(exact):
        return exact.iloc[0].to_dict()

    stop = {
        "the",
        "and",
        "with",
        "for",
        "pack",
        "of",
        "a",
        "an",
        "in",
        "on",
        "to",
        "my",
        "is",
        "it",
        "this",
        "that",
    }
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", lower)
        if token not in stop and len(token) > 1
    }
    best: tuple[float, dict | None] = (0.0, None)
    for _, row in frame.iterrows():
        title = str(row["title"]).lower()
        title_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", title)
            if token not in stop and len(token) > 1
        }
        if not title_tokens:
            continue
        overlap = len(query_tokens & title_tokens)
        coverage = overlap / len(title_tokens)
        if coverage >= 0.6 and overlap >= 2 and coverage > best[0]:
            best = (coverage, row.to_dict())
    return best[1]


def product_price(product: dict | None) -> int | None:
    if not product:
        return None
    try:
        return int(float(product.get("price_inr") or 0))
    except (TypeError, ValueError):
        return None


def assured_warranty(product: dict | None) -> bool:
    if not product:
        return False
    assured = str(product.get("assured", "")).strip().lower() == "true"
    seller = str(product.get("fulfilled_by", "")).strip().lower()
    return assured and seller == "zentromart"


def warranty_months(product: dict | None) -> int:
    if not product:
        return 0
    try:
        return int(float(product.get("warranty_months") or 0))
    except (TypeError, ValueError):
        return 0


def return_category(product: dict | None) -> str:
    return str((product or {}).get("return_category", "")).strip().lower()


CATEGORY_MARKERS = {
    "innerwear": (
        "innerwear",
        "lingerie",
        "briefs",
        "swimwear",
        "swimsuit",
        "socks",
    ),
    "beauty_sealed": (
        "beauty",
        "cosmetic",
        "makeup",
        "lipstick",
        "serum",
        "skincare",
        "cream",
    ),
    "grocery": ("grocery", "groceries", "fresh produce", "food", "snack"),
    "digital": ("digital", "gift card", "subscription", "e-voucher"),
    "books_media": ("book", "books", "novel", "media", "notebook"),
    "electronics": (
        "electronic",
        "laptop",
        "phone",
        "smartphone",
        "earbud",
        "headphone",
        "television",
        "smart tv",
        "camera",
        "smartwatch",
        "charger",
        "tablet",
    ),
    "home_kitchen": (
        "home",
        "kitchen",
        "cookware",
        "mixer",
        "pillow",
        "bedsheet",
        "towel",
        "furniture",
        "mattress",
    ),
    "fashion": (
        "fashion",
        "clothing",
        "apparel",
        "kurta",
        "jeans",
        "sneaker",
        "footwear",
        "shoe",
        "dress",
        "shirt",
        "saree",
        "trouser",
        "jacket",
        "bag",
    ),
}


def category_from_text(text: str) -> str | None:
    lower = (text or "").lower()
    for category, markers in CATEGORY_MARKERS.items():
        if any(marker in lower for marker in markers):
            return category
    return None


_NUMBER_WORDS = {
    "zero": 0,
    "oh": 0,
    "shunya": 0,
    "one": 1,
    "ek": 1,
    "two": 2,
    "to": 2,
    "do": 2,
    "three": 3,
    "teen": 3,
    "tin": 3,
    "four": 4,
    "chaar": 4,
    "char": 4,
    "five": 5,
    "paanch": 5,
    "panch": 5,
    "six": 6,
    "chhah": 6,
    "chhe": 6,
    "chah": 6,
    "che": 6,
    "seven": 7,
    "saat": 7,
    "sat": 7,
    "eight": 8,
    "aath": 8,
    "ath": 8,
    "nine": 9,
    "nau": 9,
    "no": 9,
    "ten": 10,
    "das": 10,
    "eleven": 11,
    "gyarah": 11,
    "gyaarah": 11,
    "twelve": 12,
    "barah": 12,
    "baarah": 12,
    "thirteen": 13,
    "terah": 13,
    "tehrah": 13,
    "fourteen": 14,
    "chaudah": 14,
    "chauda": 14,
    "fifteen": 15,
    "pandrah": 15,
    "pandara": 15,
    "sixteen": 16,
    "solah": 16,
    "sola": 16,
    "seventeen": 17,
    "satrah": 17,
    "satara": 17,
    "eighteen": 18,
    "atharah": 18,
    "athara": 18,
    "nineteen": 19,
    "unnis": 19,
    "unnees": 19,
    "twenty": 20,
    "bees": 20,
    "bis": 20,
    "thirty": 30,
    "tees": 30,
    "forty": 40,
    "chalees": 40,
    "fifty": 50,
    "pachaas": 50,
    "sixty": 60,
    "saath": 60,
    "seventy": 70,
    "sattar": 70,
    "eighty": 80,
    "assi": 80,
    "ninety": 90,
    "nabbe": 90,
    "hundred": 100,
    "sau": 100,
    "thousand": 1000,
    "hazaar": 1000,
    "hazar": 1000,
    "haar": 1000,
    "ikees": 21,
    "ikkees": 21,
    "baees": 22,
    "bais": 22,
    "teis": 23,
    "chaubis": 24,
    "pachees": 25,
    "chhabbis": 26,
    "sattais": 27,
    "atthais": 28,
    "untees": 29,
    "ikattees": 31,
    "battees": 32,
    "taintees": 33,
    "chauntees": 34,
    "paintees": 35,
    "chhattees": 36,
    "saintis": 37,
    "aartees": 38,
    "unchaalees": 39,
    "iktaalees": 41,
    "bayalees": 42,
    "taintalees": 43,
    "chavalees": 44,
    "paintaalees": 45,
    "chhiyaalees": 46,
    "saintalees": 47,
    "artaalees": 48,
    "unchaas": 49,
    "unchas": 49,
    "ikyaavan": 51,
    "baavan": 52,
    "tirpan": 53,
    "chauvan": 54,
    "pachpan": 55,
    "chhappan": 56,
    "sattavan": 57,
    "atthavan": 58,
    "unsath": 59,
    "iksath": 61,
    "baasath": 62,
    "tirsath": 63,
    "chausath": 64,
    "painsath": 65,
    "chhiyasath": 66,
    "sarsath": 67,
    "arsath": 68,
    "unhattar": 69,
    "ikhattar": 71,
    "bahattar": 72,
    "tihattar": 73,
    "chauhattar": 74,
    "pachahattar": 75,
    "pachattar": 75,
    "chhihattar": 76,
    "satahattar": 77,
    "athahattar": 78,
    "unyasi": 79,
    "ikyasi": 81,
    "bayasi": 82,
    "tirasi": 83,
    "chaurasi": 84,
    "pachasi": 85,
    "chhiyasi": 86,
    "satasi": 87,
    "athasi": 88,
    "navasi": 89,
    "ikyaanve": 91,
    "baanve": 92,
    "tiranve": 93,
    "chauranve": 94,
    "pachaanve": 95,
    "chhiyaanve": 96,
    "sattaanve": 97,
    "atthaanve": 98,
    "ninyaanve": 99,
}


def _words_to_digits(segment: str) -> str:
    values = [
        _NUMBER_WORDS[word]
        for word in re.findall(r"[a-z]+", segment.lower())
        if word in _NUMBER_WORDS
    ]
    if not values:
        return ""
    if all(value < 10 for value in values):
        return "".join(str(value) for value in values)
    if not any(value in (100, 1000) for value in values):
        total = 0
        for value in values:
            total = total * 100 + value if value >= 10 else total + value
        return str(total)

    groups: list[str] = []
    total = 0
    current = 0
    for index, value in enumerate(values):
        previous = values[index - 1] if index else None
        next_value = values[index + 1] if index + 1 < len(values) else None
        if value < 100:
            current += value
        elif value == 100:
            current = max(1, current) * 100
        else:
            if (
                total
                and previous is not None
                and previous < 10
                and current > previous
            ):
                current -= previous
                groups.append(str(total + current))
                total = 0
                current = previous
            total += max(1, current) * value
            current = 0
        if next_value is None:
            total += current
            if total:
                groups.append(str(total))
    return "".join(groups)


def extract_order_id(text: str) -> str | None:
    """Extract an 8-digit order ID, including the spaced 4+4 spoken form."""
    clean = normalize_speech(text)
    for match in re.finditer(r"\b(\d{4})\s+(\d{4})\b", clean):
        return match.group(1) + match.group(2)
    for match in re.finditer(r"(?<!\d)(\d{8})(?!\d)", clean):
        return match.group(1)
    # Speech-to-text sometimes inserts spaces between every digit pair.
    groups = re.findall(r"\d+", clean)
    if groups and sum(len(group) for group in groups) == 8:
        joined = "".join(groups)
        if len(joined) == 8:
            return joined
    marked = re.search(
        r"order\s*(?:number|id|no\.?)?\s*(?:is|hai|है)?\s*(.+)$",
        str(text).lower(),
    )
    if marked:
        spoken = _words_to_digits(marked.group(1))
        if len(spoken) >= 8:
            return spoken[:8]
    return None


def extract_verification_value(text: str) -> str | None:
    """Return a 4-digit phone suffix or 6-digit pincode from a spoken reply."""
    clean = normalize_speech(text)
    # Never mistake the order ID itself for a verification value.
    clean = re.sub(r"\b\d{4}\s+\d{4}\b", " ", clean)
    clean = re.sub(r"(?<!\d)\d{8}(?!\d)", " ", clean)
    for pattern in (r"(?<!\d)(\d{4})(?!\d)", r"(?<!\d)(\d{6})(?!\d)"):
        for match in re.finditer(pattern, clean):
            return match.group(1)
    return None


def extract_digits(text: str) -> str:
    return "".join(ch for ch in normalize_speech(text) if ch.isdigit())


def contains_any(text: str, words: Iterable[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)
