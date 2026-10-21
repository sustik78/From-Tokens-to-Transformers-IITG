"""Central configuration and dataset location helpers.

The supplied dataset is never copied into this repository.  Set DATA_DIR in
`.env` (or pass --data_dir on the CLI) and the rest of the project resolves
everything from there.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATASET_MARKERS = ("kb", "data", "practice", "tools")


def _looks_like_dataset(path: Path) -> bool:
    return all((path / marker).is_dir() for marker in DATASET_MARKERS)


def resolve_data_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Find the extracted dataset folder.

    Priority: explicit argument -> DATA_DIR env -> auto-discovery under the
    repo root (including the nested folder produced by the supplied ZIP).
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.getenv("DATA_DIR", "").strip()
    if env_value:
        raw = Path(env_value)
        candidates.append(raw if raw.is_absolute() else ROOT / raw)
    candidates.extend(
        [
            ROOT,
            ROOT / "Capstone_Project_Dataset_FTTT_2026",
            ROOT
            / "Capstone_Project_Dataset_FTTT_2026"
            / "Capstone_Project_Dataset_FTTT_2026",
        ]
    )
    # One extra level of auto-discovery, without walking the whole disk.
    for parent in (ROOT, ROOT / "Capstone_Project_Dataset_FTTT_2026"):
        if parent.is_dir():
            for child in parent.iterdir():
                if child.is_dir():
                    candidates.append(child)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_dataset(resolved):
            return resolved

    raise FileNotFoundError(
        "Could not find the dataset. Set DATA_DIR in .env or pass --data_dir "
        "to the folder that contains kb/, data/, practice/ and tools/."
    )


DATA_DIR = resolve_data_dir()


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def load_csv(relative_path: str) -> pd.DataFrame:
    """Read a dataset CSV with every identifier preserved as text."""
    return pd.read_csv(
        data_path(relative_path),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default


def reference_date() -> date:
    raw = env("REFERENCE_DATE", "2026-10-20")
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return date(2026, 10, 20)


LLM_PROVIDER = env("LLM_PROVIDER", "llm7").lower()
LLM_MODEL = env("LLM_MODEL", "DeepSeek-V4-Flash-0731")
LLM_BASE_URL = env("LLM_BASE_URL")
LLM_API_KEY = env("LLM_API_KEY")
LLM_TIMEOUT = env_int("LLM_TIMEOUT", 90)
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.2)
LLM_FALLBACK_PROVIDER = env("LLM_FALLBACK_PROVIDER", "pollinations").lower()
LLM_FALLBACK_MODEL = env("LLM_FALLBACK_MODEL", "openai-fast")

EMBEDDING_MODEL = env(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_BACKEND = env("EMBEDDING_BACKEND", "auto").lower()
INDEX_DIR = Path(env("INDEX_DIR", ".index_cache"))
if not INDEX_DIR.is_absolute():
    INDEX_DIR = ROOT / INDEX_DIR

RETRIEVAL_TOP_K = env_int("RETRIEVAL_TOP_K", 8)

GNANI_STT_LANGUAGE = env("GNANI_STT_LANGUAGE", "en-IN")
GNANI_TTS_MODEL = env("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_TTS_VOICE = env("GNANI_TTS_VOICE", "Kaveri")
GNANI_TTS_LANGUAGE = env("GNANI_TTS_LANGUAGE", "en-IN")


def require_gnani_key() -> None:
    if not os.getenv("GNANI_API_KEY"):
        raise RuntimeError(
            "GNANI_API_KEY is not set. Put it in .env (never commit .env)."
        )
