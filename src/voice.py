"""Gnani-only speech layer for the support agent."""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .agent import AgentResult, SupportAgent

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


@dataclass
class VoiceTurn:
    heard: str
    reply: str
    audio_path: Path
    result: AgentResult
    stt_language: str
    tts_language: str


def speech_text(text: str) -> str:
    """Make text easier to read aloud without changing its meaning."""
    return (
        text.replace("₹", "")
        .replace("Rs.", "rupees")
        .replace("Rs", "rupees")
        .replace("%", " percent")
        .replace("&", " and ")
        .strip()
    )


def _as_wav(path: Path) -> tuple[Path, bool]:
    if path.suffix.lower() == ".wav":
        try:
            import soundfile as sf

            with sf.SoundFile(str(path)) as handle:
                if handle.frames > 0:
                    return path, False
        except Exception:
            pass
    try:
        import librosa
        import soundfile as sf

        samples, sample_rate = librosa.load(str(path), sr=16000, mono=True)
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        handle.close()
        converted = Path(handle.name)
        sf.write(str(converted), samples, sample_rate)
        return converted, True
    except Exception as exc:
        raise RuntimeError(f"Could not read audio file {path.name}: {exc}") from exc


def _repair_wav_header(path: Path) -> bool:
    """Repair streaming WAVs whose RIFF/data sizes are placeholders."""
    try:
        raw = bytearray(path.read_bytes())
        if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            return False
        offset = 12
        while offset + 8 <= len(raw):
            chunk_id = bytes(raw[offset : offset + 4])
            size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
            body = offset + 8
            if chunk_id == b"data":
                actual = len(raw) - body
                if size != actual:
                    raw[offset + 4 : offset + 8] = actual.to_bytes(4, "little")
                riff_size = len(raw) - 8
                if int.from_bytes(raw[4:8], "little") != riff_size:
                    raw[4:8] = riff_size.to_bytes(4, "little")
                path.write_bytes(raw)
                return True
            if size == 0xFFFFFFFF:
                return False
            offset = body + size + (size & 1)
    except Exception:
        return False
    return False


def _retry(call, attempts: int = 3):
    last_error: Exception | None = None
    for delay in (0.0, 2.0, 5.0)[:attempts]:
        if delay:
            time.sleep(delay)
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - SDK errors vary by transport
            last_error = exc
    raise RuntimeError(str(last_error or "Gnani request failed"))


def _stt_language(language: str | None) -> str:
    value = language or config.GNANI_STT_LANGUAGE
    return "en-IN" if value in ("hi-en", "hi") else value


def _tts_language(language: str | None) -> str:
    value = language or config.GNANI_TTS_LANGUAGE
    return "hi-IN" if value in ("hi-en", "hi") else value


def _first_value(payload: object, keys: tuple[str, ...]) -> str:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = _first_value(value, keys)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_value(value, keys)
            if found:
                return found
    return ""


class VoicePipeline:
    def __init__(
        self,
        stt_client=None,
        tts_client=None,
        voice: str | None = None,
        tts_model: str | None = None,
    ):
        self._stt = stt_client
        self._tts = tts_client
        self.voice = voice or os.getenv("GNANI_TTS_VOICE", config.GNANI_TTS_VOICE)
        self.tts_model = tts_model or os.getenv(
            "GNANI_TTS_MODEL", config.GNANI_TTS_MODEL
        )

    def _stt_client(self):
        if self._stt is None:
            config.require_gnani_key()
            from gnani.stt import GnaniSTTClient

            self._stt = GnaniSTTClient()
        return self._stt

    def _tts_client(self):
        if self._tts is None:
            config.require_gnani_key()
            from gnani.tts import GnaniTTSClient

            self._tts = GnaniTTSClient()
        return self._tts

    def transcribe(self, audio_path: Path, language_code: str) -> str:
        wav_path, temporary = _as_wav(Path(audio_path))
        try:
            payload = _retry(
                lambda: self._stt_client().transcribe(
                    str(wav_path), language_code=_stt_language(language_code)
                )
            )
        finally:
            if temporary:
                wav_path.unlink(missing_ok=True)
        return _first_value(payload, ("transcript", "text", "transcription"))

    def speak(self, text: str, out_path: Path, language_code: str) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _retry(
            lambda: self._tts_client().synthesize(
                speech_text(text),
                voice=self.voice,
                model=self.tts_model,
                language=_tts_language(language_code),
                output_file=str(out_path),
            )
        )
        _repair_wav_header(out_path)
        return out_path

    def turn(
        self,
        audio_path: Path,
        agent: SupportAgent,
        out_path: Path,
        language_code: str | None = None,
    ) -> VoiceTurn:
        audio_path = Path(audio_path)
        stt_language = _stt_language(language_code)
        heard = self.transcribe(audio_path, stt_language)

        if not heard.strip():
            reply = "Sorry, I didn't catch that. Could you say it again?"
            result = AgentResult(reply, "clarify", "unclear", "en")
            tts_language = stt_language
        else:
            result = agent.respond(heard)
            reply = result.reply
            tts_language = (
                "hi-IN"
                if result.language in ("hi", "hi-en")
                else "en-IN"
                if result.language == "en"
                else stt_language
            )

        self.speak(reply, out_path, tts_language)
        return VoiceTurn(
            heard=heard,
            reply=reply,
            audio_path=out_path,
            result=result,
            stt_language=stt_language,
            tts_language=tts_language,
        )
