#!/usr/bin/env python
"""Batch voice runner: audio folder in, text + speech + sources out."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.agent import SupportAgent
from src.llm import LLMClient
from src.voice import SUPPORTED_AUDIO, VoicePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--language", default=None, help="Gnani STT language, e.g. en-IN")
    parser.add_argument("--voice", default=None, help="Gnani TTS voice, e.g. Kaveri")
    parser.add_argument("--provider", default="none", help="LLM provider; none for offline")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config.DATA_DIR = config.resolve_data_dir(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in args.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No supported audio files found in {args.audio_dir}")
        return 1

    llm = LLMClient(provider=args.provider)
    llm.fallback = "none"
    voice = VoicePipeline(voice=args.voice)
    manifest: dict = {
        "audio_dir": str(args.audio_dir),
        "out_dir": str(args.out_dir),
        "stt_language": args.language or config.GNANI_STT_LANGUAGE,
        "files": [],
        "errors": [],
    }

    for path in files:
        stem = path.stem
        audio_out = args.out_dir / f"{stem}_reply.wav"
        try:
            agent = SupportAgent(llm=llm, standalone=True)
            turn = voice.turn(
                path,
                agent,
                audio_out,
                language_code=args.language,
            )
            payload = {
                "input_audio": str(path),
                "heard": turn.heard,
                "reply": turn.reply,
                "reply_audio": str(audio_out),
                "stt_language": turn.stt_language,
                "tts_language": turn.tts_language,
                **turn.result.to_dict(),
            }
            (args.out_dir / f"{stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (args.out_dir / f"{stem}.txt").write_text(turn.reply + "\n", encoding="utf-8")
            (args.out_dir / f"{stem}.error.json").unlink(missing_ok=True)
            manifest["files"].append(
                {
                    "input": str(path),
                    "heard": turn.heard,
                    "reply": turn.reply,
                    "json": str(args.out_dir / f"{stem}.json"),
                    "audio": str(audio_out),
                    "chunks": turn.result.chunk_ids,
                }
            )
            print(f"ok  {path.name} -> {audio_out.name}")
        except Exception as exc:  # noqa: BLE001 - keep processing other files
            message = f"{path.name}: {exc}"
            manifest["errors"].append(message)
            (args.out_dir / f"{stem}.error.json").write_text(
                json.dumps({"input_audio": str(path), "error": str(exc)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"err {message}")

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"processed {len(manifest['files'])}/{len(files)} files; "
        f"{len(manifest['errors'])} errors"
    )
    return 0 if len(manifest["files"]) == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
