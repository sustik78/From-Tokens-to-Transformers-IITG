"""Create small Gnani TTS clips for local end-to-end voice tests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.voice import VoicePipeline

CLIPS = [
    ("01_where_is_my_order", "Hello, my order has not arrived yet. Where is it?", "en-IN"),
    (
        "02_return_window",
        "What is the return window for a fashion item delivered on 5 October 2026?",
        "en-IN",
    ),
    ("03_refund_timeline", "How long does a UPI refund take?", "en-IN"),
    ("04_cancel_shipped", "My order has already shipped. Can I still cancel it?", "en-IN"),
    ("05_hindi_refund", "मेरा रिफ़ंड कहाँ है? ऑर्डर नंबर है 6649 2475।", "hi-IN"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("test_audio"))
    parser.add_argument("--voice", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = VoicePipeline(voice=args.voice, tts_model=args.model)
    for name, text, language in CLIPS:
        out = args.out_dir / f"{name}.wav"
        try:
            pipeline.speak(text, out, language)
        except Exception as exc:  # noqa: BLE001 - surface provider errors clearly
            print(f"FAILED {name}: {exc}")
            return 1
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
