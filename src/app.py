"""Streamlit voice and text demo for the ZentroMart support agent."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import config
from src.agent import SupportAgent
from src.llm import LLMClient
from src.voice import VoicePipeline

st.set_page_config(page_title="ZentroMart VoiceOps", layout="wide")


def make_agent() -> SupportAgent:
    llm = LLMClient(provider="none")
    llm.fallback = "none"
    return SupportAgent(llm=llm)


def reset_agent() -> None:
    st.session_state.agent = make_agent()
    st.session_state.turns = []


if "agent" not in st.session_state:
    reset_agent()
if "turns" not in st.session_state:
    st.session_state.turns = []


@st.cache_resource
def get_voice() -> VoicePipeline:
    return VoicePipeline()


def show_sources(result) -> None:
    if not result.results:
        return
    with st.expander(f"Sources used ({len(result.results)} chunks)"):
        for item in result.results:
            chunk = item.chunk
            st.markdown(
                f"**{chunk.chunk_id}** · `{chunk.source}` · score {item.score:.3f}  \n"
                f"{chunk.section_title}"
            )


def show_tools(result) -> None:
    if not result.tool_calls:
        return
    with st.expander(f"Tool calls ({len(result.tool_calls)})"):
        for call in result.tool_calls:
            st.json(call.to_dict())


st.title("ZentroMart · VoiceOps 2026")
st.caption("Grounded support with Gnani speech in and out. The assistant identifies itself as AI.")

with st.sidebar:
    st.subheader("Call settings")
    language = st.selectbox(
        "Speech language",
        ["en-IN", "hi-IN", "hi-en", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"],
        index=0,
    )
    st.caption(f"TTS voice: {config.GNANI_TTS_VOICE} · model {config.GNANI_TTS_MODEL}")
    if st.button("New conversation", use_container_width=True):
        reset_agent()
        st.rerun()
    agent: SupportAgent = st.session_state.agent
    if agent.order_id:
        st.write(f"Order: `{agent.order_id}`")
        st.write(f"Verified: {'yes' if agent.verified else 'no'}")

voice_tab, text_tab = st.tabs(["Voice call", "Text chat"])

with voice_tab:
    st.write("Record a question or upload a WAV/MP3 clip. Each turn is transcribed by Gnani, answered, and spoken back.")
    recording = None
    if hasattr(st, "audio_input"):
        recording = st.audio_input("Record your question")
    upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
    audio_source = recording or upload
    if audio_source is not None and st.button("Send voice turn", type="primary"):
        suffix = Path(getattr(audio_source, "name", "turn.wav")).suffix or ".wav"
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        handle.write(audio_source.getvalue())
        handle.close()
        out_audio = Path("sample_outputs") / "ui_reply.wav"
        try:
            with st.spinner("Listening, thinking and speaking..."):
                turn = get_voice().turn(
                    Path(handle.name),
                    st.session_state.agent,
                    out_audio,
                    language_code=language,
                )
            st.session_state.turns.append(
                {
                    "heard": turn.heard,
                    "reply": turn.reply,
                    "voice": True,
                    "result": turn.result,
                    "audio": str(out_audio),
                }
            )
        except Exception as exc:  # noqa: BLE001 - show setup errors in the UI
            st.error(f"Voice turn failed: {exc}")

    for turn in reversed(st.session_state.turns):
        if turn.get("voice"):
            st.markdown(f"**You said:** {turn['heard']}")
            st.markdown(f"**Agent:** {turn['reply']}")
            audio_path = Path(turn.get("audio", ""))
            if audio_path.exists():
                st.audio(str(audio_path))
            show_sources(turn["result"])
            show_tools(turn["result"])

with text_tab:
    st.write("Type to test the same conversation state and retrieval without speech.")
    for turn in st.session_state.turns:
        if not turn.get("voice"):
            st.chat_message("user").write(turn["heard"])
            st.chat_message("assistant").write(turn["reply"])
            with st.chat_message("assistant"):
                show_sources(turn["result"])
                show_tools(turn["result"])
    prompt = st.chat_input("Ask about an order, return, refund, delivery or warranty")
    if prompt:
        result = st.session_state.agent.respond(prompt)
        st.session_state.turns.append(
            {"heard": prompt, "reply": result.reply, "voice": False, "result": result}
        )
        st.rerun()
