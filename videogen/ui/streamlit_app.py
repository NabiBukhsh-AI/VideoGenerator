"""Streamlit front end.

The original UI had a structural bug: the background-music strength slider was created
*inside* ``if generate_button:``. Streamlit reruns the whole script on every
interaction, so that slider was drawn only during the click that started generation,
and moving it triggered a rerun which discarded the run. It could never do anything but
return its default.

The fix is the standard Streamlit pattern this file follows throughout: all inputs are
declared unconditionally, their values live in ``st.session_state``, and the button
click only consumes them. Generation is also split into two steps - review the script,
then render - so you can reject a bad script before paying for images and narration.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st

# Allow `streamlit run videogen/ui/streamlit_app.py` from a source checkout.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from videogen import __version__
from videogen.config import reload_settings
from videogen.errors import VideoGenError
from videogen.ingest import load_source
from videogen.logging import ProgressReporter, configure
from videogen.models import AspectRatio, KenBurnsKind, TransitionKind, Usage, VideoSpec
from videogen.pipeline import GenerationRequest, Pipeline
from videogen.providers import KNOWN, build_llm, build_tts
from videogen.render import ffmpeg
from videogen.script.generator import ScriptGenerator, ScriptRequest
from videogen.utils import format_duration

st.set_page_config(page_title="VideoGenerator", page_icon="🎬", layout="wide")
configure("INFO")


class StreamlitProgress(ProgressReporter):
    """Drives a Streamlit progress bar and status line."""

    def __init__(self, bar, status) -> None:
        super().__init__()
        self._bar = bar
        self._status = status
        self._stage = ""

    def stage(self, name: str, detail: str = "") -> None:
        self._stage = name
        self._status.info(f"{name}{f' - {detail}' if detail else ''}")
        self._bar.progress(0.0)

    def update(self, fraction: float, detail: str = "") -> None:
        self._bar.progress(min(1.0, max(0.0, fraction)))
        if detail:
            self._status.info(f"{self._stage} - {detail}")

    def warn(self, message: str) -> None:
        st.warning(message)


def sidebar(settings):
    """Render the settings sidebar and return the chosen configuration."""
    st.sidebar.title("🎬 VideoGenerator")
    st.sidebar.caption(f"v{__version__}")

    available = settings.available_providers()
    dry_run = st.sidebar.toggle(
        "Dry run (offline)",
        value=not any(
            settings.secret(k)
            for k in ("openai_api_key", "anthropic_api_key", "elevenlabs_api_key")
        ),
        help="Use local stubs only. No API calls, no cost. Good for checking the render.",
    )

    st.sidebar.subheader("Providers")
    if dry_run:
        st.sidebar.info("Dry run: echo / placeholder / silent.")
        llm_choice = image_choice = tts_choice = None
    else:
        llm_choice = st.sidebar.selectbox("Script model", available["llm"] or list(KNOWN["llm"]))
        image_choice = st.sidebar.selectbox("Images", available["image"] or list(KNOWN["image"]))
        tts_choice = st.sidebar.selectbox("Narration", available["tts"] or list(KNOWN["tts"]))

    voice = _voice_picker(settings, tts_choice, dry_run)

    st.sidebar.subheader("Video")
    aspect = st.sidebar.selectbox(
        "Aspect ratio", list(AspectRatio),
        format_func=lambda a: {"9:16": "9:16 vertical (Shorts, Reels)",
                               "1:1": "1:1 square", "16:9": "16:9 landscape"}[a.value],
    )
    max_scenes = st.sidebar.slider("Maximum scenes", 3, 12, settings.max_scenes)
    fps = st.sidebar.select_slider("Frame rate", [24, 25, 30, 60], value=30)
    motion = st.sidebar.selectbox(
        "Camera motion", list(KenBurnsKind), index=list(KenBurnsKind).index(KenBurnsKind.AUTO),
        format_func=lambda k: k.value.replace("_", " ").title(),
    )
    transition = st.sidebar.selectbox(
        "Transition", list(TransitionKind),
        format_func=lambda t: t.value.replace("_", " ").title(),
    )

    st.sidebar.subheader("Captions")
    captions_on = st.sidebar.toggle("Burn in captions", value=True)
    caption_color = st.sidebar.color_picker("Text", "#FFFFFF", disabled=not captions_on)
    highlight_color = st.sidebar.color_picker("Spoken word", "#FFD400", disabled=not captions_on)
    words_per_line = st.sidebar.slider("Words per line", 2, 8, 4, disabled=not captions_on)

    # Declared unconditionally - this is the bug the original had.
    st.sidebar.subheader("Background music")
    music_file = st.sidebar.file_uploader("Music track", type=["mp3", "wav", "m4a", "ogg"])
    music_gain = st.sidebar.slider(
        "Music level (dB)", -40.0, 0.0, -18.0, 1.0,
        help="How loud the bed sits under the narration. It ducks automatically when the "
             "narrator speaks.",
        disabled=music_file is None,
    )

    with st.sidebar.expander("Advanced"):
        tone = st.text_input("Tone", placeholder="urgent, wry, warm...")
        language = st.text_input("Narration language", placeholder="English")
        focus = st.text_input("Angle to emphasise", placeholder="the economic impact")
        crf = st.slider("Quality (CRF)", 14, 32, 20, help="Lower is better quality, bigger file.")

    spec = VideoSpec(
        aspect=aspect, fps=fps, transition=transition, ken_burns=motion,
        captions_enabled=captions_on, caption_color=caption_color,
        caption_highlight_color=highlight_color, caption_words_per_line=words_per_line,
        music_enabled=music_file is not None, music_gain_db=music_gain, crf=crf,
    )
    return {
        "spec": spec, "dry_run": dry_run, "llm": llm_choice, "image": image_choice,
        "tts": tts_choice, "voice": voice, "music_file": music_file, "tone": tone or None,
        "language": language or None, "focus": focus or None, "max_scenes": max_scenes,
    }


def _voice_picker(settings, tts_choice, dry_run):
    if dry_run or not tts_choice:
        return None
    try:
        provider = build_tts(settings, tts_choice)
        voices = provider.list_voices()
    except VideoGenError:
        voices = []
    if voices:
        return st.sidebar.selectbox("Voice", voices)
    return st.sidebar.text_input("Voice", value=settings.elevenlabs_voice)


def source_input() -> object | None:
    """Collect source material from one of three tabs."""
    tab_file, tab_url, tab_text = st.tabs(["📄 File", "🔗 URL", "✍️ Text"])

    with tab_file:
        uploaded = st.file_uploader(
            "PDF, Markdown or plain text", type=["pdf", "txt", "md", "markdown"]
        )
        if uploaded is not None:
            return uploaded

    with tab_url:
        url = st.text_input("Article URL", placeholder="https://example.com/article")
        if url.strip():
            return url.strip()

    with tab_text:
        pasted = st.text_area("Paste your source material", height=220)
        if pasted.strip():
            return pasted.strip()

    return None


def main() -> None:
    settings = reload_settings()
    config = sidebar(settings)
    settings = settings.model_copy(
        update={"max_scenes": config["max_scenes"],
                "min_scenes": min(settings.min_scenes, config["max_scenes"])}
    )

    st.title("Turn source material into a short video")
    st.caption(
        "Upload a document, paste a URL, or type text. Review the script, then render."
    )

    if not ffmpeg.available(settings):
        st.warning(
            "**ffmpeg was not found.** You can generate scripts, images and narration, but "
            "rendering needs it. Install: `winget install Gyan.FFmpeg` (Windows), "
            "`brew install ffmpeg` (macOS), `sudo apt install ffmpeg` (Linux)."
        )

    source = source_input()

    col_script, col_render = st.columns([1, 1])
    with col_script:
        write_script = st.button(
            "1. Write the script", type="primary", disabled=source is None,
            use_container_width=True,
        )
    with col_render:
        render_now = st.button(
            "2. Render the video", disabled="script" not in st.session_state,
            use_container_width=True,
        )

    if write_script and source is not None:
        _write_script(settings, config, source)

    if "script" in st.session_state:
        _show_script_editor()

    if render_now:
        _render(settings, config)

    if "result" in st.session_state:
        _show_result()


def _write_script(settings, config, source) -> None:
    """Stage one: ingest and generate the script only."""
    try:
        with st.spinner("Reading source material..."):
            document = load_source(source, max_words=settings.max_source_words)
        st.success(f"Read {document.word_count:,} words from {document.name}")

        effective = settings.offline() if config["dry_run"] else settings
        llm = build_llm(effective, config["llm"])
        llm.health_check()

        with st.spinner(f"Writing the script with {llm.label}..."):
            script, usage = ScriptGenerator(llm, settings).generate(
                ScriptRequest(document=document, tone=config["tone"],
                              language=config["language"], focus=config["focus"])
            )

        st.session_state["script"] = script
        st.session_state["document"] = document
        st.session_state["source"] = source
        st.session_state["usage"] = usage
        st.session_state.pop("result", None)
    except VideoGenError as exc:
        _show_error(exc)
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        st.code(traceback.format_exc())


def _show_script_editor() -> None:
    """Let the user review and edit the script before paying for assets."""
    script = st.session_state["script"]

    st.subheader(script.title)
    if script.description:
        st.caption(script.description)

    meta = st.columns(3)
    meta[0].metric("Scenes", len(script.scenes))
    meta[1].metric("Words", script.total_words)
    meta[2].metric("Estimated runtime", format_duration(script.estimated_duration_ms))

    if script.hashtags:
        st.write(" ".join(f"`#{tag}`" for tag in script.hashtags))

    st.markdown("**Edit any scene before rendering:**")
    edited = False
    for scene in script.scenes:
        with st.expander(f"Scene {scene.index + 1}: {scene.narration[:60]}...", expanded=False):
            narration = st.text_area(
                "Narration", scene.narration, key=f"nar_{scene.index}", height=80
            )
            prompt = st.text_area(
                "Image prompt", scene.image_prompt, key=f"img_{scene.index}", height=80
            )
            if narration != scene.narration and narration.strip():
                scene.narration = narration
                edited = True
            if prompt != scene.image_prompt and prompt.strip():
                scene.image_prompt = prompt
                edited = True

    if edited:
        st.info("Scene edits saved. They will be used when you render.")


def _render(settings, config) -> None:
    """Stage two: generate assets and render the video."""
    if "script" not in st.session_state:
        st.error("Write a script first.")
        return

    music_path = None
    if config["music_file"] is not None:
        # Persist the upload so ffmpeg can read it from disk.
        import tempfile

        suffix = Path(config["music_file"].name).suffix or ".mp3"
        # delete=False so ffmpeg can reopen it by path; removed in the finally block.
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # noqa: SIM115
        config["music_file"].seek(0)
        handle.write(config["music_file"].read())
        handle.close()
        music_path = Path(handle.name)

    bar = st.progress(0.0)
    status = st.empty()
    progress = StreamlitProgress(bar, status)

    request = GenerationRequest(
        source=st.session_state["source"],
        # Reuse the reviewed (and possibly edited) script instead of regenerating it.
        script=st.session_state["script"],
        spec=config["spec"],
        voice=config["voice"],
        music_path=music_path,
        tone=config["tone"],
        language=config["language"],
        focus=config["focus"],
        llm_override=config["llm"],
        image_override=config["image"],
        tts_override=config["tts"],
        dry_run=config["dry_run"],
    )

    try:
        result = Pipeline(settings, progress).run(request)
        st.session_state["result"] = result
        status.success("Done.")
        bar.progress(1.0)
    except VideoGenError as exc:
        _show_error(exc)
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        st.code(traceback.format_exc())
    finally:
        if music_path is not None:
            music_path.unlink(missing_ok=True)


def _show_result() -> None:
    result = st.session_state["result"]
    st.divider()
    st.subheader("Your video")

    video_path = Path(result.video_path)
    if video_path.is_file():
        col_video, col_meta = st.columns([2, 3])
        with col_video:
            st.video(str(video_path))
        with col_meta:
            st.metric("Duration", format_duration(result.duration_ms))
            st.write(f"**Saved to:** `{video_path}`")

            with video_path.open("rb") as handle:
                st.download_button(
                    "Download MP4", handle.read(), file_name=video_path.name,
                    mime="video/mp4", type="primary",
                )
            if result.subtitle_path and Path(result.subtitle_path).exists():
                st.download_button(
                    "Download subtitles (.srt)",
                    Path(result.subtitle_path).read_text(encoding="utf-8"),
                    file_name=Path(result.subtitle_path).name,
                )

            usage: Usage = result.usage
            if usage.estimated_cost_usd > 0:
                st.caption(
                    f"Estimated cost: ${usage.estimated_cost_usd:.3f} - "
                    f"{usage.images_generated} images, {usage.tts_characters:,} TTS characters"
                )
    else:
        st.info(f"Project written to `{result.project_dir}`")


def _show_error(exc: VideoGenError) -> None:
    st.error(exc.message)
    if exc.hint:
        st.info(exc.hint)


if __name__ == "__main__":
    main()
