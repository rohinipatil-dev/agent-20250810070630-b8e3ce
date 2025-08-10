import os
import sys
import io
import re
import tempfile
import subprocess
from typing import Optional, Tuple

import streamlit as st
from openai import OpenAI


# Initialize OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI()


def ensure_yt_dlp_installed() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp>=2024.4.9"])
            import yt_dlp  # noqa: F401
            return True
        except Exception:
            return False


def is_valid_vimeo_url(url: str) -> bool:
    if not url:
        return False
    pattern = r"(https?://)?(www\.)?vimeo\.com/[\w/]+"
    return re.match(pattern, url.strip()) is not None


def get_audio_mime(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    mapping = {
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "mp4": "audio/mp4",
        "aac": "audio/aac",
        "wav": "audio/wav",
        "webm": "audio/webm",
        "ogg": "audio/ogg",
        "oga": "audio/ogg",
        "flac": "audio/flac",
        "mka": "audio/x-matroska",
        "opus": "audio/opus",
    }
    return mapping.get(ext, "audio/mpeg")


def download_best_audio_from_vimeo(url: str) -> Tuple[bytes, str, dict]:
    """
    Downloads best available audio-only stream without requiring ffmpeg postprocessing.
    Returns: (audio_bytes, file_extension, info_dict)
    """
    if not ensure_yt_dlp_installed():
        raise RuntimeError("Failed to install yt-dlp. Please install it manually: pip install yt-dlp")

    import yt_dlp

    with tempfile.TemporaryDirectory(prefix="vimeo_dl_") as tmpdir:
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
            "ignoreerrors": False,
            "cachedir": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("Failed to extract info from Vimeo URL.")
            # Prepare filename based on selected format
            downloaded_path = ydl.prepare_filename(info)
            if not os.path.exists(downloaded_path):
                # Try to locate the file from requested downloads if prepare_filename didn't match
                requested = info.get("requested_downloads") or []
                if requested:
                    downloaded_path = requested[0].get("_filename", downloaded_path)

            if not os.path.exists(downloaded_path):
                raise RuntimeError("Downloaded audio file not found.")

            ext = os.path.splitext(downloaded_path)[1].lstrip(".")
            with open(downloaded_path, "rb") as f:
                audio_bytes = f.read()

    return audio_bytes, ext, info


def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "audio.m4a") -> str:
    """
    Sends audio bytes to OpenAI Whisper for transcription.
    """
    # Use a temporary file-like object in memory
    file_like = io.BytesIO(audio_bytes)
    file_like.name = filename  # OpenAI SDK uses the name to detect file type
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_like,
    )
    return transcript.text


def format_video_info(info: dict) -> str:
    title = info.get("title") or "Untitled"
    uploader = info.get("uploader") or "Unknown uploader"
    duration = info.get("duration") or 0
    minutes = duration // 60
    seconds = duration % 60
    return f"Title: {title}\nUploader: {uploader}\nDuration: {minutes}m {seconds}s"


def main():
    st.set_page_config(page_title="Vimeo to Text Transcriber", page_icon="🎧", layout="centered")
    st.title("Vimeo to Text Transcriber")
    st.write("Paste a public Vimeo link. The app will extract the audio, transcribe it using Whisper, and display the text.")

    # API key warning
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Please set it in your environment before transcribing.")

    vimeo_url = st.text_input("Vimeo URL", placeholder="https://vimeo.com/xxxxxxxxx")

    col1, col2 = st.columns([1, 1])
    transcribe_clicked = col1.button("Transcribe", type="primary")
    clear_clicked = col2.button("Clear")

    if clear_clicked:
        for key in ["transcript_text", "audio_bytes", "audio_ext", "video_info"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    if transcribe_clicked:
        if not is_valid_vimeo_url(vimeo_url):
            st.error("Please provide a valid Vimeo URL.")
            st.stop()

        with st.spinner("Downloading audio from Vimeo..."):
            try:
                audio_bytes, audio_ext, info = download_best_audio_from_vimeo(vimeo_url)
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["audio_ext"] = audio_ext
                st.session_state["video_info"] = info
            except Exception as e:
                st.error(f"Audio download failed: {e}")
                st.stop()

        with st.spinner("Transcribing audio with OpenAI Whisper..."):
            try:
                fake_name = f"audio.{st.session_state['audio_ext']}"
                transcript = transcribe_audio_bytes(st.session_state["audio_bytes"], filename=fake_name)
                st.session_state["transcript_text"] = transcript
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

    # Display results if available
    if "audio_bytes" in st.session_state and "audio_ext" in st.session_state:
        st.subheader("Audio Preview")
        mime = get_audio_mime(st.session_state["audio_ext"])
        st.audio(st.session_state["audio_bytes"], format=mime)

    if "video_info" in st.session_state:
        st.subheader("Video Info")
        st.text(format_video_info(st.session_state["video_info"]))

    if "transcript_text" in st.session_state:
        st.subheader("Transcript")
        st.text_area("Transcribed Text", value=st.session_state["transcript_text"], height=300)
        st.download_button(
            label="Download Transcript",
            data=st.session_state["transcript_text"],
            file_name="transcript.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()