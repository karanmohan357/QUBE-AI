# VOICE_BOT.py

import tempfile

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from gtts import gTTS

from QUERY import answer_question_voice_bot  # uses your existing QA bot

SAMPLE_RATE = 16000
_whisper_model = None


def load_whisper_model(model_name: str = "base"):
    """
    Lazy‑load a Whisper model so it's only loaded once.
    """
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model


def record_audio(duration: int = 8, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Record mono audio from the microphone and save it to a temp WAV file.

    Returns: path to the recorded WAV file.
    """
    # Record from default microphone
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    # Convert float32 [-1, 1] to int16
    audio_int16 = np.int16(audio * 32767)

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(tmp_file.name, sample_rate, audio_int16)
    return tmp_file.name


def transcribe_audio(audio_path: str, model_name: str = "base") -> str:
    """
    Transcribe a WAV/MP3 file to text using Whisper.
    """
    model = load_whisper_model(model_name)
    result = model.transcribe(audio_path, language="en")
    text = result.get("text", "").strip()
    return text


def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Convert text to speech using gTTS and return path to a temp MP3 file.
    """
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts = gTTS(text=text, lang=lang)
    tts.save(tmp_file.name)
    return tmp_file.name


def process_audio_query(
    book_file: str = "generated_book.docx",
    duration: int = 8,
    model_name: str = "base",
):
    """
    High‑level helper:
      1. Record audio from mic
      2. Transcribe to text
      3. Answer using your book QA (answer_question)
      4. Synthesize spoken answer with gTTS

    Returns a dict:
      {
        "question_text": str,
        "answer_text": str,
        "question_audio_path": str,
        "answer_audio_path": str,
      }
    """
    # 1) Record from mic
    question_audio_path = record_audio(duration=duration, sample_rate=SAMPLE_RATE)

    # 2) Speech‑to‑text
    question_text = transcribe_audio(question_audio_path, model_name=model_name)
    if not question_text:
        raise ValueError("No voice detected / transcription is empty. Please try again.")

    # 3) Get answer from your existing book QA bot
    answer_text = answer_question_voice_bot(book_file=book_file, question=question_text)
    # Just in case it returns a dict in future
    if isinstance(answer_text, dict) and "result" in answer_text:
        answer_text = answer_text["result"]
    answer_text = str(answer_text)

    # 4) Text‑to‑speech for the answer
    answer_audio_path = text_to_speech(answer_text)

    return {
        "question_text": question_text,
        "answer_text": answer_text,
        "question_audio_path": question_audio_path,
        "answer_audio_path": answer_audio_path,
    }
