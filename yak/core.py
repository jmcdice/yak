from pathlib import Path
from typing import Optional
from openai import OpenAI

def transcribe_audio(
    client: OpenAI,
    file_path: Path,
    output_path: Path,
    model: str = "whisper-1",
    response_format: str = "text"
) -> Optional[str]:
    try:
        with file_path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format=response_format
            )
        with output_path.open("w") as out:
            out.write(transcription)
        print(f"yak: transcribed {file_path.name} → {output_path.name}")
        return transcription
    except Exception as e:
        print(f"yak: ERROR: {file_path.name}: {e}")
        return None

def find_audio_files(directory: Path, patterns: list[str]) -> list[Path]:
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(set(files))
