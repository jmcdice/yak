import argparse
import os
from pathlib import Path
import concurrent.futures
from openai import OpenAI
from .core import transcribe_audio, find_audio_files

def parse_args():
    parser = argparse.ArgumentParser(
        description="yak: Batch transcribe audio files using OpenAI Whisper"
    )
    parser.add_argument("--path", default=".", help="Directory with audio files")
    parser.add_argument("--patterns", default="*.m4a,*.wav,*.mp3", help="Comma-separated glob patterns (default: *.m4a,*.wav,*.mp3)")
    parser.add_argument("--output-dir", default=None, help="Output directory for transcripts (default: same as audio file)")
    parser.add_argument("--combine", action="store_true", help="Combine all transcripts into combined.txt")
    parser.add_argument("--parallel", type=int, default=1, help="Number of threads (default: 1)")
    parser.add_argument("--model", default="whisper-1", help="OpenAI Whisper model name")
    parser.add_argument("--response-format", default="text", help="Transcription format: text/json/srt/vtt")
    return parser.parse_args()

def main():
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("yak: error: Set OPENAI_API_KEY environment variable.")
        exit(1)

    client = OpenAI(api_key=api_key)

    audio_dir = Path(args.path).resolve()
    patterns = [pat.strip() for pat in args.patterns.split(",")]
    audio_files = find_audio_files(audio_dir, patterns)

    if not audio_files:
        print(f"yak: no files found in {audio_dir} with patterns: {patterns}")
        exit(0)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    combined_transcripts = []
    errors = []

    def transcribe_and_collect(file_path: Path):
        outdir = output_dir or file_path.parent
        out_file = outdir / (file_path.stem + "_transcript.txt")
        result = transcribe_audio(
            client, file_path, out_file,
            model=args.model, response_format=args.response_format
        )
        if result:
            return result
        else:
            errors.append(str(file_path))
            return None

    print(f"yak: found {len(audio_files)} file(s):")
    for f in audio_files:
        print(f"  - {f.name}")

    if args.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            results = list(executor.map(transcribe_and_collect, audio_files))
    else:
        results = []
        for f in audio_files:
            results.append(transcribe_and_collect(f))

    combined_transcripts = [t for t in results if t]

    if args.combine and combined_transcripts:
        combo_path = (output_dir or audio_dir) / "combined.txt"
        with combo_path.open("w") as combo:
            for t in combined_transcripts:
                combo.write(t.strip() + "\n\n")
        print(f"\nyak: all transcripts combined into {combo_path}")

    if errors:
        print("\nyak: failed to transcribe:")
        for e in errors:
            print(f"  - {e}")

if __name__ == "__main__":
    main()
