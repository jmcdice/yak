# yak

Batch audio transcription using OpenAI Whisper.

## Install

```bash
pip install git+https://github.com/jmcdice/yak.git
```

## Usage

```bash
export OPENAI_API_KEY=sk-...
yak --path ./audio --patterns "*.m4a,*.wav" --parallel 4 --combine
```

See `yak --help` for all options.

## License

MIT
