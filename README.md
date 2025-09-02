# voicetotext

A minimal example project for experimenting with audio transcription.

## Running the example

The repository no longer includes an embedded audio file. Supply your own
WAV file or allow the example script to fetch a small sample clip at
runtime:

```bash
python examples/transcribe_example.py --audio path/to/your/audio.wav
```

Omit the `--audio` flag to download a short sample clip
automatically. The resulting transcript will be written to
`transcript.txt` by default or to the path given by `--output`.

The pipeline requires [ffmpeg](https://ffmpeg.org/) to split the audio
into chunks. Make sure it is installed and available on your `PATH`.
