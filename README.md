# voicetotext

A minimal example project for experimenting with audio transcription.

## Requirements

- **Python**: 3.10 or newer.
- **System dependencies**: [ffmpeg](https://ffmpeg.org/) must be installed and
  available on your `PATH`.
- **Python packages**:

  ```bash
  pip install openai-whisper torch pydub webrtcvad
  ```

  Install a CUDA-enabled build of `torch` if you plan to use a GPU.

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

## Choosing CPU or GPU

The transcription pipeline runs on the CPU by default. To explicitly select a
device use the `--device` flag:

```bash
# CPU (default)
python src/cli.py input.wav output.txt --device cpu

# GPU
python src/cli.py input.wav output.txt --device cuda
```

The repository also includes `examples/transcribe_gpu.py`, which invokes the
pipeline with `device="cuda"` for convenience.

## Choosing a model

The pipeline uses the `base` Whisper model by default. To use a different
model, provide the `--model` flag:

```bash
python src/cli.py input.wav output.txt --model medium
```

## Tuning transcription parameters

The CLI exposes a few Whisper decoding options. The most common are
`--temperature` (default: `0.0`) and `--beam-size` (default: `5`). For example:

```bash
python src/cli.py input.wav output.txt --temperature 0.7 --beam-size 3
```

Higher temperatures make outputs more random, while larger beam sizes explore
more decoding paths.

## 한국어 안내

### 요구 사항

- **Python**: 3.10 이상.
- **시스템 의존성**: [ffmpeg](https://ffmpeg.org/)이 설치되어 있고 `PATH`에 있어야 합니다.
- **Python 패키지**:

  ```bash
  pip install openai-whisper torch pydub webrtcvad
  ```

  GPU를 사용하려면 CUDA가 활성화된 `torch` 버전을 설치하세요.

### 예제 실행하기

저장소에는 기본 제공 오디오 파일이 포함되지 않습니다. 자신의 WAV 파일을 제공하거나 예제 스크립트가 실행 시 작은 샘플 클립을 가져오도록 할 수 있습니다:

```bash
python examples/transcribe_example.py --audio path/to/your/audio.wav
```

`--audio` 플래그를 생략하면 짧은 샘플 클립을 자동으로 다운로드합니다. 변환된 텍스트는 기본적으로 `transcript.txt`에 저장되며 `--output` 플래그로 경로를 지정할 수 있습니다.

파이프라인은 오디오를 조각으로 나누기 위해 [ffmpeg](https://ffmpeg.org/)를 필요로 합니다. 설치되어 있고 `PATH`에 있는지 확인하세요.

### CPU 또는 GPU 선택하기

전사 파이프라인은 기본적으로 CPU에서 실행됩니다. 특정 디바이스를 선택하려면 `--device` 플래그를 사용하세요:

```bash
# CPU (기본값)
python src/cli.py input.wav output.txt --device cpu

# GPU
python src/cli.py input.wav output.txt --device cuda
```

또한 저장소에는 편의를 위해 `device="cuda"`로 파이프라인을 호출하는 `examples/transcribe_gpu.py`도 포함되어 있습니다.

### 모델 선택하기

기본적으로 파이프라인은 `base` Whisper 모델을 사용합니다. 다른 크기의
모델을 사용하려면 `--model` 플래그를 지정하세요:

```bash
python src/cli.py input.wav output.txt --model medium
```

### 디코딩 파라미터 조정하기

CLI는 Whisper 디코더의 몇 가지 옵션을 제공합니다. 대표적으로
`--temperature`(기본값: `0.0`)와 `--beam-size`(기본값: `5`)가 있습니다:

```bash
python src/cli.py input.wav output.txt --temperature 0.7 --beam-size 3
```

높은 temperature는 결과의 무작위성을 높이고, 큰 beam size는 더 많은 탐색 경로를 고려합니다.
