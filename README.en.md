# Local Video Dubbing

[繁體中文](README.md) · **English** · [日本語](README.ja.md)

Record a demo in Chinese and export it with an English dub, or dub an English video in Chinese. Video processing, speech recognition, translation, and speech synthesis all run on your computer. Initial installation and model downloads require an internet connection.

The first release supports single-speaker videos up to 5 minutes long and up to 1 GiB in size. Upload a video, choose a translation direction, run recognition and translation, review each segment, generate the dub, and download the MP4 and two SRT files. The app requests Traditional Chinese with Taiwan terminology. Check product names, numbers, and technical terms during review.

This is a prototype. End-to-end dubbing has been verified in both directions using Windows Ollama (`qwen3.5:4b`), WSL, and an RTX 4050 Laptop GPU, with successful MP4, source subtitle, and translated subtitle output. An iPhone on the same Wi-Fi network could open the web UI. Upload, playback, and download on iPhone still need full acceptance testing. The startup script now uses `gemma4:e2b` and GPU-based `medium`. This combination still needs end-to-end verification.

## 1. Set up the main environment (WSL / Ubuntu)

Clone the repository if you do not already have a working copy.

```bash
git clone https://github.com/adombciha/video-dubbing.git ~/video-dubbing
```

Run these commands in a WSL terminal. Use a virtual environment separate from the one for `~/multi-agents`.

```bash
cd ~/video-dubbing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
sudo apt update
sudo apt install -y ffmpeg
WHISPER_MODEL=medium python dubbing.py download-asr
```

If `venv` is missing, run `sudo apt install -y python3-venv` first. `start.sh` uses faster-whisper's multilingual `medium` model. Download its weights with the command above before starting.

GPU recognition also requires cuBLAS for CUDA 12 and cuDNN 9 in the main environment.

```bash
.venv/bin/python -m pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'
```

`start.sh` automatically adds the library paths for these packages. [faster-whisper GPU installation guide](https://github.com/SYSTRAN/faster-whisper#gpu)

## 2. Set up a separate CosyVoice 3 environment

CosyVoice's official dependencies use different versions from the main app. Install them in a separate Python 3.10 environment outside `.venv`. The commands below use an existing `uv` installation to create the environment. Speech model and GPU package downloads are large, so installation may take a while.

```bash
cd ~/video-dubbing
sudo apt install -y git build-essential sox libsox-dev
mkdir -p vendor models
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice
uv venv --python 3.10 .venv-cosyvoice
uv pip install --python .venv-cosyvoice/bin/python --index-strategy unsafe-first-match -r vendor/CosyVoice/requirements.txt
uv pip uninstall --python .venv-cosyvoice/bin/python deepspeed
```

These commands install the dependencies from the [official CosyVoice installation guide](https://github.com/FunAudioLLM/CosyVoice#install). The requirements file uses several package indexes, including PyTorch, Microsoft, and PyPI. `unsafe-first-match` lets uv check the next index when a higher-priority one has no compatible version. For example, a higher-priority index might otherwise mask `protobuf==4.25` on PyPI. This option relaxes uv's cross-index search restrictions. Use it only for this installation of the official requirements and leave the global default in place. [uv documentation](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)

If PyTorch, CUDA, or DeepSpeed installation fails, save the full error and check `nvidia-smi` first. Keep the package versions in the main environment in place. The project can run without a separate vLLM installation.

This project uses standard PyTorch inference and runs without DeepSpeed training or acceleration. The official requirements install DeepSpeed, so remove it from the separate environment afterward. This prevents Transformers from trying to load DeepSpeed and requiring CUDA build tools that are absent. PyTorch and CUDA remain available after removal. If you later need to run upstream DeepSpeed training, set up a separate environment with the CUDA Toolkit.

Download the model.

```bash
.venv-cosyvoice/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='models/Fun-CosyVoice3-0.5B',
)
PY
```

Check the environment after installation.

```bash
cd ~/video-dubbing
COSYVOICE_PYTHON="$PWD/.venv-cosyvoice/bin/python" WHISPER_MODEL=medium .venv/bin/python dubbing.py check --offline
```

The default source and model paths match the locations above. `check --offline` checks installed packages, the Whisper model cache, CosyVoice paths, and dependency imports. After setting up Ollama, open **Environment Status and Connection Check** (`環境狀態與連線檢查`) in the web UI to check the model list. Generate a dub to verify GPU inference and audio quality.

## 3. Windows Ollama

The app reads connection settings from `~/multi-agents/.ai-company/config.json` in read-only mode. `start.sh` sets the translation model to `gemma4:e2b`. Windows Ollama must be running with that exact model tag installed.

Check the model list in a Windows terminal.

```powershell
ollama list
```

Install a model that supports text chat and JSON output. The app never downloads or deletes Ollama models automatically. It uses `/api/tags` to list models and `/api/chat` to translate. When translation finishes, it asks Ollama to unload the model to free GPU memory for speech synthesis.

`gemma4:e2b` and `gemma4:latest` are different tags. If the Windows IP changes, override the address with `OLLAMA_URL`.

Translation requests try to disable thinking to reduce latency for short segments and fall back to the model's default mode when it lacks that option. The app checks each batch for the correct segment count, IDs and order, and for nonempty translations. It requests a correction if validation fails and makes at most one extra call per batch. If validation still fails, it stops before applying any partial results.

## 4. Start and use the app

Once installation and model downloads are complete, start the app.

```bash
cd ~/video-dubbing
./start.sh
```

The script uses the project's virtual environment, `gemma4:e2b` for translation, faster-whisper `medium` on CUDA with FP16 for recognition, and the separate CosyVoice environment. It also sets the GPU library paths. These settings are fixed in `start.sh` and override existing shell settings for the same models and devices. The script listens on the local network. You can pass app arguments after it, as in `./start.sh --port 7861`.

Keep the app running while you use it and press `Ctrl+C` to stop it. Run `./start.sh` again after a reboot.

Open `http://127.0.0.1:7860` on your computer and follow these steps.

1. Upload an MP4, MOV, M4V, MKV, WebM, or AVI file, choose a translation direction, and click **Recognize and Translate** (`辨識與翻譯`).
2. Review the source text and translation. After editing the source text, you can click **Retranslate from Edited Source Text** (`依修正原文重新翻譯`) to replace the current translation.
3. By default, the app takes up to 12 seconds from the first speech segment as a voice reference. If that segment is under 3 seconds, noisy, or has long pauses, upload a clean 3–15-second speech sample. Use your own voice or material you are authorized to use.
4. Click **Generate Dubbed Video** (`產生配音影片`). The app measures the duration of the synthesized speech. If any segments run long, it asks Gemma to shorten them and regenerates only those segments, for up to two automatic rounds. When it succeeds, the review table shows the translation actually used. Review it again. If segments still run long, the app shows which segments need manual edits and their durations in seconds.
5. Preview or download the MP4, translated SRT, and source SRT. Subtitles are provided as separate SRT files without being burned into the video.

Translation requests include the timing and a suggested text length for each segment. The measured duration after synthesis determines whether a line fits. Dubbing keeps each original segment's start time, uses the gap before the next line first, and speeds up speech if needed, up to about 1.2× the original speed. When a segment exceeds that limit, the app requests a shorter translation based on its measured duration and regenerates the speech. This can increase processing time and change wording. Review the updated translation afterward. If a segment still runs long after two correction rounds, the app stops before exporting to preserve speech without cutting it off or overlapping it.

Each render backs up the source text and the translations before shortening in `render-*/original-segments.json`. Regenerated audio is stored in `retry-1/` and `retry-2/`. The app updates the job translation and review table only after the full render succeeds. If it fails, the submitted text stays available for editing and retrying. The translated subtitles use the final text, with timings adjusted to match the synthesized speech.

CosyVoice starts one subprocess per dubbing job. It loads the model and prepares the reference voice features once, then reuses both across retries. After each speech generation round, it moves the weights and reference features to RAM and releases PyTorch's unused GPU cache before Gemma shortens the text. It moves the weights back to the GPU for the next round. Keep enough RAM available for the model. The GPU driver may still hold some GPU memory. The worker exits on completion, failure, or timeout, and the next job loads the model again. Recognition uses a separate subprocess, and the UI runs one model task at a time.

Each render writes performance data to `render-*/cosyvoice-timing.jsonl`, including seconds spent on `torch_import`, `cosyvoice_import`, `model_load`, `reference`, per-segment `speech`, `park_cpu`, `resume_gpu`, and Gemma `shorten`. Full worker output is in `cosyvoice.log` in the same directory. When comparing runs of the same video, include the initial load and all regeneration rounds. Verify actual RTX 4050 performance in a regular WSL terminal. The tests use mock models.

The original video stays intact. A copy of it, segment text, synthesized speech, and finished output are saved under `data/jobs/job-*/`. Each dubbing retry creates a new `render-*` directory and keeps earlier results. Job data persists until you remove it. Close the app before manually deleting job directories you no longer need.

## 5. Use an iPhone on the same Wi-Fi network

`start.sh` enables local network connections.

```bash
./start.sh
```

Open `http://<Windows LAN IPv4>:7860` in Safari and choose a prerecorded video from Photos or Files. Your computer processes the video. Open finished downloads from Safari's download list and use the available iOS options to save or share them.

WSL local network access depends on the Windows and WSL networking mode. If Windows can open the app and the iPhone cannot, check WSL mirrored networking, the Windows firewall, or port forwarding. `--host 0.0.0.0` enables listening on all interfaces, and access still depends on the network configuration.

Use the IPv4 address of the active Windows Wi-Fi or Ethernet connection on your phone. On the iPhone, `127.0.0.1` points to the phone itself. `0.0.0.0` is the server's listen address. In Windows PowerShell, find the interface with the current default gateway.

```powershell
Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object InterfaceAlias, @{Name='IPv4';Expression={$_.IPv4Address.IPAddress}}
```

If the phone still cannot connect, check the networking mode and WSL IP address in a regular WSL terminal. These commands only read network information.

```bash
wslinfo --networking-mode
hostname -I
```

In NAT mode, Windows can open the WSL page through `localhost`, and the phone usually needs Windows port forwarding and firewall rules to reach it. Mirrored mode can accept local network connections directly, with access still subject to the firewall. Confirm the actual mode, IP, and browser error before making changes specific to TCP 7860. Keep the firewall enabled. [Microsoft WSL networking guide](https://learn.microsoft.com/en-us/windows/wsl/networking)

Set `DUBBING_USERNAME` and `DUBBING_PASSWORD` if you need sign-in. Avoid public sharing links. The app always starts with `share=False`. To enable HTTPS, supply a certificate you trust.

```bash
./start.sh --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

Record videos in the native Camera app and then upload them. The Safari workflow uses file uploads without needing in-page camera access. The app is available on iPhone through the browser, with no native app or PWA package.

## 6. Live trilingual caption bar (Windows Chrome)

Stop any service using port 7860 before starting live caption mode.

```bash
cd ~/video-dubbing
./start.sh --live
```

Open `http://127.0.0.1:7860` in **Chrome on Windows**.

1. Select **Chrome Tab** (`Chrome 分頁`) or **System Audio** (`系統音訊`), then set the source and subtitle languages. Traditional Chinese, English, and Japanese are supported.
2. Click **Start Translating** (`開始翻譯`). In Chrome, select the tab or entire screen to capture and enable **Share Audio** (`分享音訊`). If the source supplies no audio track, the app prompts you to choose again.
3. After you confirm audio sharing, a narrow always-on-top caption bar opens automatically. If Chrome fails to open it, you can still capture audio from the main page or click **Float on Top** (`懸浮置頂`).
4. Wait for **Capturing Audio** (`正在收音`) before starting playback. The floating bar shows the translation, status, and stop button. The main page also shows the source text and processing time. Keep the original tab open. Closing the floating bar stops capture as well.
5. Click **Stop** (`停止`) to finish processing captured segments, release audio and models, and return the toolbar to the original tab.

`start.sh` uses the current Windows gateway as the Ollama address in WSL NAT mode and `127.0.0.1` in mirrored mode. Set `OLLAMA_URL` to use another Ollama address. An explicit setting takes precedence.

This mode takes its input directly from the digital audio track. It uses no microphone and generates subtitles without synthesizing dubbed audio. **Mute Tab During Capture** (`分頁靜音擷取`) asks Chrome to suppress local audio playback for the captured tab. Keep the video player's own volume above zero, since turning it all the way down may also silence the captured track. System audio availability and audio capture while muted depend on Chrome, Windows, and the player. Check both with a short clip first.

The captioner buffers about 3 seconds of audio before sending a segment when it detects a 0.6-second pause. During continuous speech, it sends a segment at about 5 seconds at the latest. Recognition and translation add to that time. This groups short phrases to reduce repeated inference and puts the first subtitle for a short phrase slightly later. Clicking **Stop** (`停止`) still sends remaining audio shorter than 3 seconds.

This mode also uses faster-whisper `medium`, with `int8_float16` on CUDA, and loads it once each time capture starts. Live recognition makes one decoding pass (`beam_size=1`, `best_of=1`, `temperature=0`) and skips timestamps. This avoids repeated decoding for low-confidence segments and may reduce accuracy for noisy or hard-to-understand speech. Gemma `gemma4:e2b` stays loaded during capture. Before and after preloading, and again before each translation, the app reads and reuses the actual context size from Ollama's `/api/ps`. This avoids repeatedly changing the configuration when another app shares the same model. The app accepts only a positive integer setting for the exact model name and starts with 2048. Initial preparation takes longer. This mode runs without CosyVoice.

The main page shows the audio duration, recognition time, translation time, and processing speed for each segment. Speed is the audio duration divided by that segment's processing time. A value above `1×` means processing was faster than playback. This measurement excludes time spent capturing audio, loading models for the first time, and waiting in the queue. The app may fall behind during continuous playback if speed stays below `1×`. When three segments are waiting, it stops capture automatically and finishes processing the audio it already received to keep caption delays from growing. Segmentation and mock-model tests verify request counts and timing behavior. Compare the same audio on the GPU to verify actual speed and recognition quality.

If Ollama aborts because of repetitive output (`token repeat limit reached`), the app retries once with plain text output and a low sampling temperature. If the retry also fails, it marks the translation as failed, displays the source text, and moves on to the next segment.

Background jobs that share Ollama can also cause queuing. In a local short-phrase test with an RTX 4050 (6 GB), `medium`, and `gemma4:e2b`, the same roughly 3.8-second English clip sometimes took 7–23 seconds to process while the Hindsight background service was running. After pausing that service, recognition and translation together took 0.94, 0.75, and 0.81 seconds across three runs. Model preparation took about 25 seconds in that test. These measurements cover a short English sample. Longer audio and Chinese and Japanese still need acceptance testing for quality and speed. Check other Ollama clients if processing times fluctuate.

The caption service listens only on the local machine by default and uses the same sign-in settings. Tab and system audio capture require a secure context in desktop Chrome (localhost or HTTPS). Plain HTTP over the local network on an iPhone cannot use this capture mode. Run `./start.sh` to use video upload and dubbing. Run only one mode at a time to avoid contention for port 7860 or GPU memory.

## Configuration

The table lists defaults when you run Python directly. With `start.sh`, translation uses `gemma4:e2b`, recognition uses `medium` on `cuda`, and CosyVoice uses the project's `.venv-cosyvoice/bin/python`.

| Environment variable | Default or purpose |
| --- | --- |
| `OLLAMA_URL` | Uses the `multi-agents` setting when available, otherwise `http://localhost:11434` |
| `OLLAMA_MODEL` | Uses the `multi-agents` setting when available, otherwise `gemma4:latest` |
| `WHISPER_MODEL` | `small`. Accepts a local model path. Run `download-asr` after changing models. |
| `ASR_DEVICE` | `auto`. Also accepts `cpu` or `cuda`. CPU uses int8 and CUDA uses float16. |
| `COSYVOICE_PYTHON` | The main app's Python. Point it to `.venv-cosyvoice/bin/python` after the installation above. |
| `COSYVOICE_ROOT` | Absolute path to `vendor/CosyVoice` |
| `COSYVOICE_MODEL_DIR` | Absolute path to `models/Fun-CosyVoice3-0.5B` |
| `FFMPEG` / `FFPROBE` | `ffmpeg` / `ffprobe` on PATH. Accepts paths to the executables. |
| `DUBBING_OUTPUT_DIR` | `data/jobs` under the project directory |
| `DUBBING_USERNAME` / `DUBBING_PASSWORD` | Optional. Set them together. |

`auto` chooses the ASR device based only on whether a CUDA device is visible. If the cuDNN or CUDA libraries are incompatible, the app reports an error. Use `ASR_DEVICE=cpu` to verify the workflow in that case. Ollama translation quality, voice similarity, accent, and speed depend on the models and hardware. The synthesized speech may differ from the original in emotion and tone of voice.

The recognition worker directly reads 16 kHz mono PCM16 WAV converted by FFmpeg and sends normalized audio samples to Whisper. This avoids decoding the audio twice and sidesteps differences in audio-opening parameters across PyAV versions.

## Verification

```bash
python -m pytest -q
python dubbing.py check
```

Unit tests cover missing translation segments, incorrect IDs, prompts for both translation directions, text review, overlapping timings, speech that runs long, silent audio, subtitles, and preservation of the original file. Automatic shortening tests check that only segments that run long are regenerated, retries stop after two rounds, original translations are backed up, submitted text is preserved on failure, and the review table and subtitles stay in sync. FFmpeg integration tests generate video, adjust speed, combine audio and video, and verify duration. They explicitly skip when FFmpeg or ffprobe is missing. The tests use synthetic audio. Evaluating model quality requires a separate test with real content.

An earlier version took about 455 seconds for Chinese-to-English dubbing and 445 seconds for English-to-Chinese dubbing on two 30-second samples with short lines and silence. A separate 17.4-second video took about 238 seconds from submission to the first completed line on the first render and about 221 seconds on a rerender. These results predate model reuse. The current version adds model reuse, reference voice caching, and transfers between RAM and GPU. Use the same clips and the new timing logs to measure the speedup. The progress percentage indicates the current stage only. Timings from samples with short lines and silence cannot be extrapolated linearly to longer demos.

For acceptance testing, use one single-speaker video in Chinese and one in English, each at least 30 seconds long, and check the following.

- Product names, numbers, and units are translated correctly, and every line is included.
- The dub is intelligible and has no cut-off or overlapping speech. The narration matches the actions on screen.
- The finished video plays on a computer and an iPhone, and the subtitles can be opened.
- An Ollama disconnect, silent video, or damaged video produces an understandable error and preserves the original video.

Lip sync, separation of multiple speakers, and background audio separation are outside the scope of this release. Live caption mode produces only subtitles. Video dubbing replaces the entire original audio track, removing background music and sounds from actions on screen along with it.
