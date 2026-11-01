# 地端 Video Dubbing

用中文錄製 Demo 並輸出英文配音，也支援英文影片轉中文配音。影片、辨識、翻譯與語音合成都在自己的電腦執行；首次安裝與模型下載需要網路。

第一版支援單人、5 分鐘以內、1 GiB 以下的影片。操作順序：上傳影片 → 選方向 → 辨識與翻譯 → 校對逐句文字 → 產生配音 → 下載 MP4 與兩份 SRT。中文翻譯要求繁體中文及臺灣用語，仍須校對產品名稱、數字與術語。

目前為原型：已驗證本機中英文 Whisper CPU 辨識、流程測試與實際 FFmpeg 合成。CosyVoice 語音合成、Windows Ollama 翻譯及 iPhone 實機操作仍待端到端驗證。

## 1. 安裝主環境（WSL / Ubuntu）

取得專案；已有工作目錄時略過這一步：

```bash
git clone https://github.com/adombciha/video-dubbing.git ~/video-dubbing
```

在 WSL 終端機執行，與 `~/multi-agents` 使用不同的虛擬環境：

```bash
cd ~/video-dubbing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
sudo apt update
sudo apt install -y ffmpeg
python dubbing.py download-asr
```

若缺少 `venv`，先執行 `sudo apt install -y python3-venv`。執行時不會自動下載 Whisper 模型；請先完成 `download-asr`。預設為 `small`，先確認短片效果，再依硬體決定是否改用較大模型。

## 2. 安裝 CosyVoice 3 獨立環境

CosyVoice 官方依賴與主介面版本不同，請放在另一個 Python 3.10 環境，不要安裝進 `.venv`。下例使用已安裝的 `uv` 建立環境。語音模型與 GPU 套件下載較大，安裝可能需要一段時間。

```bash
cd ~/video-dubbing
sudo apt install -y git build-essential sox libsox-dev
mkdir -p vendor models
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice
uv venv --python 3.10 .venv-cosyvoice
uv pip install --python .venv-cosyvoice/bin/python --index-strategy unsafe-first-match -r vendor/CosyVoice/requirements.txt
```

以上使用 [CosyVoice 官方安裝說明](https://github.com/FunAudioLLM/CosyVoice#install) 的依賴。官方清單包含 PyTorch、Microsoft 及 PyPI 多個來源；`unsafe-first-match` 讓 uv 在優先來源沒有相容版本時繼續尋找下一來源，避免例如 `protobuf==4.25` 存在於 PyPI 卻被較優先的來源遮蔽。此選項僅對本次官方清單安裝使用，不設為全域預設；它會放寬 uv 的跨來源搜尋限制。[uv 官方說明](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)

若 PyTorch、CUDA 或 DeepSpeed 安裝失敗，先保留完整錯誤訊息並確認 `nvidia-smi`，不要強行將主環境的套件降版。此專案不需要另外安裝 vLLM。

下載模型：

```bash
.venv-cosyvoice/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='models/Fun-CosyVoice3-0.5B',
)
PY
```

每次啟動主程式前設定執行環境：

```bash
cd ~/video-dubbing
source .venv/bin/activate
export COSYVOICE_PYTHON="$PWD/.venv-cosyvoice/bin/python"
python dubbing.py check
```

預設原始碼與模型路徑已對應上面的位置。`check` 檢查套件、Whisper 模型快取、CosyVoice 路徑與依賴匯入，以及 Ollama 模型清單；仍需實際產生配音，才能確認 GPU 推論與聲音品質。

## 3. Windows Ollama

程式會唯讀參考 `~/multi-agents/.ai-company/config.json` 的 `ollama_url` 與 `ollama_model`。沒有該檔案時，預設連到 `http://localhost:11434`，模型為 `gemma4:latest`。不會修改或啟動 `multi-agents`。

若 Windows IP 或模型名稱改變，可用環境變數覆寫：

```bash
export OLLAMA_URL="http://172.17.144.1:11434"
export OLLAMA_MODEL="gemma4:latest"
python dubbing.py check
```

模型必須已安裝且支援文字對話及 JSON 格式輸出；工具不會自動拉取或刪除 Ollama 模型。程式使用 `/api/tags` 查模型、`/api/chat` 翻譯，並在翻譯完成後要求卸載該模型，降低後續語音合成的顯存需求。

## 4. 啟動與操作

```bash
cd ~/video-dubbing
source .venv/bin/activate
export COSYVOICE_PYTHON="$PWD/.venv-cosyvoice/bin/python"
python app.py
```

電腦開啟 `http://127.0.0.1:7860`。依序：

1. 上傳 MP4、MOV、M4V、MKV、WebM 或 AVI，選擇翻譯方向，按「辨識與翻譯」。
2. 校對原文與譯文。修改原文後，可按「依修正原文重新翻譯」，會覆蓋目前譯文。
3. 預設從影片第一段人聲取最多 12 秒作聲音參考；若片段不足 3 秒、雜音多或停頓過長，另外上傳 3–15 秒乾淨人聲。只使用你自己的聲音或有授權的素材。
4. 按「產生配音影片」。如果某段放不下，畫面會顯示編號、配音長度與可用秒數；縮短該段譯文後再生成。
5. 預覽或下載 MP4、譯文 SRT 與原文 SRT。SRT 是獨立檔案，未燒錄在畫面內。

配音對齊時保留原段落起點，先使用下一句之前的空檔，再視需要加速至最多約 1.2 倍。超時會停止輸出，不截斷或重疊人聲。譯文字幕時間依實際配音調整。聲音辨識與 CosyVoice 分別在子程序執行，程序退出後釋放其模型資源；介面一次只執行一個模型工作。

原影片不覆寫；原片副本、逐句文字、配音與成品保存在 `data/jobs/job-*/`。重試配音會建立新的 `render-*` 目錄，保留先前結果。這些資料不會自動清除；關閉程式後可手動刪除不需要的工作資料夾。

## 5. iPhone 同 Wi-Fi 使用

電腦先確認可用，再啟動區網監聽：

```bash
python app.py --host 0.0.0.0
```

Safari 開啟 `http://<Windows 的區網 IPv4>:7860`，從相簿或檔案選擇已錄好的影片。影片運算仍在電腦，不在手機。下載後可從 Safari 的下載項目開啟，再依 iOS 提供的選項儲存或分享。

WSL 的區網連線方式依 Windows / WSL 網路模式而異；若 Windows 本機可開而 iPhone 無法開啟，需檢查 WSL mirrored networking、Windows 防火牆或連接埠轉送，不能只靠 `--host 0.0.0.0` 保證可達。

如需登入，可設定 `DUBBING_USERNAME` 與 `DUBBING_PASSWORD`；不要使用公開分享連結。啟動時一律 `share=False`。可提供自己信任的憑證啟用 HTTPS：

```bash
python app.py --host 0.0.0.0 --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

目前使用原生相機錄影後上傳，不要求 Safari 在網頁內存取相機，也未包裝成原生 App 或 PWA。

## 設定

| 環境變數 | 預設 / 用途 |
| --- | --- |
| `OLLAMA_URL` | 沿用 `multi-agents`，否則 `http://localhost:11434` |
| `OLLAMA_MODEL` | 沿用 `multi-agents`，否則 `gemma4:latest` |
| `WHISPER_MODEL` | `small`，也可使用本機模型路徑；切換後先執行 `download-asr` |
| `ASR_DEVICE` | `auto`，也可設 `cpu` 或 `cuda`；CPU 使用 int8，CUDA 使用 float16 |
| `COSYVOICE_PYTHON` | 主程式 Python；依上述步驟安裝後需指向 `.venv-cosyvoice/bin/python` |
| `COSYVOICE_ROOT` | `vendor/CosyVoice` 的絕對路徑 |
| `COSYVOICE_MODEL_DIR` | `models/Fun-CosyVoice3-0.5B` 的絕對路徑 |
| `FFMPEG` / `FFPROBE` | PATH 中的 `ffmpeg` / `ffprobe`，可設執行檔路徑 |
| `DUBBING_OUTPUT_DIR` | 專案下的 `data/jobs` |
| `DUBBING_USERNAME` / `DUBBING_PASSWORD` | 選填，必須成對設定 |

`auto` 只根據 CUDA 裝置是否可見來選擇 ASR 裝置；若 cuDNN / CUDA 函式庫不相容，會回報錯誤，這時可先使用 `ASR_DEVICE=cpu` 驗證流程。Ollama 翻譯品質、聲音相似度、口音與處理速度取決於實際模型及硬體，不保證完全保留原情緒或聲線。

辨識 worker 直接讀取 FFmpeg 已轉換的 16 kHz 單聲道 PCM16 WAV，再將標準化音訊樣本傳給 Whisper，避免重複解碼及 PyAV 版本間的音訊開啟參數差異。

## 驗證

```bash
python -m pytest -q
python dubbing.py check
```

單元測試涵蓋翻譯缺段、錯誤編號、雙向翻譯提示、文字校對、時間重疊、語音超時、空白聲音、字幕與原檔保留。FFmpeg 整合測試實際產生影片、調速、合成與檢查時長，缺少 FFmpeg / ffprobe 時會明確略過；測試中使用合成音訊，不代表模型品質驗收。

實際驗收請各用一支 30 秒以上的中文及英文單人影片，確認：

- 產品名稱、數字、單位沒有翻錯，且沒有漏句。
- 配音可辨識、無截尾、無重疊，操作畫面與敘述對得上。
- 成品可在電腦與 iPhone 播放，字幕可以開啟。
- Ollama 斷線、無聲或損壞影片時有可理解的錯誤，原始影片仍在。

本版不含嘴型同步、多人說話者分離、即時翻譯及背景音分離。配音替換整條原音軌，因此原有背景音樂及操作聲也會移除。
