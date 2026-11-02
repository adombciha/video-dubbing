# 地端 Video Dubbing

用中文錄製 Demo 並輸出英文配音，也支援英文影片轉中文配音。影片、辨識、翻譯與語音合成都在自己的電腦執行；首次安裝與模型下載需要網路。

第一版支援單人、5 分鐘以內、1 GiB 以下的影片。操作順序：上傳影片 → 選方向 → 辨識與翻譯 → 校對逐句文字 → 產生配音 → 下載 MP4 與兩份 SRT。中文翻譯要求繁體中文及臺灣用語，仍須校對產品名稱、數字與術語。

目前為原型：已在 Windows Ollama（`qwen3.5:4b`）、WSL 與 RTX 4050 Laptop GPU 環境完成中英雙向端到端配音，確認 MP4、原文字幕與譯文字幕均可產出。iPhone 同 Wi-Fi 網頁連線已確認；手機的上傳、播放與下載仍需完整驗收。啟動腳本現使用 `gemma4:e2b` 與 GPU `medium`，此組合尚待端到端驗證。

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
WHISPER_MODEL=medium python dubbing.py download-asr
```

若缺少 `venv`，先執行 `sudo apt install -y python3-venv`。`start.sh` 使用 faster-whisper 的多語言 `medium` 模型；請先完成上述權重下載。

GPU 辨識另需在主環境安裝 CUDA 12 的 cuBLAS 與 cuDNN 9：

```bash
.venv/bin/python -m pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'
```

`start.sh` 會自動加入這兩個套件的函式庫路徑。[faster-whisper GPU 安裝說明](https://github.com/SYSTRAN/faster-whisper#gpu)

## 2. 安裝 CosyVoice 3 獨立環境

CosyVoice 官方依賴與主介面版本不同，請放在另一個 Python 3.10 環境，不要安裝進 `.venv`。下例使用已安裝的 `uv` 建立環境。語音模型與 GPU 套件下載較大，安裝可能需要一段時間。

```bash
cd ~/video-dubbing
sudo apt install -y git build-essential sox libsox-dev
mkdir -p vendor models
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice
uv venv --python 3.10 .venv-cosyvoice
uv pip install --python .venv-cosyvoice/bin/python --index-strategy unsafe-first-match -r vendor/CosyVoice/requirements.txt
uv pip uninstall --python .venv-cosyvoice/bin/python deepspeed
```

以上使用 [CosyVoice 官方安裝說明](https://github.com/FunAudioLLM/CosyVoice#install) 的依賴。官方清單包含 PyTorch、Microsoft 及 PyPI 多個來源；`unsafe-first-match` 讓 uv 在優先來源沒有相容版本時繼續尋找下一來源，避免例如 `protobuf==4.25` 存在於 PyPI 卻被較優先的來源遮蔽。此選項僅對本次官方清單安裝使用，不設為全域預設；它會放寬 uv 的跨來源搜尋限制。[uv 官方說明](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)

若 PyTorch、CUDA 或 DeepSpeed 安裝失敗，先保留完整錯誤訊息並確認 `nvidia-smi`，不要強行將主環境的套件降版。此專案不需要另外安裝 vLLM。

此專案只使用標準 PyTorch 推理，不使用 DeepSpeed 訓練或加速。官方清單會安裝 DeepSpeed，因此安裝後將它從獨立環境移除，避免 Transformers 匯入它時要求未安裝的 CUDA 編譯工具。移除它不會移除 PyTorch 或停用 CUDA。若未來要執行上游的 DeepSpeed 訓練，請另設環境及 CUDA Toolkit。

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

安裝後可單獨檢查環境：

```bash
cd ~/video-dubbing
COSYVOICE_PYTHON="$PWD/.venv-cosyvoice/bin/python" WHISPER_MODEL=medium .venv/bin/python dubbing.py check --offline
```

預設原始碼與模型路徑已對應上面的位置。`check --offline` 檢查套件、Whisper 模型快取、CosyVoice 路徑與依賴匯入。下一步設定 Ollama 後，可在介面的「環境狀態與連線檢查」檢查模型清單；仍需實際產生配音，才能確認 GPU 推論與聲音品質。

## 3. Windows Ollama

程式會唯讀參考 `~/multi-agents/.ai-company/config.json` 的連線設定；`start.sh` 將翻譯模型固定為 `gemma4:e2b`。Windows Ollama 必須正在執行，且已安裝這個完整名稱的模型。

在 Windows 終端機確認模型清單：

```powershell
ollama list
```

模型必須已安裝且支援文字對話及 JSON 格式輸出；工具不會自動拉取或刪除 Ollama 模型。程式使用 `/api/tags` 查模型、`/api/chat` 翻譯，並在翻譯完成後要求卸載該模型，降低後續語音合成的顯存需求。

`gemma4:e2b` 與 `gemma4:latest` 是不同 tag。若 Windows IP 改變，可透過 `OLLAMA_URL` 覆寫連線位址。

翻譯請求優先關閉 thinking，降低短句翻譯的等待時間；模型不支援時會退回預設模式。每批都檢查段落數、編號、順序與非空譯文，格式錯誤會要求修正；每批最多額外呼叫一次，失敗仍會停止，不套用不完整結果。

## 4. 啟動與操作

完成安裝與模型下載後，只需執行：

```bash
cd ~/video-dubbing
./start.sh
```

腳本自動使用專案的虛擬環境、`gemma4:e2b` 翻譯、faster-whisper `medium` 的 CUDA / FP16 辨識及 CosyVoice 獨立環境，並設定 GPU 函式庫路徑。這些設定固定在 `start.sh`，會覆寫終端機先前的同名模型與裝置設定。腳本已啟用區網監聽；主程式參數可接在後面，例如 `./start.sh --port 7861`。

使用期間保持程式執行，按 `Ctrl+C` 停止。重開電腦後再次執行 `./start.sh`。

電腦開啟 `http://127.0.0.1:7860`。依序：

1. 上傳 MP4、MOV、M4V、MKV、WebM 或 AVI，選擇翻譯方向，按「辨識與翻譯」。
2. 校對原文與譯文。修改原文後，可按「依修正原文重新翻譯」，會覆蓋目前譯文。
3. 預設從影片第一段人聲取最多 12 秒作聲音參考；若片段不足 3 秒、雜音多或停頓過長，另外上傳 3–15 秒乾淨人聲。只使用你自己的聲音或有授權的素材。
4. 按「產生配音影片」。如果某段放不下，畫面會顯示編號、配音長度與可用秒數；縮短該段譯文後再生成。
5. 預覽或下載 MP4、譯文 SRT 與原文 SRT。SRT 是獨立檔案，未燒錄在畫面內。

配音對齊時保留原段落起點，先使用下一句之前的空檔，再視需要加速至最多約 1.2 倍。超時會停止輸出，不截斷或重疊人聲。譯文字幕時間依實際配音調整。聲音辨識與 CosyVoice 分別在子程序執行，程序退出後釋放其模型資源；介面一次只執行一個模型工作。

原影片不覆寫；原片副本、逐句文字、配音與成品保存在 `data/jobs/job-*/`。重試配音會建立新的 `render-*` 目錄，保留先前結果。這些資料不會自動清除；關閉程式後可手動刪除不需要的工作資料夾。

## 5. iPhone 同 Wi-Fi 使用

`start.sh` 已啟用區網監聽：

```bash
./start.sh
```

Safari 開啟 `http://<Windows 的區網 IPv4>:7860`，從相簿或檔案選擇已錄好的影片。影片運算仍在電腦，不在手機。下載後可從 Safari 的下載項目開啟，再依 iOS 提供的選項儲存或分享。

WSL 的區網連線方式依 Windows / WSL 網路模式而異；若 Windows 本機可開而 iPhone 無法開啟，需檢查 WSL mirrored networking、Windows 防火牆或連接埠轉送，不能只靠 `--host 0.0.0.0` 保證可達。

手機網址應使用 Windows 正在連線的 Wi-Fi 或乙太網路 IPv4。`127.0.0.1` 在手機上指向手機自己；`0.0.0.0` 是伺服器的監聽設定。先在 Windows PowerShell 查詢目前有預設閘道的介面：

```powershell
Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object InterfaceAlias, @{Name='IPv4';Expression={$_.IPv4Address.IPAddress}}
```

若仍無法連線，在正常 WSL 終端機查詢網路模式與 WSL IP；以下指令都只讀取設定：

```bash
wslinfo --networking-mode
hostname -I
```

NAT 模式下，Windows 能透過 `localhost` 開啟 WSL 網頁，不代表手機能連入；通常需要 Windows 連接埠轉送與防火牆規則。Mirrored 模式可直接接受區網連線，但仍需檢查防火牆。先確認實際模式、IP 與瀏覽器錯誤，再針對 TCP 7860 處理；不要直接停用整個防火牆。[Microsoft WSL 網路說明](https://learn.microsoft.com/en-us/windows/wsl/networking)

如需登入，可設定 `DUBBING_USERNAME` 與 `DUBBING_PASSWORD`；不要使用公開分享連結。啟動時一律 `share=False`。可提供自己信任的憑證啟用 HTTPS：

```bash
./start.sh --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

目前使用原生相機錄影後上傳，不要求 Safari 在網頁內存取相機，也未包裝成原生 App 或 PWA。

## 設定

下表列出直接執行 Python 時的預設值。日常使用 `start.sh` 時，翻譯模型為 `gemma4:e2b`、辨識模型為 `medium`、辨識裝置為 `cuda`，CosyVoice Python 指向專案的 `.venv-cosyvoice/bin/python`。

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

目前實測兩支 30 秒範例（短句與靜音），中文轉英文約 455 秒、英文轉中文約 445 秒。每次配音都會重新載入模型，進度百分比代表處理階段，不能據此估算剩餘時間。這是該次環境的結果，尚未做效能最佳化，也不能據此線性推算較長 Demo 的耗時。

實際驗收請各用一支 30 秒以上的中文及英文單人影片，確認：

- 產品名稱、數字、單位沒有翻錯，且沒有漏句。
- 配音可辨識、無截尾、無重疊，操作畫面與敘述對得上。
- 成品可在電腦與 iPhone 播放，字幕可以開啟。
- Ollama 斷線、無聲或損壞影片時有可理解的錯誤，原始影片仍在。

本版不含嘴型同步、多人說話者分離、即時翻譯及背景音分離。配音替換整條原音軌，因此原有背景音樂及操作聲也會移除。
