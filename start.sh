#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python="$PWD/.venv/bin/python"

export OLLAMA_MODEL="gemma4:e2b"
export WHISPER_MODEL="medium"
export ASR_DEVICE="cuda"
export COSYVOICE_PYTHON="$PWD/.venv-cosyvoice/bin/python"

if ! cuda_libs=$("$python" -c 'import nvidia.cublas, nvidia.cudnn; print(nvidia.cublas.__path__[0]+"/lib:"+nvidia.cudnn.__path__[0]+"/lib")'); then
    printf '%s\n' '無法載入 GPU 依賴；請依 README 安裝主環境的 cuBLAS 與 cuDNN 9。' >&2
    exit 1
fi
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "${1:-}" == "--live" ]]; then
    shift
    exec "$python" "$PWD/live_bar.py" --host 127.0.0.1 "$@"
fi

exec "$python" "$PWD/app.py" --host 0.0.0.0 "$@"
