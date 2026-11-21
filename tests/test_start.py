import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("missing_cuda", [False, True])
@pytest.mark.parametrize("live", [False, True])
def test_launcher_sets_gpu_models_or_stops_before_startup(tmp_path, monkeypatch, missing_cuda, live):
    project = tmp_path / "video dubbing"
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    script = project / "start.sh"
    shutil.copy(Path(__file__).resolve().parents[1] / "start.sh", script)
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    if os.environ['TEST_MISSING_CUDA'] == '1': sys.exit(1)\n"
        "    print('/cuda/cublas/lib:/cuda/cudnn/lib')\n"
        "else:\n"
        "    keys = ('OLLAMA_MODEL', 'WHISPER_MODEL', 'ASR_DEVICE', 'COSYVOICE_PYTHON', 'LD_LIBRARY_PATH')\n"
        "    print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'env': {k: os.environ[k] for k in keys}}))\n"
    )
    python.chmod(0o755)
    monkeypatch.setenv("TEST_MISSING_CUDA", "1" if missing_cuda else "0")
    monkeypatch.setenv("OLLAMA_MODEL", "previous-model")
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    args = [str(script)] + (["--live"] if live else []) + ["--port", "7861"]
    result = subprocess.run(args, cwd=tmp_path,
                            capture_output=True, text=True, check=False, timeout=10)
    if missing_cuda:
        assert result.returncode == 1
        assert "GPU 依賴" in result.stderr
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        actual = json.loads(result.stdout)
        assert actual["cwd"] == str(project)
        assert actual["args"] == [str(project / ("live_bar.py" if live else "app.py")), "--host",
                                  "127.0.0.1" if live else "0.0.0.0", "--port", "7861"]
        assert actual["env"]["OLLAMA_MODEL"] == "gemma4:e2b"
        assert actual["env"]["WHISPER_MODEL"] == "medium"
        assert actual["env"]["ASR_DEVICE"] == "cuda"
        assert actual["env"]["COSYVOICE_PYTHON"] == str(project / ".venv-cosyvoice/bin/python")
        assert actual["env"]["LD_LIBRARY_PATH"] == "/cuda/cublas/lib:/cuda/cudnn/lib:/existing/lib"


@pytest.mark.parametrize(("mode", "override", "expected"), [
    ("nat", None, "http://172.28.32.1:11434"),
    ("mirrored", None, "http://127.0.0.1:11434"),
    ("nat", "http://my-ollama:11434", "http://my-ollama:11434"),
    ("unknown", None, ""),
])
def test_launcher_discovers_windows_ollama_unless_explicit(tmp_path, monkeypatch, mode, override, expected):
    script = tmp_path / "start.sh"
    shutil.copy(Path(__file__).resolve().parents[1] / "start.sh", script)
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/bash\nif [[ "$1" == "-c" ]]; then echo /cuda; else echo "${OLLAMA_URL:-}"; fi\n')
    wslinfo = tmp_path / "wslinfo"
    wslinfo.write_text(f"#!/bin/bash\necho {mode}\n")
    ip = tmp_path / "ip"
    ip.write_text("#!/bin/bash\necho 'default via 172.28.32.1 dev eth0'\n")
    for command in (python, wslinfo, ip):
        command.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    if override:
        monkeypatch.setenv("OLLAMA_URL", override)
    result = subprocess.run([str(script), "--live"], capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == expected
