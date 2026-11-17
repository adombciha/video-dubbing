import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import live_bar as live
from dubbing import DubbingError, Settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DUBBING_USERNAME", raising=False)
    monkeypatch.delenv("DUBBING_PASSWORD", raising=False)
    engines = []

    class Engine:
        def __init__(self, settings, source, target):
            self.source, self.target = source, target
            self.open = Mock()
            self.close = Mock()
            self.process = Mock(return_value={"type": "result", "source": "Hello", "text": "你好", "seconds": 0.1})
            engines.append(self)

    monkeypatch.setattr(live, "LiveEngine", Engine)
    with TestClient(live.create_app(Settings())) as app:
        yield app, engines


def connect(app):
    return app.websocket_connect("/ws/live", headers={"origin": "http://testserver"})


def test_audio_decoding_checks_size_and_keeps_pcm_values():
    samples = np.tile(np.array([-32768, 0, 16384, 32767], dtype="<i2"), 100)
    audio = live.decode_audio(samples.tobytes())
    np.testing.assert_allclose(audio[:4], [-1, 0, 0.5, 32767 / 32768])
    for data in (None, b"", b"x" * 321, bytes(live.MAX_PCM_BYTES + 2)):
        with pytest.raises(DubbingError):
            live.decode_audio(data)


@pytest.mark.parametrize("source,target", list(product(live.LANGUAGES, repeat=2)))
def test_three_language_pairs_stream_and_release_engine(client, source, target):
    app, engines = client
    with connect(app) as socket:
        socket.send_json({"source": source, "target": target})
        assert socket.receive_json()["type"] == "loading"
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(np.full(1600, 1000, dtype="<i2").tobytes())
        assert socket.receive_json()["text"] == "你好"
        socket.send_text("stop")
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()
    assert (engines[0].source, engines[0].target) == (source, target)
    engines[0].open.assert_called_once()
    engines[0].close.assert_called_once()


def test_invalid_language_and_pcm_do_not_reach_model(client):
    app, engines = client
    with connect(app) as socket:
        socket.send_json({"source": "xx", "target": "zh"})
        assert socket.receive_json()["type"] == "error"
    assert not engines
    with connect(app) as socket:
        socket.send_json({"source": "en", "target": "ja"})
        socket.receive_json()
        socket.receive_json()
        socket.send_bytes(b"invalid")
        assert socket.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()
    engines[0].process.assert_not_called()
    engines[0].close.assert_called_once()


def test_only_one_capture_can_use_models_and_disconnect_releases_it(client):
    app, engines = client
    with connect(app) as first:
        first.send_json({"source": "ja", "target": "zh"})
        first.receive_json()
        first.receive_json()
        with connect(app) as second:
            assert "已有字幕工作" in second.receive_json()["message"]
        assert len(engines) == 1
    with connect(app) as again:
        again.send_json({"source": "en", "target": "zh"})
        again.receive_json()
        assert again.receive_json()["type"] == "ready"
    assert len(engines) == 2
    assert all(engine.close.call_count == 1 for engine in engines)


def test_origin_and_optional_login_protect_audio_endpoints(client, monkeypatch):
    app, engines = client
    with pytest.raises(WebSocketDisconnect), app.websocket_connect("/ws/live", headers={"origin": "https://unrelated.test"}):
        pass
    assert not engines
    monkeypatch.setenv("DUBBING_USERNAME", "demo")
    monkeypatch.setenv("DUBBING_PASSWORD", "local-password")
    with TestClient(live.create_app(Settings())) as protected:
        assert protected.get("/").status_code == 401
        assert protected.get("/static/app.js").status_code == 401
        assert protected.get("/", auth=("demo", "local-password")).status_code == 200
        with pytest.raises(WebSocketDisconnect), connect(protected):
            pass
        with protected.websocket_connect("/ws/live", headers={
            "origin": "http://testserver", "authorization": "Basic ZGVtbzpsb2NhbC1wYXNzd29yZA==",
        }) as socket:
            socket.send_json({"source": "en", "target": "ja"})
            socket.receive_json()
            assert socket.receive_json()["type"] == "ready"


@pytest.mark.parametrize("failure", ["open", "process"])
def test_inference_failures_report_error_and_cleanup(client, failure, monkeypatch):
    app, engines = client
    original_factory = live.LiveEngine

    def failing_engine(*args):
        engine = original_factory(*args)
        getattr(engine, failure).side_effect = DubbingError("GPU or Ollama unavailable")
        return engine

    monkeypatch.setattr(live, "LiveEngine", failing_engine)
    with connect(app) as socket:
        socket.send_json({"source": "en", "target": "zh"})
        socket.receive_json()
        if failure == "process":
            assert socket.receive_json()["type"] == "ready"
            socket.send_bytes(bytes(3200))
        assert socket.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()
    engines[0].close.assert_called_once()


def test_engine_reuses_medium_gpu_and_keeps_japanese_translation(monkeypatch):
    model = SimpleNamespace(model=SimpleNamespace(unload_model=Mock()), transcribe=Mock(
        side_effect=lambda *a, **k: (iter([SimpleNamespace(text="こんにちは")]), None)))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    request = Mock(return_value={"message": {"content": json.dumps({"text": "你好"})}})
    monkeypatch.setattr(live, "ollama_request", request)
    engine = live.LiveEngine(Settings(whisper_model="medium", asr_device="cuda"), "ja", "zh")
    engine.open()
    assert engine.process(np.ones(1600, dtype=np.float32))["text"] == "你好"
    engine.process(np.ones(1600, dtype=np.float32))
    payload = request.call_args.args[2]
    assert "繁體中文" in payload["messages"][0]["content"]
    assert payload["keep_alive"] == "3m" and payload["think"] is False
    assert payload["options"]["num_ctx"] == 2048
    assert factory.call_count == 1 and factory.call_args.kwargs["compute_type"] == "int8_float16"
    assert all(call.kwargs["language"] == "ja" for call in model.transcribe.call_args_list)
    engine.close()
    model.model.unload_model.assert_called_once()
    assert request.call_args.args[2]["keep_alive"] == 0


@pytest.mark.parametrize("target", live.LANGUAGES)
def test_live_translation_targets_are_explicit_and_bad_json_is_rejected(monkeypatch, target):
    request = Mock(return_value={"message": {"content": json.dumps({"text": "translated"})}})
    monkeypatch.setattr(live, "ollama_request", request)
    engine = live.LiveEngine(Settings(), "ja" if target != "ja" else "en", target)
    assert engine.translate("example") == "translated"
    assert live.LANGUAGES[target] in request.call_args.args[2]["messages"][0]["content"]
    request.return_value = {"message": {"content": '{"text":""}'}}
    with pytest.raises(DubbingError, match="格式錯誤"):
        engine.translate("example")


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js required for audio frontend checks")
def test_audio_frontend_self_checks():
    result = subprocess.run(["node", "--test", str(Path(__file__).with_name("live_bar.test.mjs"))],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
