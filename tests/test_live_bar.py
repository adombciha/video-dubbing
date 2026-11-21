import asyncio
import json
import shutil
import socket
import subprocess
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

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


def test_uvicorn_accepts_real_websocket_upgrade(client):
    app, engines = client

    async def check_upgrade():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.setblocking(False)
            host, port = listener.getsockname()
            server = live.uvicorn.Server(live.uvicorn.Config(
                app.app, loop="asyncio", log_level="warning", timeout_graceful_shutdown=2,
            ))
            task = asyncio.create_task(server.serve(sockets=[listener]))
            writer = None
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5)
                writer.write((
                    f"GET /ws/live HTTP/1.1\r\nHost: {host}:{port}\r\n"
                    f"Origin: http://{host}:{port}\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                    "Sec-WebSocket-Version: 13\r\n\r\n"
                ).encode("ascii"))
                await writer.drain()
                response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
                assert response.startswith(b"HTTP/1.1 101 "), response.decode("ascii", errors="replace")
                headers = response.decode("ascii").split("\r\n")
                accept = next(line.split(":", 1)[1].strip() for line in headers
                              if line.lower().startswith("sec-websocket-accept:"))
                assert accept == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
            finally:
                if writer is not None:
                    writer.close()
                    await writer.wait_closed()
                server.should_exit = True
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(check_upgrade())
    assert not engines


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


def test_client_disconnect_during_close_does_not_escape_asgi(client):
    app, engines = client
    messages = iter([
        {"type": "websocket.connect"},
        {"type": "websocket.receive", "text": '{"source":"en","target":"zh"}'},
        {"type": "websocket.disconnect", "code": 1001},
    ])

    async def receive():
        return next(messages)

    async def send(message):
        if message["type"] == "websocket.close":
            raise OSError("Client connection already closed")

    asyncio.run(app.app({
        "type": "websocket", "scheme": "ws", "path": "/ws/live", "query_string": b"",
        "headers": [(b"host", b"testserver"), (b"origin", b"http://testserver")],
    }, receive, send))
    engines[0].close.assert_called_once()


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


@pytest.mark.parametrize("actual_context", [None, 0, "32768", 2048, 32768])
def test_engine_reuses_medium_gpu_and_keeps_japanese_translation(monkeypatch, actual_context):
    model = SimpleNamespace(model=SimpleNamespace(unload_model=Mock()), transcribe=Mock(
        side_effect=lambda *a, **k: (iter([SimpleNamespace(text="こんにちは")]), None)))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    def response(settings, endpoint, payload=None, timeout=180):
        if endpoint == "/api/ps":
            return {"models": [{"name": settings.ollama_model, "context_length": actual_context},
                               {"name": "another-model", "context_length": 65536}]}
        return {"message": {"content": json.dumps({"text": "你好"})}}

    request = Mock(side_effect=response)
    monkeypatch.setattr(live, "ollama_request", request)
    engine = live.LiveEngine(Settings(whisper_model="medium", asr_device="cuda"), "ja", "zh")
    engine.open()
    assert engine.process(np.ones(1600, dtype=np.float32))["text"] == "你好"
    engine.process(np.ones(1600, dtype=np.float32))
    payload = request.call_args.args[2]
    assert "繁體中文" in payload["messages"][0]["content"]
    assert payload["keep_alive"] == "3m" and payload["think"] is False
    expected_context = actual_context if type(actual_context) is int and actual_context > 0 else 2048
    assert payload["options"]["num_ctx"] == expected_context
    preload = next(call.args[2] for call in request.call_args_list if call.args[1] == "/api/generate")
    assert preload["options"]["num_ctx"] == expected_context
    assert factory.call_count == 1 and factory.call_args.kwargs["compute_type"] == "int8_float16"
    assert all(call.kwargs["language"] == "ja" for call in model.transcribe.call_args_list)
    for call in model.transcribe.call_args_list:
        assert call.kwargs["beam_size"] == call.kwargs["best_of"] == 1
        assert call.kwargs["temperature"] == 0
        assert call.kwargs["without_timestamps"] is True
    engine.close()
    model.model.unload_model.assert_called_once()
    assert request.call_args.args[2]["keep_alive"] == 0


def test_translation_tracks_context_changes_from_other_clients(monkeypatch):
    contexts = iter([32768, 4096])
    def response(settings, endpoint, payload=None, timeout=180):
        if endpoint == "/api/ps":
            return {"models": [{"name": settings.ollama_model, "context_length": next(contexts)}]}
        return {"message": {"content": '{"text":"你好"}'}}

    request = Mock(side_effect=response)
    monkeypatch.setattr(live, "ollama_request", request)
    engine = live.LiveEngine(Settings(), "en", "zh")
    for context in (32768, 4096):
        assert engine.translate("Hello") == "你好"
        assert request.call_args.args[2]["options"]["num_ctx"] == context


def test_live_timings_include_lazy_asr_iteration_and_translation(monkeypatch):
    clock = [10.0]

    def transcribe(*args, **kwargs):
        def segments():
            clock[0] += 0.75
            yield SimpleNamespace(text="Hello")
        return segments(), None

    def translate(text):
        clock[0] += 0.25
        return "你好"

    monkeypatch.setattr(live.time, "perf_counter", lambda: clock[0])
    engine = live.LiveEngine(Settings(), "en", "zh")
    engine.whisper = SimpleNamespace(transcribe=transcribe)
    engine.translate = translate
    result = engine.process(np.ones(live.RATE * 3, dtype=np.float32))
    assert result["asr_seconds"] == 0.75
    assert result["translation_seconds"] == 0.25
    assert result["seconds"] == 1
    assert result["audio_seconds"] == 3


def test_silence_reports_timings_without_requesting_translation(monkeypatch):
    engine = live.LiveEngine(Settings(), "en", "zh")
    engine.whisper = SimpleNamespace(transcribe=Mock(return_value=(iter([]), None)))
    engine.translate = Mock()
    result = engine.process(np.zeros(live.RATE * 3, dtype=np.float32))
    assert result["source"] == result["text"] == ""
    assert result["translation_seconds"] == 0
    engine.translate.assert_not_called()


@pytest.mark.parametrize("retry_fails", [False, True])
def test_repeat_abort_retries_once_and_does_not_lose_the_next_segment(monkeypatch, retry_fails):
    inputs = iter(["Hello", "Goodbye"])
    requests = []

    def response(settings, endpoint, payload=None, timeout=180):
        if endpoint == "/api/ps":
            return {"models": []}
        requests.append((deepcopy(payload), timeout))
        if len(requests) == 1 or (len(requests) == 2 and retry_fails):
            raise DubbingError("prediction aborted, token repeat limit reached") from HTTPError(
                settings.ollama_url, 500, "Internal Server Error", {}, None,
            )
        content = "你好" if len(requests) == 2 else '{"text":"再見"}'
        return {"message": {"content": content}, "done_reason": "stop"}

    monkeypatch.setattr(live, "ollama_request", response)
    engine = live.LiveEngine(Settings(), "en", "zh")
    engine.whisper = SimpleNamespace(transcribe=lambda *args, **kwargs: (
        [SimpleNamespace(text=next(inputs))], None,
    ))
    audio = np.ones(live.RATE, dtype=np.float32)
    result = engine.process(audio)
    assert result["source"] == "Hello"
    assert result["translation_failed"] is retry_fails
    assert result["text"] == ("" if retry_fails else "你好")
    assert engine.process(audio)["text"] == "再見"
    first, retry, next_segment = requests
    assert "format" in first[0] and "format" in next_segment[0]
    assert "format" not in retry[0]
    assert retry[0]["options"]["temperature"] > first[0]["options"]["temperature"]
    assert retry[0]["messages"][1] == first[0]["messages"][1]
    assert retry[1] < first[1]


def test_unrelated_500_reports_the_original_failure(monkeypatch):
    error = HTTPError("http://localhost:11434/api/chat", 500, "Internal Server Error", {}, None)

    def response(settings, endpoint, payload=None, timeout=180):
        if endpoint == "/api/ps":
            return {"models": []}
        raise DubbingError("model runner stopped") from error

    request = Mock(side_effect=response)
    monkeypatch.setattr(live, "ollama_request", request)
    engine = live.LiveEngine(Settings(), "en", "zh")
    with pytest.raises(DubbingError, match="model runner stopped"):
        engine.translate("Hello")
    assert [call.args[1] for call in request.call_args_list] == ["/api/ps", "/api/chat"]


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
