import io
import json
import sys
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import cosyvoice_worker
import dubbing


@pytest.fixture
def cosyvoice_modules(monkeypatch):
    speech = Mock()
    torch = SimpleNamespace(cat=Mock(return_value=speech), __version__="test", cuda=SimpleNamespace(is_available=lambda: False))
    audio = SimpleNamespace(functional=SimpleNamespace(resample=Mock(return_value=speech)), save=Mock())
    chunks = [Mock(), Mock()]
    model = SimpleNamespace(sample_rate=22050, inference_cross_lingual=Mock(return_value=iter([
        {"tts_speech": chunks[0]}, {"tts_speech": chunks[1]},
    ])))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", audio)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", SimpleNamespace(AutoModel=factory))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    return SimpleNamespace(torch=torch, audio=audio, model=model, factory=factory, chunks=chunks)


def test_cosyvoice_worker_uses_cross_language_api_and_combines_all_chunks(monkeypatch, tmp_path, cosyvoice_modules):
    request = tmp_path / "request.json"
    reference = str(tmp_path / "reference.wav")
    request.write_text(json.dumps({"reference": reference, "segments": [{"id": 1, "target": "Hello"}]}))
    monkeypatch.setattr(sys, "argv", ["cosyvoice_worker.py", str(request), "--root", str(tmp_path), "--model", str(tmp_path)])
    cosyvoice_worker.main()
    modules = cosyvoice_modules
    modules.model.inference_cross_lingual.assert_called_once_with(
        "You are a helpful assistant.<|endofprompt|>Hello", reference, stream=False,
    )
    modules.torch.cat.assert_called_once_with([chunk.cpu() for chunk in modules.chunks], dim=1)
    modules.audio.functional.resample.assert_called_once_with(modules.torch.cat.return_value, 22050, 24000)
    modules.audio.save.assert_called_once_with(str(tmp_path / "speech-1.wav"), modules.audio.functional.resample.return_value,
                                               24000, encoding="PCM_S", bits_per_sample=16)


def test_cosyvoice_check_does_not_load_weights(monkeypatch, tmp_path, cosyvoice_modules, capsys):
    monkeypatch.setattr(sys, "argv", ["cosyvoice_worker.py", "--root", str(tmp_path), "--model", str(tmp_path), "--check"])
    cosyvoice_worker.main()
    assert "imports OK" in capsys.readouterr().out
    cosyvoice_modules.factory.assert_not_called()


def test_cosyvoice_empty_generation_is_an_error(monkeypatch, tmp_path, cosyvoice_modules):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"reference": "reference.wav", "segments": [{"id": 1, "target": "Hello"}]}))
    cosyvoice_modules.model.inference_cross_lingual.return_value = iter([])
    monkeypatch.setattr(sys, "argv", ["cosyvoice_worker.py", str(request), "--root", str(tmp_path), "--model", str(tmp_path)])
    with pytest.raises(RuntimeError, match="no audio"):
        cosyvoice_worker.main()


@pytest.mark.parametrize("cuda", [False, True])
def test_cosyvoice_session_reuses_model_reference_and_parks_between_requests(monkeypatch, tmp_path, cosyvoice_modules, capsys, cuda):
    modules = cosyvoice_modules
    parts = SimpleNamespace(llm=Mock(), flow=Mock(), hift=Mock(), device="cuda" if cuda else "cpu")
    feature = Mock()
    modules.model.model = parts
    modules.model.frontend = SimpleNamespace(spk2info={}, speech_tokenizer_session=object())

    def cache_reference(*args):
        modules.model.frontend.spk2info[args[2]] = {"embedding": feature}

    modules.model.add_zero_shot_spk = Mock(side_effect=cache_reference)
    modules.model.inference_cross_lingual.side_effect = lambda *a, **k: iter([{"tts_speech": modules.chunks[0]}])
    modules.torch.cuda = SimpleNamespace(is_available=lambda: cuda, synchronize=Mock(), empty_cache=Mock())
    modules.torch.is_tensor = lambda value: isinstance(value, Mock)
    original_reply = cosyvoice_worker.reply

    def check_reply(stream, status, **details):
        if status == "done" and cuda:
            for part in (parts.llm, parts.flow, parts.hift):
                assert part.to.call_args.args == ("cpu",)
            assert modules.torch.cuda.empty_cache.call_count == modules.model.inference_cross_lingual.call_count
        original_reply(stream, status, **details)

    monkeypatch.setattr(cosyvoice_worker, "reply", check_reply)
    requests = []
    for index in range(2):
        folder = tmp_path / str(index)
        folder.mkdir()
        path = folder / "request.json"
        path.write_text(json.dumps({"reference": "same-reference.wav", "segments": [{"id": 1, "target": "Hello"}]}))
        requests.append(path)
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps({"request": str(path)}) + "\n" for path in requests)))
    timings = tmp_path / "timing.jsonl"
    monkeypatch.setattr(sys, "argv", ["cosyvoice_worker.py", "--root", str(tmp_path), "--model", str(tmp_path),
                                     "--serve", "--timings", str(timings)])
    cosyvoice_worker.main()
    replies = [json.loads(line.removeprefix(cosyvoice_worker.RESPONSE_PREFIX)) for line in capsys.readouterr().out.splitlines()]
    assert [item["status"] for item in replies] == ["ready", "done", "done"]
    assert [item["request"] for item in replies[1:]] == [str(path) for path in requests]
    modules.factory.assert_called_once()
    modules.model.add_zero_shot_spk.assert_called_once_with("", "same-reference.wav", cosyvoice_worker.REFERENCE_ID)
    assert not hasattr(modules.model.frontend, "speech_tokenizer_session")
    for part in (parts.llm, parts.flow, parts.hift):
        assert [call.args[0] for call in part.to.call_args_list] == (["cpu", "cuda", "cpu"] if cuda else [])
    assert modules.torch.cuda.empty_cache.call_count == (2 if cuda else 0)
    assert all(call.kwargs["zero_shot_spk_id"] == cosyvoice_worker.REFERENCE_ID
               for call in modules.model.inference_cross_lingual.call_args_list)
    phases = [json.loads(line)["phase"] for line in timings.read_text().splitlines()]
    assert phases.count("model_load") == phases.count("reference") == 1
    assert phases.count("resume_gpu") == (1 if cuda else 0)
    assert phases.count("speech") == 2
    assert phases.count("park_cpu") == (2 if cuda else 0)


def test_cosyvoice_startup_failure_reports_protocol_error(monkeypatch, tmp_path, cosyvoice_modules, capsys):
    cosyvoice_modules.factory.side_effect = RuntimeError("weights failed to load")
    monkeypatch.setattr(sys, "argv", ["cosyvoice_worker.py", "--root", str(tmp_path), "--model", str(tmp_path), "--serve"])
    with pytest.raises(RuntimeError, match="weights failed"):
        cosyvoice_worker.main()
    result = capsys.readouterr()
    reply = json.loads(result.out.removeprefix(cosyvoice_worker.RESPONSE_PREFIX))
    assert reply == {"status": "error", "error": "weights failed to load"}


def test_whisper_worker_uses_original_language_and_cached_model(monkeypatch, tmp_path):
    model = SimpleNamespace(transcribe=Mock(return_value=(iter([
        SimpleNamespace(start=0, end=1, text=" 你好 "), SimpleNamespace(start=1, end=2, text=" "),
    ]), None)))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as target:
        target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        target.writeframes(np.array([-32768, 0, 16384, 32767], dtype="<i2").tobytes())
    output = tmp_path / "transcript.json"
    dubbing.transcribe_worker(SimpleNamespace(device="auto", model="small", audio=str(audio), language="zh", output=str(output)))
    assert factory.call_args.kwargs["device"] == "cpu"
    assert factory.call_args.kwargs["compute_type"] == "int8"
    assert factory.call_args.kwargs["local_files_only"] is True
    model.transcribe.assert_called_once()
    samples = model.transcribe.call_args.args[0]
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, np.array([-1, 0, 0.5, 32767 / 32768], dtype=np.float32))
    assert model.transcribe.call_args.kwargs == {"language": "zh", "task": "transcribe", "vad_filter": True, "word_timestamps": True}
    rows = json.loads(output.read_text())
    assert len(rows) == 1 and rows[0]["source"] == "你好"


@pytest.mark.parametrize("rate,frames", [(24000, b"\0\0"), (16000, b"")])
def test_whisper_rejects_wrong_sample_rate_and_empty_wav(tmp_path, rate, frames):
    audio = tmp_path / "invalid.wav"
    with wave.open(str(audio), "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(frames)
    with pytest.raises(dubbing.DubbingError, match="辨識音訊"):
        dubbing.transcribe_worker(SimpleNamespace(audio=str(audio)))
