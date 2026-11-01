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
