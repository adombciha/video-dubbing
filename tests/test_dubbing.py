import array
import json
import math
import shutil
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import dubbing as d


def make_wav(path, seconds, amplitude=1500):
    samples = array.array("h", (int(amplitude * math.sin(2 * math.pi * 220 * n / d.SAMPLE_RATE))
                               for n in range(round(seconds * d.SAMPLE_RATE))))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, d.SAMPLE_RATE, 0, "NONE", "not compressed"))
        target.writeframes(samples.tobytes())
    return path


@pytest.fixture
def job(tmp_path):
    video, audio = tmp_path / "original.mp4", tmp_path / "original.wav"
    video.write_bytes(b"original video")
    make_wav(audio, 4)
    return d.Job(tmp_path, video, audio, 4, "zh", "en", [
        d.Segment(1, 0.5, 1.5, "產品售價 100 元。", "The product costs 100 dollars."),
        d.Segment(2, 2.5, 3.5, "按一下開始。", "Press Start."),
    ])


def translation_response(rows):
    return {"message": {"content": json.dumps({"segments": rows})}}


def test_translations_preserve_ids_and_release_ollama(monkeypatch, job):
    request = Mock(return_value=translation_response([{"id": 1, "text": " Price: $100. "}, {"id": 2, "text": "Press Start."}]))
    monkeypatch.setattr(d, "ollama_request", request)
    d.translate_segments(job.segments, "en", d.Settings())
    assert [s.target for s in job.segments] == ["Price: $100.", "Press Start."]
    payload = request.call_args.args[2]
    assert payload["keep_alive"] == 0
    assert payload["stream"] is False
    assert "numbers" in payload["messages"][0]["content"]
    assert [r["id"] for r in json.loads(payload["messages"][1]["content"])] == [1, 2]


@pytest.mark.parametrize("rows", [
    [{"id": 1, "text": "first"}],
    [{"id": 1, "text": "first"}, {"id": 1, "text": "duplicate"}],
    [{"id": 2, "text": "second"}, {"id": 1, "text": "first"}],
    [{"id": 1, "text": "first"}, {"id": 2, "text": " "}],
    [{"id": True, "text": "boolean"}, {"id": 2, "text": "second"}],
    [{"id": 1, "text": "first"}, {"id": 2, "text": 123}],
    None,
])
def test_bad_translation_never_partially_applies(monkeypatch, job, rows):
    before = [s.target for s in job.segments]
    monkeypatch.setattr(d, "ollama_request", lambda *a, **k: translation_response(rows))
    with pytest.raises(d.DubbingError, match="缺段"):
        d.translate_segments(job.segments, "en", d.Settings())
    assert [s.target for s in job.segments] == before


def test_reverse_translation_uses_taiwan_traditional_chinese(monkeypatch, job):
    request = Mock(return_value=translation_response([{"id": 1, "text": "售價 100 元。"}, {"id": 2, "text": "按開始。"}]))
    monkeypatch.setattr(d, "ollama_request", request)
    d.translate_segments(job.segments, "zh", d.Settings())
    assert "繁體中文（臺灣用語）" in request.call_args.args[2]["messages"][0]["content"]


def test_translation_batches_keep_global_ids(monkeypatch):
    segments = [d.Segment(n, n, n + 0.5, str(n)) for n in range(1, 15)]
    calls = []

    def response(settings, endpoint, payload):
        rows = json.loads(payload["messages"][1]["content"])
        calls.append([r["id"] for r in rows])
        return translation_response([{"id": r["id"], "text": "translated " + r["text"]} for r in rows])

    monkeypatch.setattr(d, "ollama_request", response)
    d.translate_segments(segments, "en", d.Settings())
    assert calls == [list(range(1, 13)), [13, 14]]
    assert segments[-1].target == "translated 14"


@pytest.mark.parametrize("segments", [[], [d.Segment(1, 1, 0, "wrong")], [d.Segment(1, 0, float("nan"), "wrong")],
    [d.Segment(1, 0, 2, "one"), d.Segment(2, 1, 3, "two")], [d.Segment(2, 0, 1, "wrong")],
    [d.Segment(1, 0, 5, "long")], [d.Segment(1, 0, 1, " ")]])
def test_invalid_transcripts_are_rejected(segments):
    with pytest.raises(d.DubbingError):
        d.validate_segments(segments, 4)


def test_edits_are_validated_before_changing_job(job):
    rows = [["1", "display only", "修正原文", "New translation"], ["2", "display only", " ", "second"]]
    original = job.segments[0].source
    with pytest.raises(d.DubbingError):
        d.apply_edits(job, rows)
    assert job.segments[0].source == original
    rows[1][2] = "第二句"
    d.apply_edits(job, rows)
    assert job.segments[0].source == "修正原文"
    saved = json.loads((job.folder / "job.json").read_text())
    assert saved["segments"][0]["target"] == "New translation"


def test_reordered_edits_are_rejected(job):
    with pytest.raises(d.DubbingError, match="重排"):
        d.apply_edits(job, [["2", "", "second", "two"], ["1", "", "first", "one"]])


def test_assembly_places_all_speech_and_pads_silence(job, tmp_path):
    clips = [make_wav(tmp_path / "a.wav", 1), make_wav(tmp_path / "b.wav", 0.5)]
    output, ends = d.assemble_audio(job, clips, tmp_path, d.Settings())
    assert ends == [1.5, 3]
    with wave.open(str(output), "rb") as audio:
        assert audio.getnframes() == 4 * d.SAMPLE_RATE
        samples = array.array("h", audio.readframes(audio.getnframes()))
    assert not any(samples[:round(0.5 * d.SAMPLE_RATE)])
    assert any(samples[round(0.5 * d.SAMPLE_RATE):round(1.5 * d.SAMPLE_RATE)])
    assert not any(samples[round(1.5 * d.SAMPLE_RATE):round(2.5 * d.SAMPLE_RATE)])
    assert any(samples[round(2.5 * d.SAMPLE_RATE):3 * d.SAMPLE_RATE])
    assert not any(samples[3 * d.SAMPLE_RATE:])


def test_large_overflow_reports_all_segments_without_cutting(job, tmp_path):
    clips = [make_wav(tmp_path / "a.wav", 2.8), make_wav(tmp_path / "b.wav", 2)]
    with pytest.raises(d.DubbingError) as error:
        d.assemble_audio(job, clips, tmp_path, d.Settings())
    assert "第 1 段" in str(error.value) and "第 2 段" in str(error.value)
    assert not (tmp_path / "dubbed.wav").exists()
    assert d.wav_info(clips[0]) == round(2.8 * d.SAMPLE_RATE)


def test_small_overflow_speeds_up_only_when_needed(monkeypatch, job, tmp_path):
    clips = [make_wav(tmp_path / "a.wav", 2.15), make_wav(tmp_path / "b.wav", 0.5)]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        make_wav(Path(command[-1]), 1.99)

    monkeypatch.setattr(d, "run_command", run)
    _, ends = d.assemble_audio(job, clips, tmp_path, d.Settings())
    assert len(calls) == 1
    speed = float(calls[0][calls[0].index("-af") + 1].split("=")[1])
    assert 1 < speed <= 1.2
    assert ends[0] == 2.49


def test_silent_or_empty_tts_is_rejected(tmp_path):
    for seconds in [0, 1]:
        with pytest.raises(d.DubbingError, match="靜音"):
            d.wav_info(make_wav(tmp_path / "silent.wav", seconds, amplitude=0))


def test_srt_matches_actual_voice_times(job, tmp_path):
    path = tmp_path / "translation.srt"
    job.segments[0].target = "Product\n\nname"
    d.write_srt(path, job.segments, [2.49, 3])
    text = path.read_text(encoding="utf-8-sig")
    assert "00:00:00,500 --> 00:00:02,490" in text
    assert "Product name" in text
    assert "00:00:02,500 --> 00:00:03,000" in text


@pytest.mark.parametrize("metadata", [
    {"streams": [{"codec_type": "video"}], "format": {"duration": "3"}},
    {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "nan"}},
    {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "301"}},
    {},
])
def test_video_validation_rejects_missing_audio_and_invalid_duration(monkeypatch, tmp_path, metadata):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"data")
    monkeypatch.setattr(d, "run_command", lambda *a, **k: json.dumps(metadata))
    with pytest.raises(d.DubbingError):
        d.probe_video(video, d.Settings())


def test_command_errors_and_timeout_are_actionable(monkeypatch):
    monkeypatch.setattr(d.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired(["ffmpeg"], 10)))
    with pytest.raises(d.DubbingError, match="超過 10 秒"):
        d.run_command(["ffmpeg"], timeout=10)
    monkeypatch.setattr(d.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1, stderr="bad file", stdout="")))
    with pytest.raises(d.DubbingError, match="bad file"):
        d.run_command(["ffmpeg"])


def test_ollama_connection_failure_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(d.urllib.request, "urlopen", Mock(side_effect=OSError("connection refused")))
    with pytest.raises(d.DubbingError, match="connection refused"):
        d.ollama_request(d.Settings(), "/api/tags")


def test_prepare_stages_original_and_saves_both_languages(monkeypatch, tmp_path):
    original = tmp_path / "user's demo.MOV"
    original.write_bytes(b"never overwrite me")
    settings = replace(d.Settings(), output_root=tmp_path / "jobs")
    monkeypatch.setattr(d, "probe_video", lambda *args: 4)

    def run(command, **kwargs):
        if "transcribe" in command:
            d.atomic_json(Path(command[4]), [{"id": 1, "start": 0.5, "end": 3.5, "source": "你好"}])
        else:
            make_wav(Path(command[-1]), 4)

    monkeypatch.setattr(d, "run_command", run)
    monkeypatch.setattr(d, "ollama_request", lambda *a, **k: translation_response([{"id": 1, "text": "Hello"}]))
    job = d.prepare_video(str(original), "中文 → 英文", settings)
    assert original.read_bytes() == b"never overwrite me"
    assert job.video != original and job.video.read_bytes() == original.read_bytes()
    assert (job.source_language, job.target_language) == ("zh", "en")
    assert json.loads((job.folder / "job.json").read_text())["segments"][0]["target"] == "Hello"


def test_render_outputs_and_keeps_original_untouched(monkeypatch, job, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "cosyvoice3.yaml").touch()
    settings = replace(d.Settings(), cosyvoice_model=model)
    before = job.video.read_bytes()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "cosyvoice_worker.py" in command[1]:
            request_path = Path(command[2])
            for n in [1, 2]:
                make_wav(request_path.parent / f"speech-{n}.wav", 0.6)
        elif command[-1].endswith(".wav"):
            make_wav(Path(command[-1]), 3)
        else:
            Path(command[-1]).write_bytes(b"rendered")

    monkeypatch.setattr(d, "run_command", run)
    output, translated, source = d.render_job(job, None, settings)
    assert output.read_bytes() == b"rendered"
    assert translated.is_file() and source.is_file()
    assert job.video.read_bytes() == before
    mux = commands[-1]
    assert [mux[index + 1] for index, value in enumerate(mux) if value == "-map"] == ["0:v:0", "1:a:0"]
    assert mux[mux.index("-pix_fmt") + 1] == "yuv420p"
    assert mux[mux.index("-movflags") + 1] == "+faststart"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg and ffprobe required")
def test_real_ffmpeg_timing_and_browser_compatible_mux(tmp_path, monkeypatch):
    settings = d.Settings()
    video = tmp_path / "original.mp4"
    d.run_command([settings.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=4",
                   "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)])
    duration = d.probe_video(video, settings)
    job = d.Job(tmp_path, video, tmp_path / "original.wav", duration, "zh", "en", [
        d.Segment(1, 0.5, 1.5, "第一句", "First"), d.Segment(2, 2.5, 3.5, "第二句", "Second"),
    ])
    model = tmp_path / "model"
    model.mkdir()
    (model / "cosyvoice3.yaml").touch()
    settings = replace(settings, cosyvoice_model=model)
    reference = make_wav(tmp_path / "reference.wav", 3)
    real_run = d.run_command
    original_bytes = video.read_bytes()

    def synthesize_only(command, **kwargs):
        if len(command) > 1 and command[1] == str(d.ROOT / "cosyvoice_worker.py"):
            request_path = Path(command[2])
            make_wav(request_path.parent / "speech-1.wav", 2.1)
            make_wav(request_path.parent / "speech-2.wav", 1)
            return ""
        return real_run(command, **kwargs)

    monkeypatch.setattr(d, "run_command", synthesize_only)
    output, translation, source = d.render_job(job, str(reference), settings)
    assert abs(d.probe_video(output, settings) - duration) < 0.1
    with wave.open(str(output.parent / "dubbed.wav"), "rb") as result:
        assert result.getnframes() == round(duration * d.SAMPLE_RATE)
    metadata = json.loads(real_run([settings.ffprobe, "-v", "error", "-show_streams", "-of", "json", str(output)]))
    streams = {item["codec_type"]: item for item in metadata["streams"]}
    assert streams["video"]["codec_name"] == "h264" and streams["video"]["pix_fmt"] == "yuv420p"
    assert streams["audio"]["codec_name"] == "aac"
    assert translation.is_file() and source.is_file()
    assert video.read_bytes() == original_bytes


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg and ffprobe required")
def test_real_corrupt_and_audio_free_videos_are_rejected(tmp_path):
    settings = d.Settings()
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real movie")
    with pytest.raises(d.DubbingError, match="執行失敗"):
        d.probe_video(corrupt, settings)
    silent = tmp_path / "no-audio.mp4"
    d.run_command([settings.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=160x120:d=1",
                   "-an", "-c:v", "libx264", str(silent)])
    with pytest.raises(d.DubbingError, match="聲音軌"):
        d.probe_video(silent, settings)
