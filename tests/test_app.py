import sys
from pathlib import Path

import gradio as gr
import pytest

import app
from dubbing import DubbingError, Job, Segment, Settings, SpeechTimingError


@pytest.fixture
def demo():
    return app.build_app(Settings())


def handler(demo, name):
    return next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == name)


def test_interface_builds_without_models_and_serializes_work(demo):
    assert demo.title == "地端影片配音"
    assert demo.enable_queue
    model_handlers = [fn for fn in demo.fns.values() if fn.fn.__name__ in {"prepare", "retranslate", "render"}]
    assert len(model_handlers) == 3
    assert all(fn.concurrency_id == "models" and fn.concurrency_limit == 1 for fn in model_handlers)
    editor = next(block for block in demo.blocks.values() if isinstance(block, gr.Dataframe))
    assert editor.static_columns == [0, 1]
    assert editor.type == "array"


def test_prepare_clears_previous_job_before_starting(monkeypatch, demo, tmp_path):
    job = Job(tmp_path, tmp_path / "original.mp4", tmp_path / "original.wav", 4, "zh", "en",
              [Segment(1, 0, 3, "原文", "translation")])
    monkeypatch.setattr(app, "prepare_video", lambda *a: job)
    generator = handler(demo, "prepare")("video.mp4", "中文 → 英文")
    first = next(generator)
    assert first[0] is None and first[1] == [] and first[3:] == (None,) * 4
    final = next(generator)
    assert final[0] is job
    assert final[1] == [["1", "0.00–3.00 秒", "原文", "translation"]]


def test_failed_prepare_leaves_no_stale_job(monkeypatch, demo):
    def fail(*args):
        raise DubbingError("Ollama unavailable")

    monkeypatch.setattr(app, "prepare_video", fail)
    generator = handler(demo, "prepare")("video.mp4", "中文 → 英文")
    assert next(generator)[0] is None
    failed = next(generator)
    assert failed[0] is None and "處理失敗" in failed[2]
    with pytest.raises(gr.Error, match="Ollama unavailable"):
        next(generator)


@pytest.mark.parametrize("name,args", [("render", (None, [], None)), ("retranslate", (None, []))])
def test_actions_require_prepared_video(demo, name, args):
    with pytest.raises(gr.Error, match="先上傳"):
        handler(demo, name)(*args)


def test_render_provides_preview_and_separate_downloads(monkeypatch, demo, tmp_path):
    job = Job(tmp_path, Path("input.mp4"), Path("input.wav"), 4, "en", "zh", [Segment(1, 0, 3, "Hi", "你好")])
    rows = app.table_rows(job)
    results = (tmp_path / "dubbed.mp4", tmp_path / "translated.srt", tmp_path / "source.srt")
    monkeypatch.setattr(app, "render_job", lambda *a: results)
    output = handler(demo, "render")(job, rows, None)
    assert output[:2] == (job, rows)
    assert output[2:6] == (str(results[0]), str(results[0]), str(results[1]), str(results[2]))


def test_render_refreshes_edited_translations_and_explains_shortening(monkeypatch, demo, tmp_path):
    job = Job(tmp_path, Path("input.mp4"), Path("input.wav"), 4, "zh", "en",
              [Segment(1, 0, 3, "售價一百元", "The price is 100 dollars.")])

    def render(*args):
        args[0].segments[0].target = "Price: $100."
        return (tmp_path / "dubbed.mp4", tmp_path / "translated.srt", tmp_path / "source.srt")

    monkeypatch.setattr(app, "render_job", render)
    output = handler(demo, "render")(job, app.table_rows(job), None)
    assert output[1][0][3] == "Price: $100."
    assert "自動縮短第 1 段" in output[-1]


def test_remaining_timing_error_keeps_editor_and_clears_outputs(monkeypatch, demo, tmp_path):
    job = Job(tmp_path, Path("input.mp4"), Path("input.wav"), 4, "zh", "en",
              [Segment(1, 0, 3, "原文", "Original translation")])

    def fail(*args):
        raise SpeechTimingError("已自動縮句並重新配音兩輪，仍需調整")

    monkeypatch.setattr(app, "render_job", fail)
    rows = app.table_rows(job)
    output = handler(demo, "render")(job, rows, None)
    assert output[:2] == (job, rows)
    assert output[2:6] == (None,) * 4
    assert "仍需調整" in output[-1]


def test_startup_keeps_service_local_and_sharing_disabled(monkeypatch, demo):
    calls = []
    monkeypatch.setattr(app, "build_app", lambda: demo)
    monkeypatch.setattr(demo, "launch", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.delenv("DUBBING_USERNAME", raising=False)
    monkeypatch.delenv("DUBBING_PASSWORD", raising=False)
    app.main()
    assert calls[0]["server_name"] == "127.0.0.1"
    assert calls[0]["share"] is False
    assert calls[0]["auth"] is None
