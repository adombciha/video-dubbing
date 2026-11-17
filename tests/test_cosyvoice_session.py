import json
import sys
from dataclasses import replace

import pytest

import dubbing


@pytest.fixture
def session(tmp_path, monkeypatch):
    worker = tmp_path / "cosyvoice_worker.py"
    worker.write_text('''import json, os, sys, time
def reply(**data):
    print("DUBBING_RESPONSE " + json.dumps(data), flush=True)
print("startup output", flush=True)
print("worker warning", file=sys.stderr, flush=True)
reply(status="ready")
for line in sys.stdin:
    path = json.loads(line)["request"]
    mode = json.load(open(path)).get("mode", "success")
    if mode == "error":
        reply(status="error", error="inference failed")
        break
    if mode == "exit":
        sys.exit(3)
    if mode == "timeout":
        time.sleep(30)
    if mode == "malformed":
        print("DUBBING_RESPONSE invalid-json", flush=True)
        continue
    open(path + ".handled", "w").write(str(os.getpid()))
    reply(status="done", request="wrong-request" if mode == "mismatch" else path)
''')
    monkeypatch.setattr(dubbing, "ROOT", tmp_path)
    settings = replace(dubbing.Settings(), cosyvoice_python=sys.executable)
    return dubbing.CosyVoiceSession(settings, tmp_path, timeout=1)


def test_multiple_requests_share_one_process_and_close_cleanly(session, tmp_path):
    requests = [tmp_path / "one.json", tmp_path / "two.json"]
    with session as voice:
        for path in requests:
            path.write_text("{}")
            voice.synthesize(path)
        pid = voice.process.pid
    assert all((tmp_path / (path.name + ".handled")).read_text() == str(pid) for path in requests)
    assert session.process.poll() == 0
    assert not session.reader.is_alive()
    assert session.log.closed
    log = (tmp_path / "cosyvoice.log").read_text()
    assert "startup output" in log and "worker warning" in log


@pytest.mark.parametrize("mode", ["error", "exit", "timeout", "malformed", "mismatch"])
def test_failed_requests_stop_process_and_reader(session, tmp_path, mode):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"mode": mode}))
    with pytest.raises(dubbing.DubbingError), session as voice:
        voice.synthesize(request)
    assert session.process.poll() is not None
    assert not session.reader.is_alive()
    assert session.log.closed


def test_failed_process_start_closes_log(session):
    session.settings = replace(session.settings, cosyvoice_python="/missing/python")
    with pytest.raises(dubbing.DubbingError, match="無法啟動"), session:
        pass
    assert session.log.closed
