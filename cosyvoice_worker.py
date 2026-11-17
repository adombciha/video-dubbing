"""Run CosyVoice in its own environment, reusing weights within one render."""

import argparse
import gc
import json
import sys
import time
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path

RESPONSE_PREFIX = "DUBBING_RESPONSE "
REFERENCE_ID = "dubbing-reference"


@contextmanager
def timed_phase(path, phase, **details):
    started = time.perf_counter()
    try:
        yield
    finally:
        record = {"phase": phase, "seconds": round(time.perf_counter() - started, 4), **details}
        line = json.dumps(record, ensure_ascii=False)
        if path:
            with path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
        print("TIMING " + line, file=sys.stderr, flush=True)


def reply(stream, status, **details):
    stream.write(RESPONSE_PREFIX + json.dumps({"status": status, **details}, ensure_ascii=False) + "\n")
    stream.flush()


def synthesize(model, request_path, torch, torchaudio, timings, cached_reference=False):
    request = json.loads(request_path.read_text(encoding="utf-8"))
    for segment in request["segments"]:
        with timed_phase(timings, "speech", attempt=request_path.parent.name, segment_id=segment["id"]):
            text = "You are a helpful assistant.<|endofprompt|>" + segment["target"]
            options = {"zero_shot_spk_id": REFERENCE_ID} if cached_reference else {}
            chunks = [item["tts_speech"].cpu() for item in model.inference_cross_lingual(
                text, request["reference"], stream=False, **options,
            )]
            if not chunks:
                raise RuntimeError(f"CosyVoice returned no audio for segment {segment['id']}")
            speech = torch.cat(chunks, dim=1)
            speech = torchaudio.functional.resample(speech, model.sample_rate, 24000)
            torchaudio.save(str(request_path.parent / f"speech-{segment['id']}.wav"), speech, 24000,
                           encoding="PCM_S", bits_per_sample=16)
        print(f"Generated segment {segment['id']}", flush=True)


def serve(model, torch, torchaudio, timings, stream):
    reference = None
    parked = False
    reply(stream, "ready")
    for line in sys.stdin:
        request_path = Path(json.loads(line)["request"]).resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if reference is not None and request["reference"] != reference:
            raise ValueError("A render session must use the same reference voice")
        if parked:
            with timed_phase(timings, "resume_gpu", attempt=request_path.parent.name):
                for name in ("llm", "flow", "hift"):
                    getattr(model.model, name).to(model.model.device)
            parked = False
        if reference is None:
            with timed_phase(timings, "reference"):
                model.add_zero_shot_spk("", request["reference"], REFERENCE_ID)
                model.frontend.spk2info = {REFERENCE_ID: model.frontend.spk2info[REFERENCE_ID]}
                # Cached reference features replace further tokenizer inference in this session.
                del model.frontend.speech_tokenizer_session
                gc.collect()
            reference = request["reference"]
        synthesize(model, request_path, torch, torchaudio, timings, cached_reference=True)
        if torch.cuda.is_available():
            with timed_phase(timings, "park_cpu", attempt=request_path.parent.name):
                torch.cuda.synchronize()
                for name in ("llm", "flow", "hift"):
                    getattr(model.model, name).to("cpu")
                features = model.frontend.spk2info[REFERENCE_ID]
                model.frontend.spk2info[REFERENCE_ID] = {
                    key: value.cpu() if torch.is_tensor(value) else value for key, value in features.items()
                }
                del features
                gc.collect()
                torch.cuda.empty_cache()
            parked = True
        reply(stream, "done", request=str(request_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?")
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--timings", type=Path)
    args = parser.parse_args()
    if not args.check and not args.request and not args.serve:
        parser.error("request is required unless --check or --serve is specified")
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party/Matcha-TTS"))
    control = sys.stdout
    try:
        with redirect_stdout(sys.stderr) if args.serve else nullcontext():
            with timed_phase(args.timings, "torch_import"):
                import torch
                import torchaudio
            with timed_phase(args.timings, "cosyvoice_import"):
                from cosyvoice.cli.cosyvoice import AutoModel
            if args.check:
                print(f"CosyVoice imports OK; torch={torch.__version__}; CUDA={torch.cuda.is_available()}")
                return
            with timed_phase(args.timings, "model_load"):
                model = AutoModel(model_dir=str(Path(args.model).resolve()))
            if args.serve:
                serve(model, torch, torchaudio, args.timings, control)
            else:
                synthesize(model, Path(args.request).resolve(), torch, torchaudio, args.timings)
    except Exception as exc:
        if args.serve:
            reply(control, "error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
