"""Run CosyVoice 3 in its own environment and release GPU memory on exit."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?")
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and not args.request:
        parser.error("request is required unless --check is specified")
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party/Matcha-TTS"))
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    if args.check:
        print(f"CosyVoice imports OK; torch={torch.__version__}; CUDA={torch.cuda.is_available()}")
        return

    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    model = AutoModel(model_dir=str(Path(args.model).resolve()))
    for segment in request["segments"]:
        text = "You are a helpful assistant.<|endofprompt|>" + segment["target"]
        chunks = [item["tts_speech"].cpu() for item in model.inference_cross_lingual(
            text, request["reference"], stream=False,
        )]
        if not chunks:
            raise RuntimeError(f"CosyVoice returned no audio for segment {segment['id']}")
        speech = torch.cat(chunks, dim=1)
        speech = torchaudio.functional.resample(speech, model.sample_rate, 24000)
        torchaudio.save(str(request_path.parent / f"speech-{segment['id']}.wav"), speech, 24000,
                       encoding="PCM_S", bits_per_sample=16)
        print(f"Generated segment {segment['id']}", flush=True)


if __name__ == "__main__":
    main()
