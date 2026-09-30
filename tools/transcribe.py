"""Transcribe WAV files offline with the listener's ASR model (not streaming).

Use it as an "ear" to check what the avatar really said; run via
tools/transcribe.sh, which starts it in the listener image.
"""
import sys, torch, librosa
from transformers import AutoProcessor, AutoModelForRNNT
M = "nvidia/nemotron-3.5-asr-streaming-0.6b"
proc = AutoProcessor.from_pretrained(M)
model = AutoModelForRNNT.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
for path in sys.argv[1:]:
    audio, _ = librosa.load(path, sr=16000)
    inputs = proc(audio, sampling_rate=16000, language="en-US", return_tensors="pt").to("cuda")
    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
    with torch.inference_mode():
        out = model.generate(**inputs)
    seq = out.sequences if hasattr(out, "sequences") else out
    lead = next((i for i, v in enumerate(abs(audio)) if v > 0.02), 0) / 16000
    print(f"{path.split('/')[-1]:28} speech starts at {lead:.2f}s: {proc.batch_decode(seq, skip_special_tokens=True)[0]!r}")
