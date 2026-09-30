"""Speak a WAV into the avatar's microphone path; print System 1/2's handling.

Run inside the avatar container (see tools/README.md), language then a WAV path
as the container sees it:
    docker compose exec -T webrtc-avatar python - en-US /workspace/tools/audio/question.wav < tools/mic_turn.py
"""
import asyncio, json, sys, time, wave
import numpy as np, httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

def to48k(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
        x = np.interp(np.arange(0, len(x), w.getframerate() / 48000), np.arange(len(x)), x)
    x = np.concatenate([np.zeros(24000), x, np.zeros(96000)]).clip(-32768, 32767).astype(np.int16)
    with wave.open("/tmp/mic.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000); w.writeframes(x.tobytes())
    return len(x) / 48000

async def main(lang, path):
    seconds = to48k(path)
    pc = RTCPeerConnection(); channel = pc.createDataChannel("avatar-control"); t0 = time.perf_counter()
    @channel.on("open")
    def on_open(): channel.send(json.dumps({"type": "listen_language", "language": lang}))
    @channel.on("message")
    def on_message(m):
        e = json.loads(m)
        if e.get("type") == "conversation" and e.get("turn", {}).get("decision"):
            t, d = e["turn"], e["turn"]["decision"]
            best = (t.get("retrieved") or [{}])[0]
            print(f"heard {d['text']!r}: intent={d['intent']} ({d['sources']['intent']}, {d['confidence']['intent']:.2f}) "
                  f"emotion={d['emotion']} ({d['sources']['emotion']}) topic={d['topic']} {d['latency_ms']}ms | "
                  f"gap={t.get('knowledge_gap', False)} best={best.get('source')} {best.get('score')} | reply={t['reply']!r}", flush=True)
    pc.addTransceiver("video", direction="recvonly"); pc.addTrack(MediaPlayer("/tmp/mic.wav").audio)
    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != "complete": await asyncio.sleep(0.05)
    r = httpx.post("http://127.0.0.1:8000/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    await pc.setRemoteDescription(RTCSessionDescription(**r.json()))
    await asyncio.sleep(seconds + 6); await pc.close()
asyncio.run(main(sys.argv[1], sys.argv[2]))
