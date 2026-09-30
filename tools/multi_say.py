"""One WebRTC session, several sentences in a row; record what arrives.

Run inside the avatar container (see tools/README.md):
    docker compose exec -T webrtc-avatar python - "Great, then we agree." "Okay." < tools/multi_say.py
Writes results/multi_say.wav; cut and transcribe it with tools/transcribe.sh.
"""
import asyncio, json, sys, time
import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder

SENTENCES = sys.argv[1:] or ["Great, then we agree.", "Good question, let me think.", "Great, then we agree.",
                             "Okay, I'll stop.", "Great, then we agree.", "Glad you like it!"]

async def main():
    pc = RTCPeerConnection()
    channel = pc.createDataChannel("avatar-control")
    recorder = MediaRecorder("/workspace/results/multi_say.wav")
    ready = asyncio.Event(); marks = []
    t0 = time.perf_counter()
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio": recorder.addTrack(track)
    @channel.on("message")
    def on_message(m):
        e = json.loads(m)
        if e.get("type") == "ready": ready.set()
        if e.get("type") == "playing": marks.append(round(time.perf_counter() - t0, 2))
    pc.addTransceiver("video", direction="recvonly"); pc.addTransceiver("audio", direction="recvonly")
    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != "complete": await asyncio.sleep(0.05)
    r = httpx.post("http://127.0.0.1:8000/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    await pc.setRemoteDescription(RTCSessionDescription(**r.json()))
    await recorder.start()
    await ready.wait()
    for text in SENTENCES:
        ready.clear(); await asyncio.sleep(2.0)
        channel.send(json.dumps({"text": text, "voice_mode": "preset-clone"}))
        await asyncio.wait_for(ready.wait(), 60)
    await asyncio.sleep(1.5); await recorder.stop(); await pc.close()
    print("playing events at", marks)
asyncio.run(main())
