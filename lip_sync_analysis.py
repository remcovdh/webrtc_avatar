"""Measure, per phrase, how far JoyVASA's mouth motion leads or lags the voice.

Start the avatar with AVATAR_DEBUG_DUMP_DIR=/workspace/results/motion-dumps,
speak some phrases, then run on the host:

    python3 lip_sync_analysis.py results/motion-dumps

For each dumped phrase it cross-correlates mouth opening (first principal
component of the lip keypoints) with the speech loudness over +-320 ms.
Negative lag = mouth before sound. "corr" is how well the mouth follows the
loudness at the best lag; ignore phrases below ~0.3.
"""
import glob, sys, wave
import numpy as np

LIP = [6, 12, 14, 17, 19, 20]
for npz in sorted(glob.glob(sys.argv[1] + "/*.npz")):
    d = np.load(npz)
    fps = float(d["fps"])
    lips = d["exp"][:, 0, LIP, :].reshape(len(d["exp"]), -1)
    with wave.open(npz[:-4] + ".wav") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
    hop = rate / fps
    n = len(lips)
    env = np.array([np.sqrt(np.mean(pcm[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-9) for i in range(n)])
    # Mouth activity = speed of the lip keypoints plus their distance from the
    # phrase's most closed state (first principal component), sign-free.
    centered = lips - lips.mean(0)
    pc = np.linalg.svd(centered, full_matrices=False)[2][0]
    opening = centered @ pc
    if np.corrcoef(opening, env)[0, 1] < 0:
        opening = -opening
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    e, o = z(env), z(opening)
    lags = range(-8, 9)  # +-320 ms at 25 fps; + = mouth after sound
    corr = []
    for lag in lags:
        if lag >= 0:
            c = np.corrcoef(e[: n - lag], o[lag:])[0, 1]
        else:
            c = np.corrcoef(e[-lag:], o[: n + lag])[0, 1]
        corr.append(c)
    best = int(np.argmax(corr))
    print(f"{npz.split('/')[-1]}: {n / fps:.2f}s  best lag {list(lags)[best] * 1000 / fps:+.0f} ms "
          f"(corr {corr[best]:.2f}, at 0 ms {corr[8]:.2f})")
