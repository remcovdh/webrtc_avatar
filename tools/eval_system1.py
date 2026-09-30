"""Benchmark a JevK5 GGUF on 18 Dutch/English utterances (intent + emotion).

This is the test that chose JevK5 over Laya (intent 15/18, emotion 17/18).
It loads a second copy of the model (+3.4 GB GPU), so run it with the avatar
paused (see tools/README.md):
    docker compose run --rm --no-deps -T conductor python - < tools/eval_system1.py
"""
import torch  # noqa: F401  (load before llama.cpp)
import json, math, subprocess, time
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from jevk5.prompt import LETTERS, answer, decision_options, prompt_text

def gpu(): return int(subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip())
cfg = json.load(open(hf_hub_download("alibiserikbay/JevK5", "jevk5_config.json")))
TEMP = float(cfg.get("temperature", 1.367))
before = gpu()
llm = Llama(model_path=hf_hub_download("alibiserikbay/JevK5-GGUF", "jevk5-4b-v0.3-Q4_K_M.gguf"), n_gpu_layers=-1, n_ctx=2048, logits_all=True, verbose=False)
print(f"temperature {TEMP}, GPU +{gpu()-before} MiB")
letter_ids = [llm.tokenize(l.encode(), add_bos=False, special=False) for l in LETTERS]
letter_ids = [ids[0] for ids in letter_ids]

def probs(state, question):
    options = decision_options(question)
    prompt = prompt_text(state, question["instructions"], [t for _, t in options])
    tokens = llm.tokenize(prompt.encode(), add_bos=False, special=True)
    llm.reset(); llm.eval(tokens)
    logits = llm.scores[llm.n_tokens - 1]
    z = [float(logits[letter_ids[i]]) for i in range(len(options))]
    top = max(z); w = [math.exp((v - top) / TEMP) for v in z]; s = sum(w)
    return {k: p / s for (k, _), p in zip(options, w)}

INTENTS = {
    "question": "asks a question or wants information or an explanation",
    "request": "asks the assistant to do or tell something (other than stop)",
    "remark": "shares a statement, opinion or observation without asking anything",
    "greeting": "says hello or opens the conversation", "farewell": "says goodbye or ends the conversation",
    "thanks": "thanks the assistant", "agreement": "agrees, confirms or says yes",
    "disagreement": "disagrees, corrects the assistant or says no", "stop": "tells the assistant to stop, be quiet or wait"}
EMOTIONS = {"surprised": "surprised, astonished", "angry": "annoyed, irritated, frustrated", "sad": "sad, disappointed, worried",
            "joyful": "happy, enthusiastic, amused, positive", "neutral": "calm, matter-of-fact, no particular emotion"}
Q_INTENT = {"type": "choice", "instructions": "What is the speaker doing with this utterance to a conversational assistant?", "criteria": INTENTS}
Q_EMO = {"type": "choice", "instructions": "Which emotion does the speaker express?", "criteria": EMOTIONS}
CASES = [
    ("Hello, nice to meet you!", "greeting", "joyful"), ("Hallo, leuk je te ontmoeten!", "greeting", "joyful"),
    ("What is a balloon made of?", "question", "neutral"), ("Waar is een ballon van gemaakt?", "question", "neutral"),
    ("Can you tell me something about Devoteam?", "request", "neutral"), ("Kun je iets vertellen over het weer morgen?", "request", "neutral"),
    ("I think this avatar looks really great.", "remark", "joyful"), ("Ik vind dat je mond heel raar beweegt.", "remark", "angry"),
    ("Stop talking, please.", "stop", "angry"), ("Wacht even, stop maar.", "stop", "neutral"),
    ("Thank you so much, that was helpful!", "thanks", "joyful"), ("Dankjewel, dat was duidelijk.", "thanks", "neutral"),
    ("Yes, exactly, that's right.", "agreement", "neutral"), ("Nee, dat klopt helemaal niet!", "disagreement", "angry"),
    ("Goodbye, see you tomorrow.", "farewell", "neutral"), ("Wow, I did not expect that at all!", "remark", "surprised"),
    # without the question mark the ASR often drops
    ("Why is the avatar so fast", "question", "neutral"), ("What is the capital of France", "question", "neutral"),
]
probs(CASES[0][0], Q_INTENT)
ok_i = ok_e = 0; times = []
for text, intent, emotion in CASES:
    t = time.perf_counter(); pi = probs(text, Q_INTENT); pe = probs(text, Q_EMO); times.append((time.perf_counter() - t) * 1000)
    gi = max(pi, key=pi.get); ge = max(pe, key=pe.get)
    ok_i += gi == intent; ok_e += ge == emotion
    print(f"{'✓' if gi==intent else '✗'}{'✓' if ge==emotion else '✗'} {text[:44]:44} intent={gi:12} ({pi[gi]:.2f}) emotion={ge:9} ({pe[ge]:.2f})")
print(f"intent {ok_i}/{len(CASES)}, emotion {ok_e}/{len(CASES)}; both questions per utterance: median {sorted(times)[len(times)//2]:.0f} ms, max {max(times):.0f} ms")
