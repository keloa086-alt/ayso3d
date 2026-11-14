# ═══════════════════════════════════════════════════════════════
#  Ayso Voice — صوت أنمي عربي بالتقليد (XTTS-v2)
#  Runtime → T4 GPU   |   أوقف أي خلية شغّالة قبل التشغيل (⏹)
# ═══════════════════════════════════════════════════════════════
import os, sys, subprocess, time, re, urllib.request

print("═"*56); print("  Ayso Voice — صوت أنمي عربي"); print("═"*56)

subprocess.run("pkill -f ayso_srv.py 2>/dev/null; pkill cloudflared 2>/dev/null", shell=True)
time.sleep(1)

# ── ١) المكتبات ──
print("\n١) تثبيت (٣–٥ دقايق أول مرة)…", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "torch","torchaudio","fastapi","uvicorn","python-multipart",
    "transformers==4.57.1","accelerate","soundfile","librosa",
    "bitsandbytes","coqui-tts","edge-tts","python-multipart"], check=False)
print("   ✓", flush=True)

# ── ٢) عيّنة الصوت ──
print("\n٢) تنزيل عيّنة الصوت…", flush=True)
SPK_MP3 = "/content/spk.mp3"
SPK_WAV = "/content/spk.wav"
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/keloa086-alt/ayso3d/main/assets/speaker.mp3", SPK_MP3)
subprocess.run(f"ffmpeg -y -v error -i {SPK_MP3} -ar 24000 -ac 1 {SPK_WAV}", shell=True)
print(f"   ✓ {os.path.getsize(SPK_WAV)//1024} KB", flush=True)

# ── ٣) ملف السيرفر ──
SERVER = r'''
import io, os, time, base64, tempfile, re
os.environ["COQUI_TOS_AGREED"] = "1"
import numpy as np, torch, soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SPK = "/content/spk.wav"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("DEV", dev, flush=True)

# ── كلام → نص ──
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
               device=0 if dev=="cuda" else -1,
               generate_kwargs={"language":"ar","task":"transcribe"})
print("ASR ok", flush=True)

# ── تفكير ──
MID = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.float16) if dev=="cuda" else None
llm = AutoModelForCausalLM.from_pretrained(MID, quantization_config=q,
      torch_dtype=torch.float16 if dev=="cuda" else torch.float32, device_map="auto")
llm.eval()
print("LLM ok", flush=True)

# ── صوت التقليد (XTTS-v2) ──
CLONE = False
xtts = None
try:
    from TTS.api import TTS
    xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    CLONE = True
    print("XTTS ok — تقليد صوت", flush=True)
except Exception as e:
    print("XTTS FAIL:", str(e)[:200], flush=True)

SYS = ("أنت مساعد صوتي عراقي اسمك أيصو. جاوب بجملة أو جملتين قصيرتين "
       "بلهجة عراقية بسيطة. بلا إيموجي، بلا تعداد، بلا رموز.")

def think(t):
    m=[{"role":"system","content":SYS},{"role":"user","content":t}]
    p=tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ids=tok(p, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        o=llm.generate(**ids, max_new_tokens=90, do_sample=True, temperature=0.7,
                       top_p=0.9, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def split_sent(t, maxn=180):
    parts = re.split(r'(?<=[.!؟،])\s+', t)
    out, cur = [], ""
    for p in parts:
        if len(cur)+len(p) < maxn: cur = (cur+" "+p).strip()
        else:
            if cur: out.append(cur)
            cur = p
    if cur: out.append(cur)
    return out or [t]

def speak_clone(t):
    """XTTS-v2 — يقول النص بصوت العيّنة"""
    chunks = []
    for seg in split_sent(t):
        if not seg.strip(): continue
        w = xtts.tts(text=seg, language="ar", speaker_wav=SPK, speed=1.12,
                     enable_text_splitting=False)
        chunks.append(np.asarray(w, dtype=np.float32))
    if not chunks: raise RuntimeError("no audio")
    w = np.concatenate(chunks)
    # رفع طفيف للطبقة — إحساس أنمي
    try:
        import librosa
        w = librosa.effects.pitch_shift(w, sr=24000, n_steps=1.0)
    except Exception:
        pass
    b = io.BytesIO(); sf.write(b, w, 24000, format="WAV")
    return b.getvalue()

def speak_edge(t):
    import asyncio, edge_tts
    async def go():
        c = edge_tts.Communicate(t, "ar-IQ-RanaNeural", pitch="+55Hz", rate="+10%")
        buf=b""
        async for ch in c.stream():
            if ch["type"]=="audio": buf += ch["data"]
        return buf
    return asyncio.run(go())

def speak(t):
    if CLONE:
        try:
            return speak_clone(t), "wav"
        except Exception as e:
            print("clone fail:", str(e)[:120], flush=True)
    return speak_edge(t), "mp3"

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"ok":True,"ready":True,"clone":CLONE}

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    if len(raw) < 2000:
        return {"heard":"","reply":"التسجيل قصير","audio":"","fmt":"none"}
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw); path = f.name
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception as e:
        try: os.unlink(path)
        except: pass
        return {"heard":"","reply":"ملف غير مقروء","audio":"","fmt":"none"}
    try: os.unlink(path)
    except: pass
    if data.ndim > 1: data = data.mean(1)
    if sr != 16000:
        import librosa; data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    if float(np.abs(data).max()) < 0.005:
        return {"heard":"","reply":"ما سمعت صوت — قرب المايك","audio":"","fmt":"none"}
    try:
        heard = asr({"raw":data.astype(np.float32),"sampling_rate":16000})["text"].strip()
    except Exception as e:
        return {"heard":"","reply":f"خطأ سمع: {str(e)[:40]}","audio":"","fmt":"none"}
    if not heard:
        return {"heard":"","reply":"ما كو شي واضح","audio":"","fmt":"none"}
    try: rep = think(heard)
    except Exception as e:
        return {"heard":heard,"reply":f"خطأ تفكير: {str(e)[:40]}","audio":"","fmt":"none"}
    try:
        blob, fmt = speak(rep); aud = base64.b64encode(blob).decode()
    except Exception as e:
        return {"heard":heard,"reply":rep,"audio":"","fmt":"none"}
    print(f"{time.time()-t0:.1f}s | {heard[:28]} | {rep[:28]}", flush=True)
    return {"heard":heard,"reply":rep,"audio":aud,"fmt":fmt}

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
'''
open("/content/ayso_srv.py","w").write(SERVER)
print("\n٣) ملف السيرفر ✓", flush=True)

# ── ٤) تشغيل ──
print("\n٤) تشغيل السيرفر…", flush=True)
open("/content/srv.log","w").close()
subprocess.Popen("nohup python -u /content/ayso_srv.py > /content/srv.log 2>&1 &",
                 shell=True, preexec_fn=os.setsid)

ready = False
for k in range(240):
    time.sleep(2)
    try: log = open("/content/srv.log", encoding="utf-8", errors="ignore").read()
    except Exception: log = ""
    if "XTTS ok" in log or "XTTS FAIL" in log: ready = True; break
    if "Traceback" in log:
        print("\n⚠️ خطأ:\n" + "\n".join(log.splitlines()[-20:])); break
    if k % 20 == 19: print(f"   {2*(k+1)}s …", flush=True)

if ready:
    for line in open("/content/srv.log", encoding="utf-8", errors="ignore"):
        if any(x in line for x in ("ASR ok","LLM ok","XTTS","clone","DEV")):
            print("   " + line.rstrip(), flush=True)
else:
    print("\n⚠️ ما جهز. آخر السجل:")
    print("\n".join(open("/content/srv.log",encoding="utf-8",errors="ignore").read().splitlines()[-20:]))

# ── ٥) النفق ──
print("\n٥) فتح النفق…", flush=True)
if not os.path.exists("/content/cf"):
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cf && chmod +x /content/cf", shell=True)
open("/content/cf.log","w").close()
subprocess.Popen("nohup /content/cf tunnel --url http://localhost:8000 --no-autoupdate > /content/cf.log 2>&1 &",
                 shell=True, preexec_fn=os.setsid)

url = None
for k in range(45):
    time.sleep(1)
    try: t = open("/content/cf.log", encoding="utf-8", errors="ignore").read()
    except Exception: t = ""
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", t)
    if m: url = m.group(0); break

print("\n" + "█"*56)
if url: print("  رابط السيرفر:\n  " + url)
else:
    print("  ⚠️ ما طلع رابط. آخر السجل:")
    print("\n".join(open("/content/cf.log",encoding="utf-8",errors="ignore").read().splitlines()[-8:]))
print("█"*56)
print("\n✅ الخلية خلصت. السيرفر بالخلفية — لا تسكّر التبويب.")
