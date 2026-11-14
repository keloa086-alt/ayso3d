# ═══════════════════════════════════════════════════════════════
#  Ayso Voice — نطق نظيف · صوت عراقي أنثوي (Edge-TTS)
#  Runtime → T4 GPU   |   أوقف أي خلية شغّالة قبل التشغيل (⏹)
# ═══════════════════════════════════════════════════════════════
import os, sys, subprocess, time, re

print("═"*56); print("  Ayso Voice — نطق نظيف · عراقي"); print("═"*56)

subprocess.run("pkill -f ayso_srv.py 2>/dev/null; pkill cloudflared 2>/dev/null", shell=True)
time.sleep(1)

# ── ١) المكتبات ──
print("\n١) تثبيت…", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "torch","torchaudio","fastapi","uvicorn","python-multipart",
    "transformers","accelerate","soundfile","librosa",
    "bitsandbytes","edge-tts"], check=False)
print("   ✓", flush=True)

# ── ٢) ملف السيرفر ──
SERVER = r'''
import io, os, time, base64, tempfile, asyncio
import numpy as np, torch, soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import edge_tts

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("DEV", dev, flush=True)

# ═══ إعدادات الصوت — عدّل هذي بس ═══
VOICE = "ar-IQ-RanaNeural"     # عراقية أنثى
PITCH = ""                     # بلا تعديل = أوضح نطق
RATE  = "-5%"                  # أبطأ قليلاً = أوضح
GEMINI_KEY = os.environ.get("GEMINI_KEY","")  # اختياري: مفتاح Gemini
# ═══════════════════════════════════

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
               device=0 if dev=="cuda" else -1,
               generate_kwargs={"language":"ar","task":"transcribe"})
print("ASR ok", flush=True)

MID = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.float16) if dev=="cuda" else None
llm = AutoModelForCausalLM.from_pretrained(MID, quantization_config=q,
      torch_dtype=torch.float16 if dev=="cuda" else torch.float32, device_map="auto")
llm.eval()
print("LLM ok", flush=True)

SYS = ("أنت مساعد صوتي عراقي اسمك أيصو. جاوب بجملة أو جملتين قصيرتين "
       "بلهجة عراقية بسيطة. بلا إيموجي، بلا تعداد، بلا رموز، بلا نقاط متعددة.")

def think(t):
    m = [{"role":"system","content":SYS},{"role":"user","content":t}]
    p = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        o = llm.generate(**ids, max_new_tokens=90, do_sample=True, temperature=0.75,
                         top_p=0.9, repetition_penalty=1.1,
                         pad_token_id=tok.eos_token_id)
    out = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # نظّف النص قبل النطق
    import re as _re
    out = _re.sub(r"[*_#\[\]()<>|~`]", "", out)
    out = _re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", out)
    out = _re.sub(r"\s+", " ", out).strip()
    return out

def speak_gemini(text):
    """Gemini TTS — أوضح صوت ممكن (يحتاج مفتاح مجاني)"""
    import urllib.request as _u, json as _j
    body = _j.dumps({"contents":[{"parts":[{"text":text}]}],
        "generationConfig":{"responseModalities":["AUDIO"],
        "speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}}}).encode()
    req = _u.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash-preview-tts:generateContent?key=" + GEMINI_KEY,
        data=body, headers={"Content-Type":"application/json"})
    d = _j.loads(_u.urlopen(req, timeout=60).read())
    pcm = base64.b64decode(d["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
    # PCM 16-bit 24k mono → WAV
    import wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(pcm)
    return b.getvalue()

async def _tts(text):
    # pitch و rate معاً يفشلان — نستخدم pitch فقط
    for kw in ([dict(pitch=PITCH, rate=RATE)] if PITCH else [dict(rate=RATE)], [{}]):
        try:
            c = edge_tts.Communicate(text, VOICE, **kw)
            buf = b""
            async for ch in c.stream():
                if ch["type"] == "audio":
                    buf += ch["data"]
            if len(buf) > 800:
                return buf
        except Exception as e:
            print("tts retry:", str(e)[:60], flush=True)
    raise RuntimeError("tts empty")

def speak(t):
    import re as _re
    if GEMINI_KEY:
        try: return speak_gemini(t), "wav"
        except Exception as e: print("gemini fail:", str(e)[:70], flush=True)
    # قسّم لجمل — النطق يصير أنظف على الجمل القصيرة
    parts = [p.strip() for p in _re.split(r"(?<=[.!؟])\s+", t) if p.strip()]
    if not parts: parts = [t]
    out = b""
    for s in parts:
        try: out += asyncio.run(_tts(s))
        except Exception: pass
    if not out: raise RuntimeError("no audio")
    return out, "mp3"

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"ok":True,"ready":True,"voice":VOICE}

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
    except Exception:
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
    except Exception:
        return {"heard":heard,"reply":rep,"audio":"","fmt":"none"}
    print(f"{time.time()-t0:.1f}s | {heard[:26]} | {rep[:26]}", flush=True)
    return {"heard":heard,"reply":rep,"audio":aud,"fmt":fmt}

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
'''
open("/content/ayso_srv.py","w").write(SERVER)
print("\n٢) ملف السيرفر ✓", flush=True)

# ── ٣) تشغيل ──
print("\n٣) تشغيل السيرفر…", flush=True)
open("/content/srv.log","w").close()
subprocess.Popen("nohup python -u /content/ayso_srv.py > /content/srv.log 2>&1 &",
                 shell=True, preexec_fn=os.setsid)

ready = False
for k in range(180):
    time.sleep(2)
    try: log = open("/content/srv.log", encoding="utf-8", errors="ignore").read()
    except Exception: log = ""
    if "LLM ok" in log: ready = True; break
    if "Traceback" in log:
        print("\n⚠️ خطأ:\n" + "\n".join(log.splitlines()[-20:])); break
    if k % 15 == 14: print(f"   {2*(k+1)}s …", flush=True)

if ready:
    print("   ASR ok · LLM ok · صوت: " + VOICE, flush=True)
else:
    print("\n⚠️ ما جهز. آخر السجل:")
    print("\n".join(open("/content/srv.log",encoding="utf-8",errors="ignore").read().splitlines()[-20:]))

# ── ٤) النفق ──
print("\n٤) فتح النفق…", flush=True)
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
