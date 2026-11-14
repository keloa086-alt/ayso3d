# ═══════════════════════════════════════════════════════════
#  Ayso Voice — خلية واحدة. تنتهي بنفسها والسيرفر يبقى شغّال.
#  Runtime → T4 GPU   |   قبل التشغيل: اضغط ⏹ Stop لو خلية ثانية شغّالة
# ═══════════════════════════════════════════════════════════
import os, sys, subprocess, time, re, textwrap

print("═"*56)
print("  Ayso Voice — تشغيل السيرفر")
print("═"*56)

# ── ٠) نظّف القديم ──
subprocess.run("pkill -f ayso_srv.py; pkill cloudflared", shell=True)
time.sleep(1)

# ── ١) المكتبات ──
print("\n١) تثبيت…")
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "fastapi","uvicorn","python-multipart","transformers>=4.44","accelerate",
    "soundfile","librosa","bitsandbytes"], check=False)
print("   ✓")

# ── ٢) ملف السيرفر ──
SERVER = r'''
import io, os, time, base64, tempfile, threading
import numpy as np, torch, soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, VitsModel
from transformers import AutoTokenizer as AT

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("DEV", dev, flush=True)

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

tts_tok = AT.from_pretrained("facebook/mms-tts-ara")
tts = VitsModel.from_pretrained("facebook/mms-tts-ara").to(dev).eval()
print("TTS ok", flush=True)

SYS = "أنت مساعد صوتي عراقي. جاوب بجملة أو جملتين قصيرة بلهجة عراقية بسيطة، بلا إيموجي."

def think(t):
    m=[{"role":"system","content":SYS},{"role":"user","content":t}]
    p=tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ids=tok(p, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        o=llm.generate(**ids, max_new_tokens=100, do_sample=True, temperature=0.7,
                       top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def speak(t):
    ids=tts_tok(t, return_tensors="pt").to(dev)
    with torch.no_grad():
        w=tts(**ids).waveform[0].cpu().numpy()
    b=io.BytesIO(); sf.write(b, w, tts.config.sampling_rate, format="WAV")
    return b.getvalue()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
app=FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"ok":True,"ready":True}

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    t0=time.time()
    raw=await file.read()
    if len(raw)<2000: return {"heard":"","reply":"التسجيل قصير","audio":""}
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw); path=f.name
    try:
        data,sr=sf.read(path, dtype="float32")
    except Exception as e:
        try: os.unlink(path)
        except: pass
        return {"heard":"","reply":f"ملف غير مقروء: {str(e)[:50]}","audio":""}
    try: os.unlink(path)
    except: pass
    if data.ndim>1: data=data.mean(1)
    if sr!=16000:
        import librosa; data=librosa.resample(data, orig_sr=sr, target_sr=16000)
    if float(np.abs(data).max())<0.005:
        return {"heard":"","reply":"ما سمعت صوت — قرب المايك","audio":""}
    try: heard=asr({"raw":data.astype(np.float32),"sampling_rate":16000})["text"].strip()
    except Exception as e: return {"heard":"","reply":f"خطأ سمع: {str(e)[:50]}","audio":""}
    if not heard: return {"heard":"","reply":"ما كو شي واضح","audio":""}
    try: rep=think(heard)
    except Exception as e: return {"heard":heard,"reply":f"خطأ تفكير: {str(e)[:50]}","audio":""}
    try: aud=base64.b64encode(speak(rep)).decode()
    except Exception as e: return {"heard":heard,"reply":rep,"audio":""}
    print(f"{time.time()-t0:.1f}s | {heard[:30]} | {rep[:30]}", flush=True)
    return {"heard":heard,"reply":rep,"audio":aud}

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
'''
open("/content/ayso_srv.py","w").write(SERVER)
print("\n٢) ملف السيرفر ✓")

# ── ٣) تشغيل بالخلفية ──
print("\n٣) تشغيل السيرفر…")
open("/content/srv.log","w").close()
subprocess.Popen("nohup python -u /content/ayso_srv.py > /content/srv.log 2>&1 &",
                 shell=True, preexec_fn=os.setsid)

print("   انتظار التحميل (حتى ٥ دقايق)…")
ready=False
for k in range(150):
    time.sleep(2)
    try: log=open("/content/srv.log", encoding="utf-8", errors="ignore").read()
    except Exception: log=""
    if "TTS ok" in log: ready=True; break
    if '"ready":true' in log: ready=True; break
    r=subprocess.run("curl -s -m 2 http://localhost:8000/", shell=True, capture_output=True, text=True)
    if '"ok"' in r.stdout: ready=True; break
    if "Error" in log or "Traceback" in log:
        print("\n⚠️ خطأ بالسيرفر:\n" + "\n".join(log.splitlines()[-15:])); break
    if k%15==14: print(f"   {2*(k+1)}s …")

if not ready:
    print("\n⚠️ السيرفر ما جهز. آخر السجل:")
    try: print("\n".join(open("/content/srv.log",encoding="utf-8",errors="ignore").read().splitlines()[-15:]))
    except Exception: pass
else:
    print("   السيرفر جاهز ✓")

# ── ٤) النفق ──
print("\n٤) فتح النفق…")
if not os.path.exists("/content/cf"):
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cf && chmod +x /content/cf", shell=True)
open("/content/cf.log","w").close()
subprocess.Popen("nohup /content/cf tunnel --url http://localhost:8000 --no-autoupdate > /content/cf.log 2>&1 &",
                 shell=True, preexec_fn=os.setsid)

url=None
for k in range(45):
    time.sleep(1)
    try: t=open("/content/cf.log", encoding="utf-8", errors="ignore").read()
    except Exception: t=""
    m=re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", t)
    if m: url=m.group(0); break
    if k%10==9: print(f"   {k+1}s …")

print("\n" + "█"*56)
if url:
    print("  رابط السيرفر:")
    print("  " + url)
else:
    print("  ⚠️ ما طلع رابط. آخر السجل:")
    try: print("\n".join(open("/content/cf.log",encoding="utf-8",errors="ignore").read().splitlines()[-8:]))
    except Exception: pass
print("█"*56)
print("\n✅ الخلية خلصت. السيرفر يشتغل بالخلفية — لا تسكّر التبويب.")
