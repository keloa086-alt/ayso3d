# ═══════════════════════════════════════════════════════════
#  Ayso Voice — سيرفر (Colab T4)
#  Runtime → Change runtime type → T4 GPU
# ═══════════════════════════════════════════════════════════
VERSION = "voice-srv-v1"
import os, sys, subprocess, time, io, base64, tempfile

print("═" * 56)
print(f"  Ayso Voice — السيرفر  |  {VERSION}")
print("═" * 56)

print("\n١) تثبيت…")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "fastapi", "uvicorn", "python-multipart", "transformers>=4.44",
                "accelerate", "soundfile", "librosa", "bitsandbytes"], check=False)
print("   ✓")

import numpy as np
import torch
import soundfile as sf
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("   جهاز:", dev.upper())

# ── سمع ──
print("\n٢) Whisper…")
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
               device=0 if dev == "cuda" else -1,
               generate_kwargs={"language": "ar", "task": "transcribe"})
print("   ✓")

# ── دماغ ──
print("\n٣) Qwen2.5-1.5B…")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
MID = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.float16) if dev == "cuda" else None
llm = AutoModelForCausalLM.from_pretrained(
    MID, quantization_config=q,
    torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
    device_map="auto")
llm.eval()
print("   ✓")

SYS = "أنت مساعد صوتي عراقي. جاوب بجملة أو جملتين قصيرة، بلهجة عراقية بسيطة، بلا إيموجي، بلا تعداد."


def think(text):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": text}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**ids, max_new_tokens=100, do_sample=True,
                           temperature=0.7, top_p=0.9,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── كلام ──
print("\n٤) MMS-TTS…")
from transformers import VitsModel, AutoTokenizer as AT
tts_tok = AT.from_pretrained("facebook/mms-tts-ara")
tts = VitsModel.from_pretrained("facebook/mms-tts-ara").to(dev).eval()
print("   ✓")


def speak(text):
    ids = tts_tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        wav = tts(**ids).waveform[0].cpu().numpy()
    bo = io.BytesIO()
    sf.write(bo, wav, tts.config.sampling_rate, format="WAV")
    return bo.getvalue()


# ── السيرفر ──
print("\n٥) السيرفر…")
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"ok": True, "v": VERSION,
            "usage": "POST /voice مع ملف wav (16kHz mono)"}


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    if len(raw) < 2000:
        return {"heard": "", "reply": "التسجيل قصير", "audio": ""}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw)
        path = f.name

    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception as e:
        return {"heard": "", "reply": f"ملف صوتي غير مقروء: {str(e)[:60]}", "audio": ""}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

    if data.ndim > 1:
        data = data.mean(1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    if float(np.abs(data).max()) < 0.005:
        return {"heard": "", "reply": "ما سمعت صوت — قرب المايك", "audio": ""}

    try:
        heard = asr({"raw": data.astype(np.float32), "sampling_rate": 16000})["text"].strip()
    except Exception as e:
        return {"heard": "", "reply": f"خطأ بالسمع: {str(e)[:60]}", "audio": ""}

    if not heard:
        return {"heard": "", "reply": "ما كو شي واضح", "audio": ""}

    try:
        rep = think(heard)
    except Exception as e:
        return {"heard": heard, "reply": f"خطأ بالتفكير: {str(e)[:60]}", "audio": ""}

    try:
        wav = speak(rep)
        aud = base64.b64encode(wav).decode()
    except Exception as e:
        return {"heard": heard, "reply": rep + f" (الصوت فشل: {str(e)[:40]})", "audio": ""}

    print(f"   {time.time()-t0:.1f}s | س: {heard[:40]} | ج: {rep[:40]}")
    return {"heard": heard, "reply": rep, "audio": aud}


print("   ✓")
print("\n٦) التشغيل…")
print("   الرابط يطلع تحت — انسخه وحطه بالتطبيق")
import threading
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error"),
                 daemon=True).start()
time.sleep(3)

# نفق عام (cloudflared)
import re
subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cf && chmod +x /content/cf", shell=True)
p = subprocess.Popen(["/content/cf", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
url = None
t0 = time.time()
while time.time() - t0 < 60:
    line = p.stdout.readline()
    if not line:
        break
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
        break

if url:
    print("\n" + "█" * 56)
    print("  رابط السيرفر:")
    print("  " + url)
    print("█" * 56)
    print("\nانسخه وحطه بخانة السيرفر بالتطبيق. اترك الخلية شغّالة.")
else:
    print("\n⚠️ ما طلع رابط — أعد تشغيل الخلية")

while True:
    time.sleep(60)
