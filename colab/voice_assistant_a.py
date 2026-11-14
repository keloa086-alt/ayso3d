# ═══════════════════════════════════════════════════════════
#  Ayso Voice — الخطوة أ: محادثة صوتية عربية
#  Runtime → Change runtime type → T4 GPU
# ═══════════════════════════════════════════════════════════
VERSION = "voice-a-v1"
import os, sys, subprocess, time

print("═"*56)
print(f"  Ayso Voice — الخطوة أ  |  {VERSION}")
print("═"*56)

print("\n١) تثبيت…")
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "transformers>=4.44","accelerate","gradio","soundfile","librosa",
    "bitsandbytes","sentencepiece"], check=False)
print("   ✓")

import torch, numpy as np
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("   جهاز:", dev.upper())

# ── ٢) السمع ──
print("\n٢) Whisper (سمع عربي)…")
from transformers import pipeline
asr = pipeline("automatic-speech-recognition",
               model="openai/whisper-small",
               device=0 if dev=="cuda" else -1,
               generate_kwargs={"language":"ar","task":"transcribe"})
print("   ✓")

# ── ٣) الدماغ ──
print("\n٣) Qwen2.5-1.5B (تفكير عربي)…")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
MID = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.float16) if dev=="cuda" else None
llm = AutoModelForCausalLM.from_pretrained(MID, quantization_config=q,
        torch_dtype=torch.float16 if dev=="cuda" else torch.float32,
        device_map="auto")
llm.eval()
print(f"   ✓ {sum(p.numel() for p in llm.parameters())/1e9:.2f}B")

SYS = "أنت مساعد صوتي عراقي. جاوب بجملة أو جملتين قصيرة، بلهجة عراقية بسيطة، بلا إيموجي."

def think(text):
    msgs = [{"role":"system","content":SYS},{"role":"user","content":text}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**ids, max_new_tokens=120, do_sample=True,
                           temperature=0.7, top_p=0.9,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── ٤) الكلام ──
print("\n٤) MMS-TTS (كلام عربي)…")
from transformers import VitsModel, AutoTokenizer as AT
tts_tok = AT.from_pretrained("facebook/mms-tts-ara")
tts = VitsModel.from_pretrained("facebook/mms-tts-ara").to(dev).eval()
print("   ✓")

def speak(text):
    ids = tts_tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        wav = tts(**ids).waveform[0].cpu().numpy()
    return wav, tts.config.sampling_rate

# ── ٥) الواجهة ──
print("\n٥) الواجهة…")
import gradio as gr

def pipeline_all(audio):
    if audio is None:
        return "", "ارفع أو سجّل صوت", None
    sr, data = audio
    if data.dtype != np.float32:
        data = data.astype(np.float32) / (np.abs(data).max() + 1e-9)
    if data.ndim > 1:
        data = data.mean(1)
    data = data.reshape(-1)
    # Whisper يحتاج 16k
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    t0 = time.time()
    try:
        txt = asr({"raw": data, "sampling_rate": 16000})["text"].strip()
    except Exception as e:
        return "", f"خطأ بالسمع: {str(e)[:80]}", None
    if not txt:
        return "", "ما سمعت شي — جرّب مرة ثانية", None

    try:
        rep = think(txt)
    except Exception as e:
        return txt, f"خطأ بالتفكير: {str(e)[:80]}", None

    try:
        wav, sr2 = speak(rep)
        import soundfile as sf
        path = "/content/reply.wav"
        sf.write(path, wav, sr2)
    except Exception as e:
        return txt, rep + f"\n(الكلام فشل: {str(e)[:50]})", None

    print(f"   {time.time()-t0:.1f}s | س: {txt[:40]} | ج: {rep[:40]}")
    return txt, rep, path

with gr.Blocks(title="Ayso Voice — خطوة أ") as demo:
    gr.Markdown("## Ayso Voice — الخطوة أ\nاضغط المايك، تكلم عربي، ويرد عليك بصوت.")
    with gr.Row():
        inp = gr.Audio(sources=["microphone","upload"], type="numpy", label="تكلم")
    with gr.Row():
        out_txt = gr.Textbox(label="سمعت", lines=2)
        out_rep = gr.Textbox(label="الرد", lines=3)
    out_aud = gr.Audio(label="صوت الرد", autoplay=True)
    btn = gr.Button("▶ رد", variant="primary")
    btn.click(pipeline_all, inp, [out_txt, out_rep, out_aud])

print("   ✓")
print("\n٦) تشغيل…")
demo.launch(share=True, debug=False)
