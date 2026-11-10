# ═══════════════════════════════════════════════════════════
#  Ayso Chat — تدريب LoRA على نموذج جاهز (Colab T4 مجاني)
#  Runtime → Change runtime type → T4 GPU   ← قبل التشغيل
# ═══════════════════════════════════════════════════════════
import os, sys, subprocess, time, json, glob

# ── اختَر النموذج والساعة هون ────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"   # ٣B — مستقر على T4
# بدائل: "Qwen/Qwen2.5-1.5B-Instruct" (أسرع) · "Qwen/Qwen2.5-7B-Instruct" (أقوى، ضيّق)
MAX_SAMPLES = 2000       # كم سؤال من الدataset
EPOCHS      = 1
MAX_LEN     = 512

print("١) فحص الكارت…")
r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                   capture_output=True, text=True)
print("  ", r.stdout.strip() or "ما فيه كارت — بدّل النوع لـT4")

print("٢) تثبيت…")
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "transformers>=4.44","peft","trl","bitsandbytes","datasets","accelerate"], check=False)
import torch, transformers, peft, trl, bitsandbytes
print("   torch", torch.__version__, "| transformers", transformers.__version__)
print("   peft", peft.__version__, "| trl", trl.__version__, "| bitsandbytes", bitsandbytes.__version__)
print("   CUDA متاح:", torch.cuda.is_available())

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt  = torch.float16 if dev == "cuda" else torch.float32

print("٣) تنزيل النموذج", MODEL_ID)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True) if dev=="cuda" else None
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=quant, torch_dtype=dt,
    device_map="auto", trust_remote_code=True)
model.config.use_cache = False
n_par = sum(p.numel() for p in model.parameters())/1e9
print(f"   بارامترات: {n_par:.2f}B · النوع: {'4-bit' if quant else 'fp32'}")

print("٤) تجهيز البيانات…")
from datasets import load_dataset
ds = load_dataset("FreedomIntelligence/evol-instruct-arabic", split="train")
ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
print("   عيّنات:", len(ds))

def to_text(ex):
    msgs = []
    for m in ex["conversations"]:
        role = "user" if m["from"] in ("human","user") else "assistant"
        msgs.append({"role": role, "content": m["value"]})
    try:
        return {"text": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}
    except Exception:
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return {"text": s}
ds = ds.map(to_text, remove_columns=ds.column_names)
ds = ds.filter(lambda x: len(x["text"]) < 3000)
print("   بعد التنظيف:", len(ds))
print("   نموذج:", ascii(ds[0]["text"][:160]))

print("٥) LoRA…")
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
model = prepare_model_for_kbit_training(model)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type=TaskType.CAUSAL_LM,
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lcfg)
tr, tot = model.get_nb_trainable_parameters()
print(f"   قابلة للتدريب: {tr:,} / {tot:,} ({100*tr/tot:.2f}%)")

print("٦) التدريب…")
from trl import SFTTrainer, SFTConfig
args = SFTConfig(
    output_dir="/content/ayso_chat_lora",
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    num_train_epochs=EPOCHS, learning_rate=2e-4,
    lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=10, save_strategy="no", report_to="none",
    fp16=(dev=="cuda"), optim="paged_adamw_8bit" if dev=="cuda" else "adamw_torch",
    gradient_checkpointing=True, max_length=MAX_LEN, dataset_text_field="text",
)
trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
t0 = time.time()
trainer.train()
print(f"   خلص بـ {(time.time()-t0)/60:.1f} دقيقة")

print("٧) اختبار…")
model.eval()
def ask(q, n=80):
    msgs=[{"role":"user","content":q}]
    txt=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids=tok(txt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out=model.generate(**ids, max_new_tokens=n, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

for q in ["عرّفني عن نفسك بجملة", "شنو عاصمة العراق؟", "اكتب جملة ترحيب قصيرة"]:
    print(f"   س: {q}")
    try: print(f"   ج: {ask(q)[:220]}")
    except Exception as e: print("   خطأ:", str(e)[:80])

print("٨) حفظ…")
model.save_pretrained("/content/ayso_chat_lora_final")
tok.save_pretrained("/content/ayso_chat_lora_final")
subprocess.run("cd /content && zip -q ayso-chat-lora.zip -r ayso_chat_lora_final", shell=True)
sz = os.path.getsize("/content/ayso-chat-lora.zip")/1e6
print(f"   ayso-chat-lora.zip · {sz:.0f} MB")
print("╔══════════════════════════════╗")
print("║  ✅ خلص — نزّله الآن        ║")
print("╚══════════════════════════════╝")
try:
    from google.colab import files
    files.download("/content/ayso-chat-lora.zip")
except Exception:
    print("from google.colab import files; files.download('/content/ayso-chat-lora.zip')")
