# ═══════════════════════════════════════════════════════════════
#  Ayso — تدريب LoRA فوق Qwen2.5-3B-Instruct
#  Runtime → Change runtime type → T4 GPU   |   الوقت: ~٢٥ دقيقة
# ═══════════════════════════════════════════════════════════════
import os, sys, subprocess, json, time

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR  = "/content/ayso-lora"
DATA     = "/content/ayso_commands.jsonl"
DATA_URL = "https://raw.githubusercontent.com/keloa086-alt/ayso3d/main/train/ayso_commands.jsonl"

print("═" * 56)
print("  Ayso — تدريب LoRA")
print("═" * 56)

# ── ٠) الكارت ──
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True)
print("الكارت:", r.stdout.strip() or "⚠️ ماكو — بدّل لـT4 GPU")

# ── ١) المكتبات ──
print("\n١) تثبيت…", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=4.44,<5", "peft", "trl", "bitsandbytes",
    "datasets", "accelerate"], check=False)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
print("   torch", torch.__version__, flush=True)

# ── ٢) البيانات ──
print("\n٢) البيانات…", flush=True)
if not os.path.exists(DATA):
    subprocess.run(f'wget -q -O {DATA} "{DATA_URL}"', shell=True)
if not os.path.exists(DATA) or os.path.getsize(DATA) < 1000:
    print("   ⚠️ ما نزّلت البيانات. ارفع ayso_commands.jsonl يدوياً لـ/content/")
    sys.exit(1)
ds = load_dataset("json", data_files=DATA, split="train")
print(f"   {len(ds)} مثال ✓", flush=True)

# ── ٣) النموذج ──
print("\n٣) تحميل النموذج…", flush=True)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
print("   ✓", flush=True)

# ── ٤) LoRA ──
print("\n٤) LoRA…", flush=True)
lcfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

# ── ٥) التنسيق ──
def fmt(ex):
    return {"text": tok.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt, remove_columns=ds.column_names)

# ── ٦) التدريب ──
print("\n٥) التدريب…", flush=True)
t0 = time.time()
cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    max_length=512,
    dataset_text_field="text",
    report_to="none",
    fp16=True,
)
tr = SFTTrainer(model=model, train_dataset=ds, args=cfg, processing_class=tok)
tr.train()
print(f"   ✓ انتهى بـ {(time.time()-t0)/60:.1f} دقيقة", flush=True)

# ── ٧) الحفظ ──
print("\n٦) الحفظ…", flush=True)
tr.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
size = sum(os.path.getsize(os.path.join(OUT_DIR, f))
           for f in os.listdir(OUT_DIR) if os.path.isfile(os.path.join(OUT_DIR, f)))
print(f"   ✓ {OUT_DIR} — {size/1e6:.1f} م.ب", flush=True)

# ── ٨) اختبار سريع ──
print("\n٧) اختبار…", flush=True)
from peft import PeftModel
model.eval()
tests = ["دخل واتساب", "افتحلي اليوتيوب", "دوس على البحث",
         "شنو اسمك", "احب الريف", "انزل تحت"]
SYS = ("أنت أيصو، وكيل صوتي عراقي يتحكم بجوال أندرويد. "
       "المستخدم يعطيك أمر، وأنت ترجع JSON فقط بلا شرح.")
for q in tests:
    msg = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**ids, max_new_tokens=60, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    out = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"   {q:22s} → {out[:80]}", flush=True)

# ── ٩) الضغط للتنزيل ──
print("\n٨) الضغط…", flush=True)
subprocess.run(["zip", "-qr", "/content/ayso-lora.zip", OUT_DIR], check=False)
print("   ✓ /content/ayso-lora.zip", flush=True)

try:
    from google.colab import files
    files.download("/content/ayso-lora.zip")
    print("\n✅ نزّل الملف — ابعته هنا للدمج")
except Exception:
    print("\n✅ الملف: /content/ayso-lora.zip")
print("═" * 56)
