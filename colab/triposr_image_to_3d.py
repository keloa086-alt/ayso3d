# ═══════════════════════════════════════════════════════════
#  Ayso 3D — صورة ← مجسّم ثلاثي الأبعاد (TripoSR · Colab T4)
#  Runtime → Change runtime type → T4 GPU   ← قبل التشغيل
# ═══════════════════════════════════════════════════════════
import os, sys, subprocess, time, glob, shutil

IMAGE = "/content/input.png"     # ← بدّل المسار لو صورتك باسم ثاني
#  أو ارفع صورة من جوالك داخل الخلية:
#  from google.colab import files; files.upload()
#  IMAGE = "/content/" + list(files.upload().keys())[0]

OUT_DIR   = "/content/triposr_out"
MC_RES    = 256          # دقة الشبكة — 256 جيد · 384 أدق وأبطأ
FORE_BG   = True         # إزالة الخلفية تلقائياً

print("١) فحص الكارت…")
r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                   capture_output=True, text=True)
print("  ", r.stdout.strip() or "ما فيه كارت — بدّل النوع لـT4")

if not os.path.exists(IMAGE):
    print(f"\n⚠️  ما لقيت الصورة: {IMAGE}")
    print("   ارفعها بهذي الخلية ثم أعد التشغيل:")
    print("   from google.colab import files")
    print("   up = files.upload(); IMAGE = '/content/' + list(up.keys())[0]")
    sys.exit(0)

print("٢) تثبيت المكتبات…")
subprocess.run([sys.executable,"-m","pip","install","-q",
    "torch","torchvision","trimesh","rembg","onnxruntime",
    "omegaconf","einops","huggingface_hub","opencv-python-headless"], check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","git+https://github.com/thu-ml/tri-plane-hash.git"], capture_output=True)
print("   جاهز")

print("٣) تنزيل TripoSR…")
if not os.path.exists("/content/TripoSR"):
    subprocess.run("cd /content && git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git",
                   shell=True, capture_output=True)
print("   ", "موجود" if os.path.exists("/content/TripoSR") else "فشل التنزيل")
sys.path.insert(0, "/content/TripoSR")

print("٤) تحميل النموذج…")
import torch
from tsr.system import TSR
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(32768)
model.to(dev)
model.eval()
print(f"   على {dev} · بارامترات {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

print("٥) تجهيز الصورة…")
from PIL import Image
import numpy as np
img = Image.open(IMAGE).convert("RGBA")
print(f"   الأصل: {img.size}")

if FORE_BG:
    try:
        import rembg
        img = rembg.remove(img, session=rembg.new_session())
        print("   خلفية انشالت ✓")
    except Exception as e:
        print("   rembg فشل (نكمل بلا إزالة):", str(e)[:60])

a = np.array(img)
if a[:,:,3].max() == 0:
    print("   ⚠️ الصورة فاضية بعد إزالة الخلفية — نكمل بالأصل")
    img = Image.open(IMAGE).convert("RGBA")

img.save("/content/prepared.png")

print("٦) التوليد…")
t0 = time.time()
from tsr.utils import remove_background, resize_foreground
proc = remove_background(img)
proc = resize_foreground(proc, 0.85)
arr = np.array(proc.convert("RGB")).astype(np.float32) / 255.0
arr = arr * np.array(proc.convert("RGBA"))[:, :, 3:4].astype(np.float32) / 255.0
arr = arr[np.newaxis, ...]

with torch.no_grad():
    scene_codes = model([torch.from_numpy(arr).to(dev)], device=dev)
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=MC_RES)
print(f"   خلص بـ {time.time()-t0:.1f} ثانية")

me = meshes[0]
print(f"   رؤوس {len(me.vertices):,} · وجوه {len(me.faces):,}")

print("٧) تنعيم + حفظ…")
try:
    me.fix_normals()
except Exception:
    pass
try:
    me = me.simplify_quadric_decimation(face_count=min(len(me.faces), 60000))
    print(f"   بعد التخفيف: {len(me.faces):,} وجه")
except Exception:
    pass

os.makedirs(OUT_DIR, exist_ok=True)
glb = os.path.join(OUT_DIR, "model.glb")
obj = os.path.join(OUT_DIR, "model.obj")
me.export(glb); me.export(obj)
print(f"   {glb} · {os.path.getsize(glb)/1e6:.1f} MB")

print("٨) عرض سريع…")
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    V = np.asarray(me.vertices); F = np.asarray(me.faces)
    if len(F) > 4000:
        F = F[np.random.choice(len(F), 4000, replace=False)]
    fig = plt.figure(figsize=(12,4), facecolor="white")
    for i,(el,az) in enumerate([(18,35),(0,0),(70,20)]):
        ax = fig.add_subplot(1,3,i+1, projection="3d")
        pc = Poly3DCollection(V[F], alpha=1.0)
        pc.set_facecolor("#4a90d9"); pc.set_edgecolor("none")
        ax.add_collection3d(pc)
        ax.auto_scale_xyz(V[:,0],V[:,1],V[:,2])
        ax.set_box_aspect((1,1,1)); ax.view_init(el,az); ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("/content/triposr_samples.png", dpi=95, facecolor="white", bbox_inches="tight")
    print("   ✓ /content/triposr_samples.png")
except Exception as e:
    print("   عرض فشل:", str(e)[:80])

print("٩) ضغط…")
shutil.copy("/content/prepared.png", OUT_DIR)
subprocess.run(f"cd {OUT_DIR} && zip -q /content/triposr-model.zip *", shell=True)
sz = os.path.getsize("/content/triposr-model.zip")/1e6
print(f"   triposr-model.zip · {sz:.1f} MB")

print("╔══════════════════════════════════╗")
print("║  ✅ خلص — نزّل الملف الآن        ║")
print("╚══════════════════════════════════╝")
try:
    from google.colab import files
    files.download("/content/triposr-model.zip")
except Exception:
    print("from google.colab import files; files.download('/content/triposr-model.zip')")
