# ═══════════════════════════════════════════════════════════
#  Ayso 3D — تدريب نموذج صورة ← فوكسل (من الصفر · Colab T4)
#  Runtime → Change runtime type → T4 GPU
# ═══════════════════════════════════════════════════════════
VERSION = "img3d-v1"
import os, sys, subprocess, time, glob, random

print("═"*56)
print(f"  Ayso 3D — صورة ← فوكسل  |  {VERSION}")
print("═"*56)

subprocess.run([sys.executable,"-m","pip","install","-q","scikit-image","trimesh",
                "huggingface_hub"], check=False)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("جهاز:", dev.upper())

# ── ١) البيانات ──
from huggingface_hub import list_repo_files, hf_hub_download
CLASSES = ["airplane","bed","bench","bookshelf","bottle","bowl","car","chair","cone","cup",
           "desk","door","guitar","lamp","laptop","monitor","plant","sofa","table","vase"]
PER_CLASS = 40           # 40 × 20 = 800 مجسّم
print(f"\n١) تنزيل {PER_CLASS} مجسّم لكل صنف ({PER_CLASS*len(CLASSES)} ملف)…")
allf = [f for f in list_repo_files("naderalfares/ModelNet40", repo_type="dataset")
        if f.startswith("ModelNet40/") and "/train/" in f and f.endswith(".off")]
need = []
for c in CLASSES:
    need += sorted(f for f in allf if f.startswith(f"ModelNet40/{c}/train/"))[:PER_CLASS]
for i, f in enumerate(need):
    d = "/content/data/" + f
    if not os.path.exists(d):
        hf_hub_download("naderalfares/ModelNet40", f, repo_type="dataset", local_dir="/content/data")
    if (i+1) % 200 == 0: print("  ", i+1)
print("   ملفات:", len(need))

# ── ٢) فوكسلة + إسقاط ──
def _clean(V, F):
    V = np.asarray(V, np.float32); F = np.asarray(F, np.int32)
    ok = (F >= 0).all(1) & (F < len(V)).all(1)
    return V, F[ok]

def read_off(p):
    try:
        import trimesh
        m = trimesh.load(p, force="mesh", process=False)
        V, F = _clean(m.vertices, m.faces)
        if len(V) > 20 and len(F) > 10: return V, F
    except Exception: pass
    raw = open(p,"rb").read(); i = raw.find(b"OFF")
    lines = raw[i+3:].decode("latin1").split("\n"); k = 0
    while not lines[k].strip(): k += 1
    h = lines[k].split(); nv, nf = int(h[0]), int(h[1]); k += 1
    t = "\n".join(lines[k:]).split()
    V = np.array(t[:nv*3], np.float32).reshape(nv,3); rest = t[nv*3:]
    Fc, j = [], 0
    while j < len(rest) and len(Fc) < nf*2:
        m2 = int(float(rest[j]))
        if m2 in (3,4):
            idx = [int(float(x)) for x in rest[j+1:j+1+m2]]
            Fc.append(idx[:3])
            if m2 == 4: Fc.append([idx[0],idx[2],idx[3]])
            j += 1+m2
        else: j += 1+m2
    return _clean(V, np.array(Fc, np.int32))

def vox(V, F, N=32):
    lo, hi = np.percentile(V,0.5,axis=0), np.percentile(V,99.5,axis=0)
    k = np.all((V>=lo)&(V<=hi), axis=1)
    if 0 < k.sum() < len(V):
        r = -np.ones(len(V), np.int64); r[k] = np.arange(int(k.sum()))
        F = r[F]; F = F[(F>=0).all(1)]; V = V[k]
    if len(V) < 20 or len(F) < 20: raise ValueError("small")
    mn, mx = V.min(0), V.max(0); sp = float((mx-mn).max()) or 1
    G = np.empty_like(V)
    G[:,0] = (V[:,0]-(mn[0]+mx[0])/2)/sp*N + N/2
    G[:,1] = (V[:,2]-mn[2])/sp*N
    G[:,2] = (V[:,1]-(mn[1]+mx[1])/2)/sp*N + N/2
    o = np.zeros((N,N,N), bool)
    if len(F) > 6000: F = F[np.random.choice(len(F), 6000, replace=False)]
    a,b,c = G[F[:,0]], G[F[:,1]], G[F[:,2]]
    r = np.random.rand(len(F),2); m = r.sum(1) > 1; r[m] = 1-r[m]
    p = a + r[:,0:1]*(b-a) + r[:,1:2]*(c-a)
    ip = np.floor(p).astype(int); ok = np.all((ip>=0)&(ip<N), axis=1)
    o[ip[ok,0], ip[ok,1], ip[ok,2]] = True
    for _ in range(2):
        pd = np.pad(o,1)
        cnt = sum(pd[1+dx:1+dx+N,1+dy:1+dy+N,1+dz:1+dz+N]
                  for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1) if (dx,dy,dz)!=(0,0,0))
        o |= cnt >= 7
    air = np.zeros_like(o)
    air[0,:,:]=air[-1,:,:]=air[:,0,:]=air[:,-1,:]=air[:,:,0]=air[:,:,-1]=True; air &= ~o
    while True:
        pd = np.pad(air,1)
        g = (pd[0:N,1:N+1,1:N+1]|pd[2:N+2,1:N+1,1:N+1]|pd[1:N+1,0:N,1:N+1]|
             pd[1:N+1,2:N+2,1:N+1]|pd[1:N+1,1:N+1,0:N]|pd[1:N+1,1:N+1,2:N+2]) & ~o
        nw = air|g
        if np.array_equal(nw,air): break
        air = nw
    return ~air

def views(o):
    """٣ إسقاطات: عمق + ظل، من ٣ محاور → 6 قنوات 32×32"""
    ch = []
    for ax in (2,0,1):
        idx = o.argmax(axis=ax).astype(np.float32)
        has = o.any(axis=ax)
        d = np.where(has, idx/32.0, 1.0)
        s = has.astype(np.float32)
        ch.append(d); ch.append(s)
    return np.stack(ch).astype(np.float32)   # (6,32,32)

print("\n٢) فوكسلة + إسقاط…")
X, Y, CL = [], [], []
FAIL = 0
t0 = time.time()
for i, f in enumerate(need):
    c = f.split("/")[2]
    try:
        V, Fc = read_off("/content/data/" + f)
        o = vox(V, Fc)
        if o.mean() < 0.005: FAIL += 1; continue
        X.append(views(o)); Y.append(o.astype(np.float32)); CL.append(c)
    except Exception:
        FAIL += 1
    if (i+1) % 200 == 0:
        print(f"   {i+1}/{len(need)} · نجح {len(X)} · {time.time()-t0:.0f}s")
X = np.array(X); Y = np.array(Y); CL = np.array(CL)
print(f"   نجح {len(X)} · فشل {FAIL} · X{X.shape} Y{Y.shape} · امتلاء {Y.mean()*100:.2f}%")

# ── ٣) النموذج ──
class Img2Vox(nn.Module):
    def __init__(s, N=32):
        super().__init__(); s.N = N
        s.enc = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.GroupNorm(8,64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.GroupNorm(8,64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.GroupNorm(8,128), nn.SiLU(),
            nn.Conv2d(128,128, 3, padding=1, stride=2), nn.GroupNorm(8,128), nn.SiLU(),
            nn.Flatten(), nn.Linear(128*4*4, 512), nn.SiLU(),
            nn.Linear(512, 4*4*4*256), nn.SiLU(),
        )
        s.dec = nn.Sequential(
            nn.ConvTranspose3d(256,128,4,stride=2,padding=1), nn.GroupNorm(8,128), nn.SiLU(),
            nn.ConvTranspose3d(128, 64,4,stride=2,padding=1), nn.GroupNorm(8,64), nn.SiLU(),
            nn.ConvTranspose3d( 64, 32,4,stride=2,padding=1), nn.GroupNorm(8,32), nn.SiLU(),
            nn.Conv3d(32,1,1),
        )
    def forward(s, x):
        z = s.enc(x).view(-1,256,4,4,4)
        return s.dec(z).squeeze(1)

torch.manual_seed(0); np.random.seed(0)
M = Img2Vox().to(dev)
print(f"\n٣) بارامترات: {sum(p.numel() for p in M.parameters()):,}")

# ── ٤) تدريب ──
from sklearn.model_selection import train_test_split
idx = np.arange(len(X))
itr, ite = train_test_split(idx, test_size=0.2, random_state=0, stratify=CL)
Xt = torch.from_numpy(X[itr]); Yt = torch.from_numpy(Y[itr])
Xv = torch.from_numpy(X[ite]).to(dev); Yv = torch.from_numpy(Y[ite]).to(dev)

EP, BS = 60, 16
opt = torch.optim.AdamW(M.parameters(), lr=2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP*max(1,len(itr)//BS))
pw = torch.tensor(float(1/max(Y.mean(),1e-3)-1)).to(dev)
crit = nn.BCEWithLogitsLoss(pos_weight=pw)

print(f"\n٤) تدريب {EP} epoch · {len(itr)} تدريب · {len(ite)} اختبار")
t0 = time.time()
for e in range(EP):
    M.train(); perm = torch.randperm(len(Xt)); tot = n = 0
    for i in range(0, len(Xt), BS):
        b = perm[i:i+BS]
        x, y = Xt[b].to(dev), Yt[b].to(dev)
        loss = crit(M(x), y); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); sch.step()
        tot += loss.item(); n += 1
    if (e+1) % 10 == 0 or e == 0:
        M.eval()
        with torch.no_grad():
            pv = (M(Xv) > 0).float()
        inter = (pv*Yv).sum().item(); uni = ((pv+Yv) > 0).sum().item()
        print(f"   epoch {e+1:2d}/{EP} · loss {tot/n:.4f} · IoU اختبار {inter/uni*100:5.1f}% · {(time.time()-t0)/60:.1f}د")

# ── ٥) تقييم نهائي ──
M.eval()
with torch.no_grad(): pv = (M(Xv) > 0).float()
inter = (pv*Yv).sum().item(); uni = ((pv+Yv) > 0).sum().item()
iou = inter/uni
print(f"\n٥) النتيجة: IoU اختبار {iou*100:.1f}%")
print(f"   خط أساس فاضي: {(1-Y.mean())*100:.1f}%")
print(f"   امتلاء متوقع {pv.mean()*100:.1f}% مقابل حقيقي {Yv.mean()*100:.1f}%")
per = {}
for c in sorted(set(CL[ite])):
    m = CL[ite] == c
    i2 = (pv[m]*Yv[m]).sum().item(); u2 = ((pv[m]+Yv[m])>0).sum().item()
    per[c] = i2/u2 if u2 else 0
print("   أفضل ٣:", ", ".join(f"{k} {v*100:.0f}%" for k,v in sorted(per.items(), key=lambda x:-x[1])[:3]))
print("   أسوأ ٣:", ", ".join(f"{k} {v*100:.0f}%" for k,v in sorted(per.items(), key=lambda x:x[1])[:3]))

# ── ٦) حفظ + معاينة ──
torch.save(M.state_dict(), "/content/ayso_img3d.pt")
print(f"\n٦) ayso_img3d.pt · {os.path.getsize('/content/ayso_img3d.pt')/1e6:.1f} MB")
try:
    torch.onnx.export(M, torch.zeros(1,6,32,32).to(dev), "/content/ayso_img3d.onnx",
        input_names=["views"], output_names=["logits"],
        dynamic_axes={"views":{0:"batch"},"logits":{0:"batch"}},
        opset_version=17, dynamo=False)
    print("   ONNX ✓")
except Exception as ex:
    print("   ONNX فشل:", str(ex)[:80])

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    k = min(4, len(ite))
    fig = plt.figure(figsize=(4*k, 8), facecolor="white")
    for j in range(k):
        for row, (arr, ttl, col) in enumerate([(Yv[j].cpu().numpy(), "حقيقي", "#5aa469"),
                                              (pv[j].cpu().numpy(), "نموذجنا", "#4a90d9")]):
            ax = fig.add_subplot(2, k, row*k + j + 1, projection="3d")
            g = arr.astype(np.float32)
            if g.sum() > 10:
                v, f, _, _ = measure.marching_cubes(g, level=0.5)
                pc = Poly3DCollection(v[f], alpha=1.0); pc.set_facecolor(col); pc.set_edgecolor("none")
                ax.add_collection3d(pc); ax.auto_scale_xyz(v[:,0],v[:,1],v[:,2])
            ax.set_box_aspect((1,1,1)); ax.view_init(20,35); ax.set_axis_off()
            ax.set_title(f"{CL[ite][j]} — {ttl}", fontsize=12)
    plt.tight_layout()
    plt.savefig("/content/img3d_samples.png", dpi=95, facecolor="white", bbox_inches="tight")
    print("   img3d_samples.png ✓")
except Exception as ex:
    print("   عرض فشل:", str(ex)[:80])

subprocess.run("cd /content && zip -q -j ayso-img3d.zip ayso_img3d.pt ayso_img3d.onnx img3d_samples.png 2>/dev/null", shell=True)
print("╔══════════════════════════════════╗")
print("║  ✅ خلص — نزّل الملف            ║")
print("╚══════════════════════════════════╝")
try:
    from google.colab import files
    files.download("/content/ayso-img3d.zip")
except Exception:
    print("from google.colab import files; files.download('/content/ayso-img3d.zip')")
