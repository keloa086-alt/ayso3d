# ═══════════════════════════════════════════════════════════
#  Ayso 3D — تدريب بكولاب: خلية واحدة، فهم مباشر
#  Runtime → Change runtime type → T4 GPU  ← قبل التشغيل
# ═══════════════════════════════════════════════════════════
import sys, os, time
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
print("١) فحص الكارت…")
import torch, torch.nn as nn
print("   GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ ماكو كارت")

print("٢) تثبيت…")
os.system("pip -q install trimesh scikit-image 2>/dev/null")

print("٣) تنزيل ١٢ مجسّم لكل صنف…")
from huggingface_hub import list_repo_files, hf_hub_download
CLASSES = ["airplane","bed","bench","bookshelf","bottle","bowl","car","chair","cone","cup",
           "desk","door","guitar","lamp","laptop","monitor","plant","sofa","table","vase"]
allf=[f for f in list_repo_files("naderalfares/ModelNet40", repo_type="dataset")
      if f.startswith("ModelNet40/") and "/train/" in f and f.endswith(".off")]
need=[]
for c in CLASSES:
    need += sorted(f for f in allf if f.startswith(f"ModelNet40/{c}/train/"))[:12]
print("   ملفات:", len(need))
for i,f in enumerate(need):
    d="/content/data/"+f
    if not os.path.exists(d): hf_hub_download("naderalfares/ModelNet40", f, repo_type="dataset", local_dir="/content/data")
    if (i+1)%60==0: print("  ", i+1)

print("٤) فوكسلة…")
import numpy as np, glob
def read_off(p):
    L=open(p,'rb').read().decode('latin1').split('\n'); i=0
    while 'OFF' not in L[i]: i+=1
    h=L[i].replace('OFF','').strip().split()
    if len(h)>=2: nv,nf=int(h[0]),int(h[1]); i+=1
    else: i+=1; h=L[i].split(); nv,nf=int(h[0]),int(h[1]); i+=1
    V=np.array(' '.join(L[i:i+nv]).split(),dtype=np.float32).reshape(nv,3); i+=nv
    b=' '.join(L[i:i+nf]).split(); F=[];k=0
    while k<len(b):
        m=int(b[k])
        if m==3: F.append((int(b[k+1]),int(b[k+2]),int(b[k+3]))); k+=4
        elif m==4: F.append((int(b[k+1]),int(b[k+2]),int(b[k+3]))); F.append((int(b[k+1]),int(b[k+3]),int(b[k+4]))); k+=5
        else: k+=1+m
    return V,np.array(F,np.int32)
def vox(V,F,N=32):
    lo,hi=np.percentile(V,0.5,axis=0),np.percentile(V,99.5,axis=0)
    k=np.all((V>=lo)&(V<=hi),axis=1)
    if k.sum()>100: V=V[k]
    mn,mx=V.min(0),V.max(0); sp=float((mx-mn).max()) or 1
    G=np.empty_like(V)
    G[:,0]=(V[:,0]-(mn[0]+mx[0])/2)/sp*N+N/2
    G[:,1]=(V[:,2]-mn[2])/sp*N
    G[:,2]=(V[:,1]-(mn[1]+mx[1])/2)/sp*N+N/2
    o=np.zeros((N,N,N),bool)
    iv=np.floor(G).astype(int); ok=np.all((iv>=0)&(iv<N),axis=1)
    o[iv[ok,0],iv[ok,1],iv[ok,2]]=True
    if len(F)>8000: F=F[np.random.choice(len(F),8000,replace=False)]
    a,b_,c=G[F[:,0]],G[F[:,1]],G[F[:,2]]
    e=np.max(np.stack([np.linalg.norm(b_-a,axis=1),np.linalg.norm(c-b_,axis=1),np.linalg.norm(a-c,axis=1)]),0)
    S=np.clip(np.ceil(e*1.5),2,12).astype(int)
    for j in range(len(F)):
        s=S[j];u=np.arange(s+1);uu,vv=np.meshgrid(u,u,indexing='ij');m=(uu+vv)<=s
        t1=uu[m]/s;t2=vv[m]/s;t0=1-t1-t2
        p=a[j]*t0[:,None]+b_[j]*t1[:,None]+c[j]*t2[:,None]
        ip=np.floor(p).astype(int);ok=np.all((ip>=0)&(ip<N),axis=1)
        o[ip[ok,0],ip[ok,1],ip[ok,2]]=True
    for _ in range(2):
        pd=np.pad(o,1)
        cnt=sum(pd[1+dx:1+dx+N,1+dy:1+dy+N,1+dz:1+dz+N] for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1) if (dx,dy,dz)!=(0,0,0))
        o|=cnt>=7
    air=np.zeros_like(o)
    air[0,:,:]=air[-1,:,:]=air[:,0,:]=air[:,-1,:]=air[:,:,0]=air[:,:,-1]=True; air&=~o
    while True:
        pd=np.pad(air,1)
        g=(pd[0:N,1:N+1,1:N+1]|pd[2:N+2,1:N+1,1:N+1]|pd[1:N+1,0:N,1:N+1]|pd[1:N+1,2:N+2,1:N+1]|pd[1:N+1,1:N+1,0:N]|pd[1:N+1,1:N+1,2:N+2])&~o
        nw=air|g
        if np.array_equal(nw,air): break
        air=nw
    return ~air
OCCS,CLS=[],[]
for c in CLASSES:
    for f in sorted(glob.glob(f"/content/data/ModelNet40/{c}/train/*.off"))[:12]:
        try:
            V,F=read_off(f)
            if len(V)>20 and len(F)>10: OCCS.append(vox(V,F)); CLS.append(c)
        except Exception: pass
    print(f"   {c:12s} {sum(1 for x in CLS if x==c)}")
OCCS=np.array(OCCS,bool); CLS=np.array(CLS)
print("   المجموع:", OCCS.shape, "| امتلاء", round(OCCS.mean()*100,2),"%")

print("٥) الموديل والتدريب…")
D=2048
def feat(t,D=D):
    a=np.zeros(D,np.float32)
    def h(s):
        x=2166136261
        for ch in s.encode(): x^=ch; x=(x*16777619)&0xFFFFFFFF
        return x
    for w in t.strip().lower().split():
        a[h("w:"+w)%D]+=2.2
        for i in range(len(w)): a[h("3:"+w[max(0,i-1):i+2])%D]+=1.0
    n=np.linalg.norm(a); return a/n if n>0 else a
NAMES={"airplane":["airplane","طائرة","jet plane"],"bed":["bed","سرير","wooden bed"],
"bench":["bench","مقعد","park bench"],"bookshelf":["bookshelf","رف كتب","shelf"],
"bottle":["bottle","قنينة","glass bottle"],"bowl":["bowl","وعاء","deep bowl"],
"car":["car","سيارة","sedan"],"chair":["chair","كرسي","wooden chair"],
"cone":["cone","مخروط","traffic cone"],"cup":["cup","كأس","coffee cup"],
"desk":["desk","مكتب","office desk"],"door":["door","باب","wooden door"],
"guitar":["guitar","قيثارة","acoustic guitar"],"lamp":["lamp","مصباح","table lamp"],
"laptop":["laptop","حاسوب محمول","notebook"],"monitor":["monitor","شاشة","screen"],
"plant":["plant","نبات","green plant"],"sofa":["sofa","أريكة","couch"],
"table":["table","طاولة","dining table"],"vase":["vase","مزهرية","ceramic vase"]}
class Gen(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(D,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU())
        s.fc=nn.Linear(256,128*64)
        s.u1=nn.Sequential(nn.ConvTranspose3d(128,96,4,2,1),nn.BatchNorm3d(96),nn.ReLU())
        s.u2=nn.Sequential(nn.ConvTranspose3d(96,64,4,2,1),nn.BatchNorm3d(64),nn.ReLU())
        s.u3=nn.Sequential(nn.ConvTranspose3d(64,32,4,2,1),nn.BatchNorm3d(32),nn.ReLU())
        s.out=nn.Conv3d(32,1,1)
    def forward(s,x):
        h=s.enc(x); h=s.fc(h).view(-1,128,4,4,4)
        return s.out(s.u3(s.u2(s.u1(h)))).squeeze(1)
dev="cuda" if torch.cuda.is_available() else "cpu"
M=Gen().to(dev)
print("   بارامترات:", f"{sum(p.numel() for p in M.parameters()):,}")
X=[];Y=[]
for o,c in zip(OCCS,CLS):
    for nm in NAMES.get(c,[c]): X.append(feat(nm)); Y.append(o.astype(np.float32))
X=torch.from_numpy(np.array(X)); Y=torch.from_numpy(np.array(Y))
print("   عيّنات:", len(X))
opt=torch.optim.AdamW(M.parameters(),lr=2e-3,weight_decay=1e-4)
EP=30; BS=16
steps=max(1,len(X)//BS)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=EP*steps)
pw=torch.tensor(float(1/max(OCCS.mean(),1e-3)-1)).to(dev)
crit=nn.BCEWithLogitsLoss(pos_weight=pw)
t0=time.time()
for e in range(EP):
    perm=torch.randperm(len(X)); tot=0;n=0
    for i in range(steps):
        idx=perm[i*BS:(i+1)*BS]
        x,y=X[idx].to(dev),Y[idx].to(dev)
        loss=crit(M(x),y); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(M.parameters(),1.0); opt.step(); sch.step()
        tot+=loss.item(); n+=1
    print(f"   epoch {e+1:2d}/{EP} · loss {tot/n:.4f} · {(time.time()-t0)/60:.1f}د")

print("٦) التقييم…")
M.eval()
def pred(t):
    with torch.no_grad():
        p=torch.sigmoid(M(torch.from_numpy(feat(t)).unsqueeze(0).to(dev)))[0].cpu().numpy()
    return p>.5,p
def iou(a,b):
    i=np.logical_and(a,b).sum(); u=np.logical_or(a,b).sum(); return i/u if u else 0
s=0
for o,c in zip(OCCS,CLS): s+=iou(pred(NAMES.get(c,[c])[0])[0],o)
print("   IoU نصوص شُوفت :", round(s/len(OCCS)*100,1),"%")
s=0
for o,c in zip(OCCS,CLS): s+=iou(pred(c.replace('_',' '))[0],o)
print("   IoU نصوص جديدة :", round(s/len(OCCS)*100,1),"%")
avg={c:(OCCS[CLS==c].mean(0)>.5) for c in set(CLS)}
print("   سقف الشكل المتوسط:", round(sum(iou(avg[c],o) for o,c in zip(OCCS,CLS))/len(OCCS)*100,1),"%")
print("   خط الأساس فاضي :", round((1-OCCS.mean())*100,1),"%")

print("٧) تصدير…")
torch.save(M.state_dict(),"/content/ayso3d_state.pt")
try:
    torch.onnx.export(M,torch.zeros(1,D).to(dev),"/content/ayso3d.onnx",
        input_names=["text"],output_names=["logits"],
        dynamic_axes={"text":{0:"batch"},"logits":{0:"batch"}},opset_version=17)
    print("   ONNX ✓")
except Exception as ex: print("   ONNX فشل:",str(ex)[:90])
try:
    from skimage import measure; import trimesh
    for t in ["chair","table","car","lamp"]:
        _,pr=pred(t)
        v,f,_,_=measure.marching_cubes(pr.astype(np.float32),level=.5)
        me=trimesh.Trimesh(v,f,process=True); me.apply_scale(1/32)
        me.export(f"/content/{t}.glb"); print(f"   {t}.glb ✓ {len(me.faces)}")
except Exception as ex: print("   GLB فشل:",str(ex)[:90])
os.system("cd /content && zip -q ayso3d-model.zip ayso3d_state.pt ayso3d.onnx *.glb 2>/dev/null")
print("╔══════════════════════════════╗")
print("║  ✅ خلص — نزّل الملف الآن   ║")
print("╚══════════════════════════════╝")
try:
    from google.colab import files
    files.download("/content/ayso3d-model.zip")
except Exception: print("نزّله يدوياً من مجلد /content")
