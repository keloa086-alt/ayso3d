# ═══ إصلاح: torchmcubes ← skimage (يشتغل بلا ترجمة CUDA) ═══
import sys, subprocess, os
subprocess.run([sys.executable,"-m","pip","install","-q","scikit-image"], check=False)

open("/content/torchmcubes.py","w").write('import torch, numpy as np\nfrom skimage import measure\n\ndef marching_cubes(vol, thresh, **kw):\n    v = np.asarray(vol.detach().cpu().numpy() if hasattr(vol, "detach") else vol)\n    if v.ndim == 4:\n        v = v[0]\n    v = v.astype(np.float32)\n    if not np.isfinite(v).all() or v.min() >= thresh or v.max() <= thresh:\n        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.int64)\n    verts, faces, _, _ = measure.marching_cubes(v, level=float(thresh))\n    return (torch.from_numpy(verts.astype(np.float32)),\n            torch.from_numpy(faces.astype(np.int64)))\n')
if "/content" not in sys.path: sys.path.insert(0, "/content")
import importlib; importlib.invalidate_caches()
import torchmcubes
print("torchmcubes بديل ✓ |", torchmcubes.__file__)

# ── تحميل النموذج ──
import torch
from tsr.system import TSR
MODEL = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
MODEL.renderer.set_chunk_size(32768)
MODEL.to("cuda"); MODEL.eval()
print("✓ جاهز على T4 — روح للخلية ٢")
