// تدريب شبكة أصغر (MLP بطبقة مخفية) على بيانات حقيقية — بديل النموذج الخطي
// الاستخدام: node --max-old-space-size=1500 tools/train-mlp.mjs '{"N":24,"D":384,"H":192,"EPOCHS":20}'
import fs from "fs";

const src = fs.readFileSync("/home/user/tools/train-real.mjs", "utf8");
const cut = src.slice(src.indexOf("/* ── قراءة OFF ── */"), src.indexOf("/* ── تجهيز البيانات ── */"));
const { readOFF, voxelizeOFF } = new Function("fs", cut + "\nreturn {readOFF, voxelizeOFF};")(fs);

const cfg = { N: 24, D: 384, H: 192, EPOCHS: 20, LR: 0.01, BATCH: 4, NEG: 3, ...JSON.parse(process.argv[2] || "{}") };
const { N, D, H, EPOCHS } = cfg;
const K = N*N*N;

const CLASSES = {
  airplane:["airplane","طائرة","طائرة كبيرة","jet plane"],
  bathtub:["bathtub","حوض استحمام","حوض"],
  bed:["bed","سرير","سرير خشبي"],
  bench:["bench","مقعد","مقعد حديقة"],
  bookshelf:["bookshelf","رف كتب","مكتبة"],
  bottle:["bottle","قنينة","زجاجة"],
  bowl:["bowl","وعاء","صحن"],
  car:["car","سيارة","سيارة حمراء"],
  chair:["chair","كرسي","كرسي خشبي"],
  cone:["cone","مخروط","قمع"],
  cup:["cup","كأس","كوب"],
  curtain:["curtain","ستارة","ستائر"],
  desk:["desk","مكتب","طاولة مكتب"],
  door:["door","باب","باب خشبي"],
  dresser:["dresser","خزانة","خزانة ملابس"],
  flower_pot:["flower pot","اصيص","اصيص زهور"],
  glass_box:["glass box","صندوق زجاجي","علبة"],
  guitar:["guitar","قيثارة","جيتار"],
  keyboard:["keyboard","لوحة مفاتيح","كيبورد"],
  lamp:["lamp","مصباح","مصباح طاولة"],
  laptop:["laptop","حاسوب محمول","لابتوب"],
  mantel:["mantel","مدفأة","رف مدفأة"],
  monitor:["monitor","شاشة","شاشة حاسوب"],
  night_stand:["night stand","كومودينو","طاولة سرير"],
  person:["person","شخص","انسان"],
  piano:["piano","بيانو","بيانو كبير"],
  plant:["plant","نبات","نبتة"],
  radio:["radio","مذياع","راديو"],
  range_hood:["range hood","شفاط مطبخ","غطاء مطبخ"],
  sink:["sink","مغسلة","حوض غسل"],
  sofa:["sofa","أريكة","كنبة"],
  stairs:["stairs","درج","سلالم"],
  stool:["stool","كرسي بلا ظهر","تابور"],
  table:["table","طاولة","طاولة خشب"],
  tent:["tent","خيمة","خيمة كبيرة"],
  toilet:["toilet","مرحاض","حمام"],
  tv_stand:["tv stand","طاولة تلفاز","حامل تلفاز"],
  vase:["vase","مزهرية","مزهرية زهور"],
  wardrobe:["wardrobe","دولاب","خزانة ملابس"],
  xbox:["xbox","صندوق العاب","جهاز العاب"]
};

/* ── ميزات نصية (تجزئة) ── */
const h32 = s => { let x = 2166136261 >>> 0; for (let i=0;i<s.length;i++){ x ^= s.charCodeAt(i); x = Math.imul(x, 16777619) >>> 0; } return x >>> 0; };
function feat(text){
  const acc = new Float32Array(D);
  const t = "«" + text + "»";
  for (const w of t.split(/\s+/)) if (w) acc[h32("w:"+w) % D] += 1.8;
  for (let i=0;i+2<t.length;i++) acc[h32("3:"+t.slice(i,i+3)) % D] += 1.0;
  let n2 = 0; for (let i=0;i<D;i++) n2 += acc[i]*acc[i];
  const inv = n2 ? 1/Math.sqrt(n2) : 1;
  const ids = [], vals = [];
  for (let i=0;i<D;i++) if (acc[i]) { ids.push(i); vals.push(acc[i]*inv); }
  return { ids, vals };
}

/* ── البيانات ── */
const ROOT = "/tmp/mn40";
const files = fs.readdirSync(ROOT).filter(f=>f.endsWith(".off"));
const byClass = {};
for (const f of files) (byClass[f.replace(/_\d+\.off$/,"")] = byClass[f.replace(/_\d+\.off$/,"")] || []).push(ROOT+"/"+f);
const trainSet = [], testSet = [];
for (const cls of Object.keys(byClass)){
  const arr = byClass[cls].sort();
  const cutAt = Math.max(1, Math.floor(arr.length*0.8));
  trainSet.push(...arr.slice(0,cutAt).map(p=>({cls,p})));
  testSet.push(...arr.slice(cutAt).map(p=>({cls,p})));
}
const t0 = Date.now();
const occ = {};
let fillSum = 0;
for (const it of [...trainSet, ...testSet]){
  const o = voxelizeOFF(readOFF(it.p), N);
  occ[it.p] = o;
  let n=0; for (let q=0;q<K;q++) n+=o[q];
  it.fill = n/K; if (!testSet.includes(it)) fillSum += it.fill;
}
const meanFill = fillSum / trainSet.length;
console.log(`ملفات ${files.length} · تدريب ${trainSet.length} · اختبار ${testSet.length} · N=${N} D=${D} H=${H} · فوكسلة ${((Date.now()-t0)/1000).toFixed(0)}s · امتلاء ${(meanFill*100).toFixed(1)}%`);

const ds = [];
for (const it of trainSet){
  for (const nm of (CLASSES[it.cls]||[it.cls]).slice(0,3)) ds.push({ ...feat(nm), y: occ[it.p] });
}
console.log("عيّنات نصية:", ds.length);

/* ── المعمار: x → ReLU(xW1+b1) → logits = hW2+b2 ── */
const W1 = new Float32Array(D*H), b1b = new Float32Array(H);
const W2 = new Float32Array(H*K), b2b = new Float32Array(K);
const gW1 = new Float32Array(D*H), gb1 = new Float32Array(H);
const gW2 = new Float32Array(H*K), gb2 = new Float32Array(K);
const mW1 = new Float32Array(D*H), vW1 = new Float32Array(D*H);
const mW2 = new Float32Array(H*K), vW2 = new Float32Array(H*K);
const mb1 = new Float32Array(H), vb1 = new Float32Array(H);
const mb2 = new Float32Array(K), vb2 = new Float32Array(K);
/* تهيئة He */
for (let i=0;i<D*H;i++) W1[i] = (Math.random()*2-1) * Math.sqrt(2/D);
for (let i=0;i<H*K;i++) W2[i] = (Math.random()*2-1) * Math.sqrt(2/H);
b2b.fill(Math.log(meanFill/(1-meanFill)));

const hid = new Float32Array(H), zt = new Float32Array(K);
const sel = new Int32Array(K), gk = new Float32Array(K);

function forward(ids, vals){
  hid.fill(0);
  for (const i of ids) hid[i] = vals[ids.indexOf(i)];   /* مؤقت — يُستبدل أدناه */
}
function forward2(ids, vals){
  hid.fill(0);
  for (let a=0;a<ids.length;a++) hid[ids[a]] = vals[a];
  /* الطبقة 1: hid → pre-act */
  const pre = new Float32Array(H);
  for (let a=0;a<ids.length;a++){
    const base = ids[a]*H, v = vals[a];
    for (let j=0;j<H;j++) pre[j] += W1[base+j]*v;
  }
  for (let j=0;j<H;j++){ pre[j] += b1b[j]; hid[j] = pre[j] > 0 ? pre[j] : 0; }
  /* الطبقة 2 */
  for (let k=0;k<K;k++) zt[k] = b2b[k];
  for (let j=0;j<H;j++){
    const hv = hid[j]; if (hv === 0) continue;
    const base = j*K;
    for (let k=0;k<K;k++) zt[k] += W2[base+k]*hv;
  }
}

const b1=.9, b2=.999;
let step = 0;
const T0 = Date.now();
for (let ep=0; ep<EPOCHS; ep++){
  for (let i=ds.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; const t=ds[i]; ds[i]=ds[j]; ds[j]=t; }
  const lr = cfg.LR * (1 - .5*ep/EPOCHS);
  let loss = 0, nb = 0;
  for (const sm of ds){
    step++;
    forward2(sm.ids, sm.vals);
    /* اختيار الخلايا: كل الموجبة + سالبات عشوائية */
    let m = 0, pos = 0;
    for (let k=0;k<K;k++) if (sm.y[k]) { sel[m++]=k; pos++; }
    const wantNeg = Math.min(K-pos, pos*cfg.NEG);
    let tries = 0;
    while (m < pos+wantNeg && tries < K*4){ const k=(Math.random()*K)|0; if (!sm.y[k]) sel[m++]=k; tries++; }
    gk.fill(0);
    for (let q=0;q<m;q++){
      const k = sel[q], z = Math.max(-18, Math.min(18, zt[k]));
      const p = 1/(1+Math.exp(-z));
      loss += -(sm.y[k]*Math.log(p+1e-7) + (1-sm.y[k])*Math.log(1-p+1e-7)); nb++;
      gk[k] = p - sm.y[k];
    }
    /* تدرّج الطبقة 2 (للخلايا المختارة فقط) */
    const gh = new Float32Array(H);
    for (let j=0;j<H;j++){
      const hv = hid[j]; if (hv === 0) continue;
      const base = j*K; let s = 0;
      for (let q=0;q<m;q++){ const k = sel[q]; s += gk[k]*W2[base+k]; }
      gh[j] = s;
    }
    /* تحديث آدم */
    const c1 = 1-Math.pow(b1,step), c2 = 1-Math.pow(b2,step);
    const adam = (p, g, mm, vv, i) => {
      mm[i] = b1*mm[i] + (1-b1)*g;
      vv[i] = b2*vv[i] + (1-b2)*g*g;
      p[i] -= lr * (mm[i]/c1) / (Math.sqrt(vv[i]/c2) + 1e-8);
    };
    for (let j=0;j<H;j++){
      const hv = hid[j]; if (hv === 0) continue;
      const base = j*K;
      for (let q=0;q<m;q++){ const k = sel[q]; const g = gk[k]*hv; if (g) adam(W2, g, mW2, vW2, base+k); }
    }
    for (let q=0;q<m;q++){ const k = sel[q]; adam(b2b, gk[k], mb2, vb2, k); }
    /* تدرّج الطبقة 1 */
    for (let a=0;a<sm.ids.length;a++){
      const i = sm.ids[a], v = sm.vals[a], base = i*H;
      for (let j=0;j<H;j++){ const g = gh[j]*v; if (g) adam(W1, g, mW1, vW1, base+j); }
    }
    for (let j=0;j<H;j++){ const g = gh[j]; if (g) adam(b1b, g, mb1, vb1, j); }
  }
  console.log(`  epoch ${ep+1}/${EPOCHS} · loss ${(loss/nb).toFixed(4)} · ${((Date.now()-T0)/1000).toFixed(0)}s`);
}

/* ── تقييم ── */
function predict(ids, vals, thr){
  forward2(ids, vals);
  const out = new Uint8Array(K);
  for (let k=0;k<K;k++) out[k] = zt[k] > thr ? 1 : 0;
  return out;
}
function iou(a,b){ let i=0,u=0; for (let k=0;k<K;k++){ const x=a[k]?1:0,y=b[k]?1:0; if(x&&y)i++; if(x||y)u++; } return u?i/u:0; }
const comb = (a,b) => { const o = new Uint8Array(K); for (let k=0;k<K;k++) o[k] = a[k] && b[k] ? 1 : 0; return o; };
const thr = Math.log(meanFill/(1-meanFill));
let sTr=0, nTr=0;
for (const sm of ds){ sTr += iou(predict(sm.ids, sm.vals, thr), sm.y); nTr++; }
let sUn=0, nUn=0;
for (const it of trainSet){
  const nm = (CLASSES[it.cls]||[it.cls])[3] || it.cls;
  const f = feat(nm);
  sUn += iou(predict(f.ids, f.vals, thr), occ[it.p]); nUn++;
}
let sInst=0, nInst=0;
for (const it of testSet){
  for (const nm of (CLASSES[it.cls]||[it.cls]).slice(0,2)){
    const f = feat(nm);
    sInst += iou(predict(f.ids, f.vals, thr), occ[it.p]); nInst++;
  }
}
console.log(JSON.stringify({ N, D, H, K, epochs:EPOCHS, samples:ds.length,
  trainIou:+(sTr/nTr*100).toFixed(2), unseenText:+(sUn/nUn*100).toFixed(2), unseenShape:+(sInst/nInst*100).toFixed(2),
  secs:((Date.now()-T0)/1000).toFixed(0) }));

/* ── حفظ int4 (W2 هو الأكبر) ── */
function quant(W, rows, cols){
  const nib = new Uint8Array(Math.ceil(W.length/2)), sc = new Float32Array(rows);
  for (let r=0;r<rows;r++){ let mx=0; for (let c=0;c<cols;c++){ const a=Math.abs(W[r*cols+c]); if(a>mx) mx=a; } sc[r]=mx/7||1e-6; }
  for (let i=0;i<W.length;i++){ const r=Math.floor(i/cols);
    let q = Math.round(W[i]/sc[r]) + 8; q = Math.max(0,Math.min(15,q));
    if (i&1) nib[i>>1] |= (q<<4); else nib[i>>1] |= q; }
  return { nib, sc };
}
const q2 = quant(W2, H, K), q1 = quant(W1, D, H);
const parts = [Buffer.from(q2.nib), Buffer.from(q2.sc.buffer), Buffer.from(b2b.buffer),
               Buffer.from(q1.nib), Buffer.from(q1.sc.buffer), Buffer.from(b1b.buffer)];
const head = Buffer.alloc(16);
head.write("AYMLP1\0\0", 0, "latin1");
head.writeUInt16LE(N, 8); head.writeUInt16LE(D, 10); head.writeUInt16LE(H, 12);
const buf = Buffer.concat([head, ...parts]);
fs.writeFileSync("/home/user/tools/model-mlp.bin", buf);
console.log("model-mlp.bin MB", (buf.length/1048576).toFixed(2), "| params", (D*H + H*K));
fs.writeFileSync("/home/user/tools/mlp-meta.json", JSON.stringify({ N, D, H, K, epochs:EPOCHS, samples:ds.length,
  trainIou:+(sTr/nTr*100).toFixed(2), unseenText:+(sUn/nUn*100).toFixed(2), unseenShape:+(sInst/nInst*100).toFixed(2),
  meanFill, bytes:buf.length, at:new Date().toISOString() }));
