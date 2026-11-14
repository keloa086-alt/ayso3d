# ═══ استخراج رابط النفق — صح ═══
import re, subprocess, sys, time, os

# تأكد السيرفر شغّال محلياً
r = subprocess.run("curl -s -m 5 http://localhost:8000/", shell=True, capture_output=True, text=True)
print("السيرفر محلياً:", r.stdout.strip()[:80] or "⚠️ مو شغّال")

if not os.path.exists("/content/cf"):
    print("تنزيل cloudflared…")
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cf && chmod +x /content/cf", shell=True)
print("cloudflared ✓\n")

print("فتح النفق… (١٠ ثواني)")
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
    print("\nانسخه وحطه بخانة السيرفر بالتطبيق.")
    print("⚠️ اترك هذا الكود شغّال — لا توقف الخلية.")
    # نبقي العملية حية
    while True:
        time.sleep(30)
        if p.poll() is not None:
            print("⚠️ النفق انقطع — شغّل الخلية من جديد")
            break
else:
    print("\n⚠️ ما طلع رابط. آخر ١٠ أسطر:")
    subprocess.run("echo", shell=True)
