"""ดึงไฟล์ export (JSON) ของทุกโปรเจกต์ใน Label Studio ที่ชื่อขึ้นต้นด้วย project_prefix

รัน    : python fetch.py   (อ่านค่าจาก configs/labelstudio_config.yml, token จาก .env: LABEL_STUDIO_TOKEN=<token>)
ผลลัพธ์: <output_dir>/<ชื่อโปรเจกต์>-<id>.json  (รันซ้ำ = เขียนทับไฟล์เดิมของโปรเจกต์นั้น)
"""
import json  # อ่านคำตอบของ API
import os  # อ่าน token จาก environment variable
import re  # ลบตัวอักษรที่ใช้เป็นชื่อไฟล์ไม่ได้
import urllib.request  # เรียก API (stdlib ไม่ต้องลงเพิ่ม)
from pathlib import Path  # จัดการพาธไฟล์

import yaml  # อ่านไฟล์ config .yml

ROOT = Path(__file__).parent  # โฟลเดอร์โปรเจกต์


def call(url, headers, data=None):
    """เรียก API แล้วคืน body เป็น bytes (data != None = POST JSON)"""
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:  # export โปรเจกต์ใหญ่อาจใช้เวลานาน
        return r.read()


def auth_headers(base, token):
    """Personal Access Token (JWT มีจุด 2 จุด) ต้องแลกเป็น access token ก่อน, legacy token ใช้ตรงๆ"""
    if token.count(".") == 2:
        access = json.loads(call(f"{base}/api/token/refresh", {}, {"refresh": token}))["access"]
        return {"Authorization": f"Bearer {access}"}
    return {"Authorization": f"Token {token}"}


def main():
    cfg = yaml.safe_load((ROOT / "configs/labelstudio_config.yml").read_text(encoding="utf-8"))
    env = ROOT / ".env"  # ไฟล์เก็บ token (ไม่ขึ้น git) บรรทัดละ KEY=VALUE
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))  # ตั้งใน terminal ไว้แล้ว = ใช้ค่านั้นก่อน
    token = os.environ.get("LABEL_STUDIO_TOKEN")
    if not token:
        raise SystemExit("ยังไม่ได้ตั้ง LABEL_STUDIO_TOKEN  ใส่ในไฟล์ .env เช่น  LABEL_STUDIO_TOKEN=<token>")
    base = cfg["url"].rstrip("/")
    headers = auth_headers(base, token)
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    projects, url = [], f"{base}/api/projects?page_size=100"
    while url:  # API แบ่งหน้า ไล่ตาม next จนหมด
        page = json.loads(call(url, headers))
        projects += page["results"]
        url = page.get("next")
    projects = [p for p in projects if p["title"].startswith(cfg["project_prefix"])]
    if not projects:
        raise SystemExit(f"ไม่พบโปรเจกต์ที่ชื่อขึ้นต้นด้วย '{cfg['project_prefix']}'")

    for p in projects:
        raw = call(f"{base}/api/projects/{p['id']}/export?exportType=JSON", headers)  # เฉพาะ task ที่ label แล้ว
        name = re.sub(r'[\\/:*?"<>|\s]+', "_", p["title"])  # ชื่อไฟล์ที่ Windows รับได้
        (out / f"{name}-{p['id']}.json").write_bytes(raw)
        print(f"{p['title']:<30} id={p['id']:<4} {len(json.loads(raw)):>5} task -> {out.name}/{name}-{p['id']}.json")


if __name__ == "__main__":
    main()
