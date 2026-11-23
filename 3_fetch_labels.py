"""ขั้นที่ 3 ดึงผล label (JSON) ของทุกโปรเจกต์ใน Label Studio ที่ชื่อขึ้นต้นด้วย fetch.project_prefix

รัน    : python 3_fetch_labels.py   (อ่านค่าจาก configs/labelstudio.yml ส่วน fetch, key จาก .env: LABEL_STUDIO_API_KEY=<key>)
ผลลัพธ์: <fetch.output_dir>/<ชื่อโปรเจกต์>-<id>.json  (รันซ้ำ = เขียนทับไฟล์เดิมของโปรเจกต์นั้น)
"""
import json  # เขียนไฟล์ export
import os  # os.chdir
import re  # ลบตัวอักษรที่ใช้เป็นชื่อไฟล์ไม่ได้
from pathlib import Path  # จัดการพาธไฟล์

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo

from teenoi.config import load_config  # noqa: E402  อ่าน + ตรวจ config
from teenoi.labelstudio import LabelStudio  # noqa: E402  เรียก Label Studio API


def main():
    cfg = load_config("configs/labelstudio.yml", ["url", "fetch.project_prefix", "fetch.output_dir"])
    prefix, out = cfg["fetch"]["project_prefix"], Path(cfg["fetch"]["output_dir"])
    ls = LabelStudio(cfg["url"])
    out.mkdir(parents=True, exist_ok=True)

    projects = [p for p in ls.projects() if p["title"].startswith(prefix)]
    if not projects:
        raise SystemExit(f"ไม่พบโปรเจกต์ที่ชื่อขึ้นต้นด้วย '{prefix}'")
    for p in projects:
        tasks = ls.export(p["id"])  # เฉพาะ task ที่ label แล้ว
        name = re.sub(r'[\\/:*?"<>|\s]+', "_", p["title"])  # ชื่อไฟล์ที่ Windows รับได้
        path = out / f"{name}-{p['id']}.json"
        path.write_text(json.dumps(tasks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")  # รูปแบบเดียวกับไฟล์ export ของ Label Studio
        print(f"{p['title']:<30} id={p['id']:<4} {len(tasks):>5} task -> {path}")


if __name__ == "__main__":
    main()
