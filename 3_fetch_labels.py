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


def main():  # จุดเริ่มของขั้นที่ 3
    cfg = load_config("configs/labelstudio.yml", ["url", "fetch.project_prefix", "fetch.output_dir"])  # อ่าน config + ตรวจว่ามี key ครบ
    prefix, out = cfg["fetch"]["project_prefix"], Path(cfg["fetch"]["output_dir"])  # คำนำหน้าชื่อโปรเจกต์ / โฟลเดอร์ปลายทาง
    ls = LabelStudio(cfg["url"])  # เชื่อม server (อ่าน API key จาก .env)
    out.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

    projects = [p for p in ls.projects() if p["title"].startswith(prefix)]  # เฉพาะโปรเจกต์ที่ชื่อขึ้นต้นด้วย prefix
    if not projects:  # ไม่เจอเลย
        raise SystemExit(f"ไม่พบโปรเจกต์ที่ชื่อขึ้นต้นด้วย '{prefix}'")  # หยุดพร้อมบอกสาเหตุ
    for p in projects:  # ทีละโปรเจกต์
        tasks = ls.export(p["id"])  # เฉพาะ task ที่ label แล้ว
        name = re.sub(r'[\\/:*?"<>|\s]+', "_", p["title"])  # ชื่อไฟล์ที่ Windows รับได้
        path = out / f"{name}-{p['id']}.json"  # เช่น data/label/teenoi-data-9.json
        path.write_text(json.dumps(tasks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")  # รูปแบบเดียวกับไฟล์ export ของ Label Studio
        print(f"{p['title']:<30} id={p['id']:<4} {len(tasks):>5} task -> {path}")  # บอกจำนวน task ที่ได้


if __name__ == "__main__":  # รันตรงด้วย python 3_fetch_labels.py (ไม่รันตอนถูก import เช่นใน tests)
    main()  # เริ่มทำงาน
