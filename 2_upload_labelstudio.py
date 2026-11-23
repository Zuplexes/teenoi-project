"""ขั้นที่ 2 upload รีวิวที่ clean แล้วขึ้น Label Studio: python 2_upload_labelstudio.py (ตั้งค่าใน configs/labelstudio.yml ส่วน upload)
สร้างโปรเจกต์ให้ถ้ายังไม่มี และรีวิวที่ upload แล้วจะไม่ถูกส่งซ้ำ"""
import csv  # อ่าน CSV
import os  # os.chdir
from pathlib import Path  # จัดการ path

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo

from teenoi.config import load_config  # noqa: E402  อ่าน + ตรวจ config
from teenoi.labelstudio import LabelStudio  # noqa: E402  เรียก Label Studio API

CONFIG = "configs/labelstudio.yml"  # ไฟล์ config ของขั้นนี้ (ใช้ส่วน url และ upload:)


def read_rows(paths):  # อ่านรีวิวจาก CSV หลายไฟล์ คืน dict review_id -> แถว
    rows = {}  # review_id -> row (ตัดซ้ำข้ามไฟล์)
    for p in paths:  # ทีละไฟล์
        with open(p, encoding="utf-8", newline="") as f:  # newline="" ตามที่ csv module แนะนำ
            for r in csv.DictReader(f):  # ทีละแถว (dict ตามหัวคอลัมน์)
                rows[str(r["review_id"])] = {"review_id": r["review_id"], "review": r["review"]}  # เก็บเฉพาะ 2 คอลัมน์ที่ Label Studio ใช้ (id ซ้ำ = ทับของเดิม)
    return rows  # รีวิวที่ไม่ซ้ำทั้งหมด


def main():  # จุดเริ่มของขั้นที่ 2
    cfg = load_config(CONFIG, ["url", "upload.project_title", "upload.label_config", "upload.files"])  # อ่าน config + ตรวจว่ามี key ครบ
    up = cfg["upload"]  # ส่วน upload:
    ls = LabelStudio(cfg["url"])  # เชื่อม server (อ่าน API key จาก .env)
    pid = next((p["id"] for p in ls.projects() if p["title"] == up["project_title"]), None)  # id ของโปรเจกต์ชื่อนี้ (ไม่เจอ = None)
    if pid is None:  # ยังไม่มีโปรเจกต์ -> สร้างพร้อม Labeling Interface
        xml = Path(up["label_config"]).read_text(encoding="utf-8")  # Labeling Interface ที่ใช้สร้างโปรเจกต์
        pid = ls.call("POST", "/api/projects", {"title": up["project_title"], "label_config": xml})["id"]  # สร้างโปรเจกต์ใหม่ แล้วเก็บ id
        print(f"สร้างโปรเจกต์ใหม่ '{up['project_title']}' (id={pid})")  # บอกผู้ใช้

    rows = read_rows(up["files"])  # รีวิวทั้งหมดในไฟล์ที่ตั้งไว้
    done = {str(t["data"].get("review_id")) for t in ls.export(pid, all_tasks=True)}  # review_id ที่อยู่ในโปรเจกต์แล้ว
    new = [r for rid, r in rows.items() if rid not in done]  # เฉพาะรีวิวที่ยังไม่อยู่ในโปรเจกต์
    print(f"รีวิวในไฟล์ {len(rows)} | มีในโปรเจกต์แล้ว {len(rows) - len(new)} | จะ upload {len(new)}")  # สรุปก่อน upload

    n = up.get("batch_size", 500)  # จำนวนรีวิวต่อ 1 request
    for i in range(0, len(new), n):  # ส่งทีละ batch (i = 0, n, 2n, ...)
        ls.call("POST", f"/api/projects/{pid}/import", new[i:i + n])  # import task เข้าโปรเจกต์
        print(f"  upload แล้ว {min(i + n, len(new))}/{len(new)}")  # ความคืบหน้า
    print(f"เสร็จ: {ls.url}/projects/{pid}")  # ลิงก์โปรเจกต์ไว้เปิดไป label ต่อ


if __name__ == "__main__":  # รันตรงด้วย python 2_upload_labelstudio.py (ไม่รันตอนถูก import เช่นใน tests)
    main()  # เริ่มทำงาน
