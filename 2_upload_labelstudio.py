"""ขั้นที่ 2 upload รีวิวที่ clean แล้วขึ้น Label Studio: python 2_upload_labelstudio.py (ตั้งค่าใน configs/labelstudio.yml ส่วน upload)
สร้างโปรเจกต์ให้ถ้ายังไม่มี และรีวิวที่ upload แล้วจะไม่ถูกส่งซ้ำ"""
import csv  # อ่าน CSV
import os  # os.chdir
from pathlib import Path  # จัดการ path

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo

from teenoi.config import load_config  # noqa: E402  อ่าน + ตรวจ config
from teenoi.labelstudio import LabelStudio  # noqa: E402  เรียก Label Studio API

CONFIG = "configs/labelstudio.yml"


def read_rows(paths):
    rows = {}  # review_id -> row (ตัดซ้ำข้ามไฟล์)
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows[str(r["review_id"])] = {"review_id": r["review_id"], "review": r["review"]}
    return rows


def main():
    cfg = load_config(CONFIG, ["url", "upload.project_title", "upload.label_config", "upload.files"])
    up = cfg["upload"]
    ls = LabelStudio(cfg["url"])
    pid = next((p["id"] for p in ls.projects() if p["title"] == up["project_title"]), None)
    if pid is None:  # ยังไม่มีโปรเจกต์ -> สร้างพร้อม Labeling Interface
        xml = Path(up["label_config"]).read_text(encoding="utf-8")
        pid = ls.call("POST", "/api/projects", {"title": up["project_title"], "label_config": xml})["id"]
        print(f"สร้างโปรเจกต์ใหม่ '{up['project_title']}' (id={pid})")

    rows = read_rows(up["files"])
    done = {str(t["data"].get("review_id")) for t in ls.export(pid, all_tasks=True)}  # review_id ที่อยู่ในโปรเจกต์แล้ว
    new = [r for rid, r in rows.items() if rid not in done]
    print(f"รีวิวในไฟล์ {len(rows)} | มีในโปรเจกต์แล้ว {len(rows) - len(new)} | จะ upload {len(new)}")

    n = up.get("batch_size", 500)
    for i in range(0, len(new), n):
        ls.call("POST", f"/api/projects/{pid}/import", new[i:i + n])
        print(f"  upload แล้ว {min(i + n, len(new))}/{len(new)}")
    print(f"เสร็จ: {ls.url}/projects/{pid}")


if __name__ == "__main__":
    main()
