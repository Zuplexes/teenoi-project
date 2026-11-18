"""upload รีวิวที่ clean แล้วขึ้น Label Studio: python upload_labelstudio.py (ตั้งค่าใน configs/labelstudio.yml)"""
import csv  # อ่าน CSV
import json  # body/response ของ API
import os  # อ่าน API key จาก env
import urllib.error  # จับ HTTPError
import urllib.request  # เรียก API (stdlib ไม่ต้องลง requests)
from pathlib import Path  # จัดการ path

import yaml  # อ่าน config

ROOT = Path(__file__).resolve().parent  # path ใน config เป็น relative จาก root ของ repo
CONFIG = ROOT / "configs/labelstudio.yml"


class LabelStudio:
    def __init__(self, url, key):
        self.url = url.rstrip("/")
        if key.count(".") == 2:  # Personal Access Token (JWT) ต้องแลกเป็น access token ก่อน
            access = self._call("POST", "/api/token/refresh", {"refresh": key}, auth=None)["access"]
            self.auth = f"Bearer {access}"
        else:  # Legacy token
            self.auth = f"Token {key}"
        # ponytail: access token หมดอายุราว 5 นาที ถ้า upload นานกว่านั้นจน 401 ค่อยเพิ่มการ refresh ซ้ำ

    def _call(self, method, path, body=None, auth=""):
        req = urllib.request.Request(self.url + path, method=method,
                                     data=None if body is None else json.dumps(body).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        if auth is not None:
            req.add_header("Authorization", auth or self.auth)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or "null")
        except urllib.error.HTTPError as e:  # แสดงข้อความจาก server ให้รู้ว่าผิดตรงไหน
            raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:500]}")

    def get_or_create_project(self, title, label_config_path):
        res = self._call("GET", "/api/projects?page_size=1000")
        for p in res.get("results", res) if isinstance(res, dict) else res:  # เวอร์ชันใหม่มี pagination
            if p["title"] == title:
                return p["id"]
        xml = (ROOT / label_config_path).read_text(encoding="utf-8")
        pid = self._call("POST", "/api/projects", {"title": title, "label_config": xml})["id"]
        print(f"สร้างโปรเจกต์ใหม่ '{title}' (id={pid})")
        return pid

    def existing_review_ids(self, pid):
        """review_id ที่อยู่ในโปรเจกต์แล้ว กัน upload ซ้ำเมื่อรันหลายรอบ"""
        tasks = self._call("GET", f"/api/projects/{pid}/export?exportType=JSON&download_all_tasks=true")
        return {str(t["data"].get("review_id")) for t in tasks}

    def import_tasks(self, pid, rows):
        return self._call("POST", f"/api/projects/{pid}/import", rows)


def read_rows(paths):
    rows = {}  # review_id -> row (ตัดซ้ำข้ามไฟล์)
    for p in paths:
        with open(ROOT / p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows[str(r["review_id"])] = {"review_id": r["review_id"], "review": r["review"]}
    return rows


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    env = ROOT / ".env"  # ไฟล์เก็บ key (อยู่ใน .gitignore)
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not k.strip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))  # env ที่ตั้งไว้แล้วมาก่อน .env
    key = os.environ.get("LABEL_STUDIO_API_KEY", "").strip()
    if not key:
        raise SystemExit("ยังไม่ได้ใส่ LABEL_STUDIO_API_KEY ในไฟล์ .env (ดูวิธีใน configs/labelstudio.yml)")

    ls = LabelStudio(cfg["url"], key)
    pid = ls.get_or_create_project(cfg["project_title"], cfg["label_config"])
    rows = read_rows(cfg["files"])
    done = ls.existing_review_ids(pid)
    new = [r for rid, r in rows.items() if rid not in done]
    print(f"รีวิวในไฟล์ {len(rows)} | มีในโปรเจกต์แล้ว {len(rows) - len(new)} | จะ upload {len(new)}")

    n = cfg.get("batch_size", 500)
    for i in range(0, len(new), n):
        ls.import_tasks(pid, new[i:i + n])
        print(f"  upload แล้ว {min(i + n, len(new))}/{len(new)}")
    print(f"เสร็จ: {cfg['url'].rstrip('/')}/projects/{pid}")


if __name__ == "__main__":
    main()
