"""เรียก Label Studio API (ใช้ร่วมกันทั้ง 2_upload_labelstudio.py และ 3_fetch_labels.py)

API key ไม่ใส่ใน config (ขึ้น git) ให้ใส่ในไฟล์ .env ที่ root ของ repo (อยู่ใน .gitignore):
    LABEL_STUDIO_API_KEY=<Label Studio -> Account & Settings -> Personal Access Token หรือ Access Token>
"""
import json  # body/response ของ API
import os  # อ่าน API key จาก env
import urllib.error  # จับ HTTPError
import urllib.request  # เรียก API (stdlib ไม่ต้องลง requests)
from pathlib import Path  # อ่านไฟล์ .env
from urllib.parse import urlsplit  # ตัด host ออกจากลิงก์หน้าถัดไป


def load_env(path=".env"):
    """อ่าน KEY=VALUE จากไฟล์ .env ใส่ os.environ (ค่าที่ตั้งใน terminal ไว้แล้วมาก่อน)"""
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not k.strip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


class LabelStudio:
    def __init__(self, url):
        load_env()
        key = os.environ.get("LABEL_STUDIO_API_KEY", "").strip()
        if not key:
            raise SystemExit("ยังไม่ได้ใส่ LABEL_STUDIO_API_KEY ในไฟล์ .env (ดูวิธีใน teenoi/labelstudio.py)")
        self.url = url.rstrip("/")
        if key.count(".") == 2:  # Personal Access Token (JWT) ต้องแลกเป็น access token ก่อน
            self.auth = "Bearer " + self.call("POST", "/api/token/refresh", {"refresh": key}, auth=None)["access"]
        else:  # Legacy token ใช้ตรง ๆ
            self.auth = f"Token {key}"
        # ponytail: access token หมดอายุราว 5 นาที ถ้างานนานกว่านั้นจน 401 ค่อยเพิ่มการ refresh ซ้ำ

    def call(self, method, path, body=None, auth=""):
        """เรียก API คืน JSON (path ขึ้นต้นด้วย /api/...) error -> หยุดพร้อมข้อความจาก server"""
        req = urllib.request.Request(self.url + path, method=method,
                                     data=None if body is None else json.dumps(body).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        if auth is not None:
            req.add_header("Authorization", auth or self.auth)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:  # export โปรเจกต์ใหญ่อาจใช้เวลานาน
                return json.loads(r.read() or "null")
        except urllib.error.HTTPError as e:  # แสดงข้อความจาก server ให้รู้ว่าผิดตรงไหน
            raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:500]}")

    def projects(self):
        """ทุกโปรเจกต์ (API แบ่งหน้า ไล่ตาม next จนหมด)"""
        out, path = [], "/api/projects?page_size=100"
        while path:
            page = self.call("GET", path)
            if isinstance(page, list):  # Label Studio เวอร์ชันเก่าไม่แบ่งหน้า
                return page
            out += page["results"]
            nxt = page.get("next")  # ลิงก์เต็มของหน้าถัดไป (None = หน้าสุดท้าย)
            path = urlsplit(nxt)._replace(scheme="", netloc="").geturl() if nxt else None
        return out

    def export(self, pid, all_tasks=False):
        """task ของโปรเจกต์เป็น JSON (all_tasks=False = เฉพาะที่ label แล้ว)"""
        extra = "&download_all_tasks=true" if all_tasks else ""
        return self.call("GET", f"/api/projects/{pid}/export?exportType=JSON{extra}")
