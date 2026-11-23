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


def load_env(path=".env"):  # โหลดค่าจากไฟล์ .env
    """อ่าน KEY=VALUE จากไฟล์ .env ใส่ os.environ (ค่าที่ตั้งใน terminal ไว้แล้วมาก่อน)"""
    p = Path(path)  # path ของไฟล์ .env
    if p.exists():  # ไม่มีไฟล์ = ข้าม (อาจตั้งค่าไว้ใน terminal แล้ว)
        for line in p.read_text(encoding="utf-8").splitlines():  # ทีละบรรทัด
            k, sep, v = line.partition("=")  # แยก "KEY=VALUE" เป็น key, "=", value
            if sep and not k.strip().startswith("#"):  # มี "=" และไม่ใช่บรรทัดคอมเมนต์
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))  # ตัดช่องว่างและเครื่องหมายคำพูดรอบค่า


class LabelStudio:  # ตัวเชื่อม Label Studio 1 server
    def __init__(self, url):  # url = ที่อยู่ server เช่น http://host:8080
        load_env()  # อ่าน .env ก่อน
        key = os.environ.get("LABEL_STUDIO_API_KEY", "").strip()  # API key (ไม่มี = "")
        if not key:  # ยังไม่ได้ตั้ง
            raise SystemExit("ยังไม่ได้ใส่ LABEL_STUDIO_API_KEY ในไฟล์ .env (ดูวิธีใน teenoi/labelstudio.py)")  # หยุดพร้อมบอกวิธีแก้
        self.url = url.rstrip("/")  # ตัด / ท้าย url กัน path ซ้อน //
        if key.count(".") == 2:  # Personal Access Token (JWT) ต้องแลกเป็น access token ก่อน
            self.auth = "Bearer " + self.call("POST", "/api/token/refresh", {"refresh": key}, auth=None)["access"]  # ขอ access token แล้วใช้แบบ Bearer
        else:  # Legacy token ใช้ตรง ๆ
            self.auth = f"Token {key}"  # ใช้ key ตรง ๆ
        # ponytail: access token หมดอายุราว 5 นาที ถ้างานนานกว่านั้นจน 401 ค่อยเพิ่มการ refresh ซ้ำ

    def call(self, method, path, body=None, auth=""):  # auth=None = ไม่ส่ง Authorization (ใช้ตอนขอ token)
        """เรียก API คืน JSON (path ขึ้นต้นด้วย /api/...) error -> หยุดพร้อมข้อความจาก server"""
        req = urllib.request.Request(self.url + path, method=method,  # สร้าง request
                                     data=None if body is None else json.dumps(body).encode("utf-8"))  # มี body = ส่งเป็น JSON
        req.add_header("Content-Type", "application/json")  # บอก server ว่า body เป็น JSON
        if auth is not None:  # ต้องยืนยันตัวตน
            req.add_header("Authorization", auth or self.auth)  # auth ที่ส่งมา หรือค่าที่ได้ตอน __init__
        try:  # ลองเรียก
            with urllib.request.urlopen(req, timeout=300) as r:  # export โปรเจกต์ใหญ่อาจใช้เวลานาน
                return json.loads(r.read() or "null")  # แปลงคำตอบเป็น JSON (ว่าง = None)
        except urllib.error.HTTPError as e:  # แสดงข้อความจาก server ให้รู้ว่าผิดตรงไหน
            raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:500]}")  # หยุดพร้อมรหัส error และข้อความ 500 ตัวแรก

    def projects(self):  # รายชื่อโปรเจกต์
        """ทุกโปรเจกต์ (API แบ่งหน้า ไล่ตาม next จนหมด)"""
        out, path = [], "/api/projects?page_size=100"  # ผลรวม / หน้าแรก
        while path:  # ยังมีหน้าถัดไป
            page = self.call("GET", path)  # อ่าน 1 หน้า
            if isinstance(page, list):  # Label Studio เวอร์ชันเก่าไม่แบ่งหน้า
                return page  # ได้ list ตรง ๆ
            out += page["results"]  # สะสมโปรเจกต์ของหน้านี้
            nxt = page.get("next")  # ลิงก์เต็มของหน้าถัดไป (None = หน้าสุดท้าย)
            path = urlsplit(nxt)._replace(scheme="", netloc="").geturl() if nxt else None  # เหลือเฉพาะ path+query เพราะ call() ต่อ self.url ให้เอง
        return out  # ทุกโปรเจกต์

    def export(self, pid, all_tasks=False):  # ดาวน์โหลด task ของ 1 โปรเจกต์
        """task ของโปรเจกต์เป็น JSON (all_tasks=False = เฉพาะที่ label แล้ว)"""
        extra = "&download_all_tasks=true" if all_tasks else ""  # พารามิเตอร์เพิ่มเมื่อขอทุก task
        return self.call("GET", f"/api/projects/{pid}/export?exportType=JSON{extra}")  # รายการ task (แต่ละ task มี data.review_id, annotations)
