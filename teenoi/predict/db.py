"""SQLite: สร้างตาราง, อ่าน/เขียน"""
import sqlite3  # ฐานข้อมูลไฟล์เดียว มากับ Python
from contextlib import contextmanager  # ใช้สร้าง with-block สำหรับบันทึกการรัน
from datetime import datetime  # เวลาปัจจุบัน
from pathlib import Path  # สร้างโฟลเดอร์ของไฟล์ฐานข้อมูล

SCHEMA = """
-- สาขา: 1 แถว = 1 สาขาบน Google Maps
CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,            -- รหัสสาขาจาก Google Maps (ลิงก์ที่ยังไม่เคยดึงใช้ url เป็นรหัสชั่วคราว)
    name TEXT, address TEXT,               -- ชื่อ, ที่อยู่
    lat REAL, lng REAL,                    -- พิกัด
    gmaps_url TEXT,                        -- ลิงก์สาขา
    google_rating REAL, review_count INTEGER,  -- ดาวเฉลี่ยและจำนวนรีวิวที่ Google แสดง
    found_at TEXT,                         -- เวลาที่พบสาขานี้ครั้งแรก
    last_scraped_at TEXT);                 -- เวลาที่ดึงรีวิวล่าสุด (NULL = ยังไม่เคยดึง)
-- รีวิว: 1 แถว = 1 รีวิว
CREATE TABLE IF NOT EXISTS reviews (
    gmaps_review_id TEXT PRIMARY KEY,      -- รหัสรีวิวจาก Google (กันรีวิวซ้ำ)
    branch_id TEXT,                        -- รีวิวของสาขาไหน
    rating INTEGER,                        -- ดาว 1-5
    date_text TEXT, date_approx TEXT,      -- วันที่แบบที่ Google แสดง (เช่น "2 เดือนที่แล้ว") และวันที่ประมาณ YYYY-MM-DD
    review_raw TEXT,                       -- ข้อความดิบ
    review_clean TEXT,                     -- ข้อความหลัง clean (NULL = สั้น/ภาษาไม่ตรง ไม่ทำนาย)
    scraped_at TEXT);                      -- เวลาที่ดึง
CREATE INDEX IF NOT EXISTS idx_reviews_branch ON reviews(branch_id);  -- ค้นรีวิวตามสาขาให้เร็ว
-- ผลทำนาย: 1 แถว = 1 รีวิว x 1 ด้าน
CREATE TABLE IF NOT EXISTS predictions (
    gmaps_review_id TEXT, aspect TEXT,     -- รีวิวไหน ด้านไหน
    label TEXT, confidence REAL,           -- ผล (positive/neutral/negative/none) และความมั่นใจ 0-1
    model_version TEXT,                    -- เวอร์ชันโมเดล (trained_at) ที่ทำนาย
    predicted_at TEXT,                     -- เวลาที่ทำนาย
    PRIMARY KEY (gmaps_review_id, aspect));  -- รีวิวหนึ่งมีผลได้ด้านละ 1 แถว
-- ลิงก์ค้นหา/ลิงก์สั้น -> branch_id จริง (กันสาขาเดียวถูกดึงซ้ำ)
CREATE TABLE IF NOT EXISTS url_aliases (url TEXT PRIMARY KEY, branch_id TEXT);
-- ประวัติการรัน: 1 แถว = 1 ขั้นของ 1 รอบ
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT, started_at TEXT, finished_at TEXT,  -- รอบไหน เริ่ม/จบเมื่อไร
    step TEXT, status TEXT, message TEXT);  -- ขั้นอะไร สถานะ running/ok/error และข้อความ error
"""
BRANCH_COLS = ["name", "address", "lat", "lng", "gmaps_url", "google_rating", "review_count"]  # คอลัมน์ข้อมูลสาขาที่อัปเดตได้


def now():
    """เวลาปัจจุบันแบบ ISO ละเอียดถึงวินาที เช่น 2026-11-18T10:00:00"""
    return datetime.now().isoformat(timespec="seconds")


def connect(path):
    """เปิดฐานข้อมูล (":memory:" = ในหน่วยความจำ ไว้ทดสอบ) และสร้างตารางที่ยังไม่มี"""
    if path != ":memory:":  # เป็นไฟล์จริง
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี
    conn = sqlite3.connect(path)  # เปิด/สร้างไฟล์ฐานข้อมูล
    conn.row_factory = sqlite3.Row  # ให้อ่านแถวด้วยชื่อคอลัมน์ได้ เช่น row["name"]
    conn.executescript(SCHEMA)  # สร้างตาราง (IF NOT EXISTS = ไม่ทับของเดิม)
    return conn


def upsert_branch(conn, branch, **extra):
    """เพิ่มสาขาใหม่ หรืออัปเดตเฉพาะค่าที่ไม่ใช่ None ของสาขาเดิม (found_at เก็บครั้งแรกไว้)
    extra: คอลัมน์เพิ่มเติม เช่น found_at, last_scraped_at"""
    row = {**{col: branch.get(col) for col in BRANCH_COLS}, **extra}  # ค่าที่จะเขียน (ไม่มี = None)
    cols = ["branch_id", *row]  # คอลัมน์ทั้งหมดที่ INSERT
    updates = ", ".join(  # ส่วน SET ตอนสาขามีอยู่แล้ว
        "found_at = COALESCE(found_at, excluded.found_at)" if col == "found_at"  # found_at: เก็บค่าเดิมถ้ามี
        else f"{col} = COALESCE(excluded.{col}, {col})" for col in row)  # อื่นๆ: ใช้ค่าใหม่ ยกเว้นค่าใหม่เป็น None
    conn.execute(
        f"INSERT INTO branches ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "  # เพิ่มแถวใหม่
        f"ON CONFLICT(branch_id) DO UPDATE SET {updates}",  # ถ้า branch_id ซ้ำ ให้อัปเดตแทน
        [branch["branch_id"], *row.values()])  # ค่าที่ใส่แทน ?


def last_success(conn, step):
    """เวลาจบล่าสุดที่ขั้น step รันสำเร็จ (None = ไม่เคย)"""
    return conn.execute("SELECT MAX(finished_at) FROM runs WHERE step = ? AND status = 'ok'", (step,)).fetchone()[0]


@contextmanager
def run_step(conn, run_id, step):
    """บันทึกประวัติการรันแต่ละขั้นลงตาราง runs: running ตอนเริ่ม แล้วเป็น ok หรือ error ตอนจบ"""
    rowid = conn.execute("INSERT INTO runs (run_id, started_at, step, status) VALUES (?, ?, ?, 'running')",
                         (run_id, now(), step)).lastrowid  # เพิ่มแถวสถานะ running แล้วจำ rowid ไว้อัปเดต
    conn.commit()  # บันทึกทันที (เห็นได้แม้โปรแกรมค้าง)
    status, message = "ok", None  # ค่าเริ่มต้น = สำเร็จ
    try:
        yield  # ทำงานของขั้นนั้น (โค้ดใน with-block)
    except BaseException as e:  # รวม Ctrl+C และ SystemExit
        status, message = "error", str(e) or type(e).__name__  # จำ error ไว้ (ไม่มีข้อความ ใช้ชื่อ error)
        raise  # ส่ง error ต่อ ไม่กลืนทิ้ง
    finally:  # ทำเสมอ ไม่ว่าสำเร็จหรือ error
        conn.execute("UPDATE runs SET finished_at = ?, status = ?, message = ? WHERE rowid = ?",
                     (now(), status, message, rowid))  # บันทึกเวลาจบและสถานะ
        conn.commit()
