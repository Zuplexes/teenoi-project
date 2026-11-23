"""หาสาขา + ดึงรีวิว ผ่าน teenoi_data (ห้ามเขียน scraper/cleaner ใหม่ที่นี่)"""
import logging  # log ความคืบหน้า
from collections import Counter  # นับจำนวนรีวิวแยกตามดาว
from datetime import datetime, timedelta  # คำนวณช่วงเวลา

from teenoi_data import branch_finder, scraper  # ตัวค้นหาสาขา / ตัวดึงรีวิว (ตัวเดียวกับตอนเตรียมข้อมูลเทรน)
from teenoi_data.cleaner import DEFAULT_CLEANING, clean_text, detect_language  # ตัว clean ข้อความ (ตัวเดียวกับตอนเทรน)

from . import db, exporter  # ฐานข้อมูล / เขียนไฟล์ branches.csv

log = logging.getLogger(__name__)


def _add_extra_urls(cfg, conn):
    """เพิ่มลิงก์ใน branches.extra_urls เป็นสาขาในฐานข้อมูล (ข้ามลิงก์ที่เคยดึงแล้ว)"""
    known = {r[0] for r in conn.execute("SELECT url FROM url_aliases")}  # ลิงก์ที่รู้ branch_id จริงแล้ว
    for url in cfg["branches"].get("extra_urls") or []:
        if url in known:  # เคยดึงแล้ว รู้ branch_id จริง ไม่ต้องใส่แถวชั่วคราวซ้ำ (ไม่งั้นสาขาเดียวถูกดึง 2 รอบ)
            continue
        place = scraper.parse_place_url(url)  # แยกข้อมูลจากลิงก์ (branch_id, ชื่อ, พิกัด ถ้ามี)
        # ลิงก์สั้น/ลิงก์ค้นหา ยังไม่รู้ branch_id ใช้ url เป็น id ชั่วคราวจนกว่าจะดึงครั้งแรก
        db.upsert_branch(conn, {**place, "branch_id": place["branch_id"] or url, "gmaps_url": url}, found_at=db.now())
    conn.commit()


def find_branches(cfg, conn):
    """ค้นหาสาขาบน Google Maps ด้วย branches.search_queries (ข้ามถ้าเพิ่งค้นภายใน refresh_days วัน)"""
    branches_cfg = cfg["branches"]
    refresh_days = branches_cfg.get("refresh_days", 30)  # ค้นใหม่ทุกกี่วัน
    _add_extra_urls(cfg, conn)  # ลิงก์ที่ใส่เองเพิ่มเสมอ
    last = db.last_success(conn, "find_branches")  # ค้นสำเร็จล่าสุดเมื่อไร
    if last and datetime.fromisoformat(last) > datetime.now() - timedelta(days=refresh_days):  # ยังไม่ถึงรอบ
        log.info("ข้ามการค้นหาสาขา: ค้นไปแล้วเมื่อ %s (refresh_days=%s)", last, refresh_days)
        return {"found": 0}
    df = branch_finder.find_branches({"browser": cfg["browser"], "branch_finder": {  # แปลง config ให้อยู่ในรูปแบบของ teenoi_data
        "search_queries": branches_cfg["search_queries"], "name_must_contain": branches_cfg.get("name_must_contain")}})
    for row in df.to_dict("records"):  # บันทึกทีละสาขา
        db.upsert_branch(conn, row, found_at=db.now())
    conn.commit()
    exporter.export_branches(cfg, conn)  # เขียน branches.csv ให้ตรวจรายชื่อ
    log.info("ค้นเจอ %d สาขา ตรวจรายชื่อได้ที่ %s", len(df), cfg["output"]["branches_file"])
    return {"found": len(df)}


def _clean(text, cleaning):
    """clean แล้วคืน None ถ้าสั้นเกิน/ภาษาไม่ตรง (เกณฑ์เดียวกับ clean_dataframe) รีวิวนั้นจะไม่ถูกทำนาย"""
    t = clean_text(text, cleaning)  # ลบอีโมจิ/ตัวอักษรซ้ำ ฯลฯ ตามตั้งค่า
    if len(t) < cleaning["min_length"]:  # สั้นเกิน
        return None
    if cleaning["languages"] and detect_language(t) not in cleaning["languages"]:  # ภาษาไม่อยู่ในที่กำหนด
        return None
    return t


def _branches_to_scrape(cfg, conn):
    """สาขาที่ต้องดึงรอบนี้: ยังไม่เคยดึงก่อน แล้วตามด้วยที่ดึงนานที่สุด"""
    scraping = cfg["scraping"]
    cutoff = (datetime.now() - timedelta(days=scraping.get("skip_if_scraped_within_days", 7))).isoformat()  # ดึงหลังเวลานี้ = ข้าม
    excluded = set(cfg["branches"].get("exclude_branch_ids") or [])  # สาขาที่ไม่เอา
    todo = [r for r in conn.execute(
        "SELECT branch_id, gmaps_url, last_scraped_at FROM branches "
        "WHERE last_scraped_at IS NULL OR last_scraped_at < ? "  # ยังไม่เคยดึง หรือดึงนานแล้ว
        "ORDER BY last_scraped_at IS NOT NULL, last_scraped_at", (cutoff,))  # ยังไม่เคยดึงก่อน แล้วเก่าสุดก่อน
        if r["branch_id"] not in excluded]
    return todo[:scraping.get("max_branches_per_run", 20)]  # จำกัดจำนวนต่อรอบ (None = ทั้งหมด)


def scrape(cfg, conn):
    """ดึงรีวิวทุกสาขาที่ถึงรอบ บันทึกลงฐานข้อมูลทีละสาขา คืนจำนวนสาขาและรีวิวใหม่"""
    scraping = cfg["scraping"]
    cleaning = {**DEFAULT_CLEANING, **(cfg.get("cleaning") or {})}  # ค่าเริ่มต้น ทับด้วยค่าใน config
    _add_extra_urls(cfg, conn)  # ใส่ลิงก์จาก config ก่อน
    todo = _branches_to_scrape(cfg, conn)
    if not todo:  # ไม่มีงาน
        log.info("ไม่มีสาขาที่ต้องดึงรอบนี้")
        return {"branches": 0, "new_reviews": 0}

    known = {}  # branch_id -> รหัสรีวิวที่มีอยู่แล้ว (scraper หยุดเลื่อนเมื่อเจอ)
    if scraping.get("incremental", True):  # โหมดดึงเฉพาะรีวิวใหม่
        for r in todo:
            if r["last_scraped_at"]:  # เคยดึงแล้วเท่านั้นถึงมีรีวิวเดิม
                known[r["branch_id"]] = {x[0] for x in conn.execute(
                    "SELECT gmaps_review_id FROM reviews WHERE branch_id = ?", (r["branch_id"],))}
    log.info("ดึงรีวิว %d สาขา (สาขาใหม่ %d)", len(todo), sum(not r["last_scraped_at"] for r in todo))

    months = scraping.get("stop_older_than_months")  # ดึงย้อนหลังกี่เดือน (None = ไม่จำกัด)
    stop_before = (datetime.now() - timedelta(days=30 * months)).date().isoformat() if months else None  # วันที่ที่หยุดเลื่อน
    stats = {"branches": 0, "new_reviews": 0}  # ตัวนับผล

    def save(info, reviews):
        """scraper เรียกหลังจบแต่ละสาขา ข้อมูลไม่หายถ้าโปรแกรมหยุดกลางทาง"""
        branch_id = info["branch_id"]  # branch_id จริงจากหน้าสาขา
        db.upsert_branch(conn, info, found_at=db.now(), last_scraped_at=db.now())  # อัปเดตข้อมูลสาขา + เวลาดึง
        # ponytail: รู้ลิงก์ต้นทางจาก source_url ของรีวิวเท่านั้น ถ้าลิงก์สั้นได้ 0 รีวิว แถวชั่วคราวจะถูกดึงซ้ำรอบหน้า
        source_url = reviews[0]["source_url"] if reviews else None  # ลิงก์ที่ใช้เปิดสาขานี้
        if source_url and source_url != branch_id:  # เปิดจากลิงก์ชั่วคราว
            conn.execute("DELETE FROM branches WHERE branch_id = ?", (source_url,))  # ลบแถวชั่วคราว
            conn.execute("INSERT OR REPLACE INTO url_aliases VALUES (?, ?)", (source_url, branch_id))  # จำว่าลิงก์นี้คือสาขาไหน
        cur = conn.executemany("INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [  # รีวิวที่มีอยู่แล้ว = ข้าม
            (r["gmaps_review_id"], branch_id, r["rating"], r["date_text"], r["date_approx"],
             r["review"], _clean(r["review"], cleaning), r["scraped_at"]) for r in reviews])  # เก็บทั้งข้อความดิบและที่ clean แล้ว
        conn.commit()
        # นับเฉพาะรีวิวในช่วงเวลา (ชุดสุดท้ายที่โหลดมามักมีรีวิวเก่ากว่าช่วงติดมาด้วย) ไม่รู้วันที่ = นับ
        in_range = [r for r in reviews if not stop_before or (r["date_approx"] or stop_before) >= stop_before]
        stars = Counter(r["rating"] for r in in_range)  # จำนวนรีวิวแยกตามดาว
        log.info("    ใน %s เดือน: %d รีวิว (มีข้อความ %d) | 5★ %d  4★ %d  3★ %d  2★ %d  1★ %d",
                 months or "ทุก", len(in_range), sum(bool(r["review"].strip()) for r in in_range),
                 *(stars[k] for k in range(5, 0, -1)))  # ดาว 5 ลงไป 1
        stats["branches"] += 1  # นับสาขา
        stats["new_reviews"] += cur.rowcount  # นับเฉพาะรีวิวที่เพิ่มจริง

    df = scraper.scrape_many(
        {"browser": cfg["browser"], "scraper": {  # แปลง config ให้อยู่ในรูปแบบของ teenoi_data
            "max_reviews_per_place": scraping["max_reviews_first_run"], "sort_by": "newest",  # เรียงใหม่สุดก่อน
            "delay": scraping.get("delay", {}), "stop_before": stop_before}},
        [r["gmaps_url"] for r in todo], known, save)  # ลิงก์ทุกสาขา, รีวิวที่มีแล้ว, ฟังก์ชันบันทึก
    if df.attrs.get("limited_view"):  # Google ไม่ให้ดูรีวิวเพราะยังไม่ล็อกอิน
        raise SystemExit("Google Maps เปิดแบบ \"มุมมองแบบจำกัด\" (ยังไม่ได้ล็อกอิน) ให้รัน python login.py แล้วรันใหม่ "
                         f"(บันทึกไปแล้ว {stats['branches']} สาขา)")
    log.info("ดึงเสร็จ %d สาขา ได้รีวิวใหม่ %d รีวิว", stats["branches"], stats["new_reviews"])
    return stats
