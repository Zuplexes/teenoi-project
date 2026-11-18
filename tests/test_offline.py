"""ทดสอบส่วนที่ไม่ต้องใช้ browser/โมเดล: python tests/test_offline.py"""
import sys  # ใช้เพิ่ม path
from pathlib import Path  # จัดการ path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # ให้ import teenoi_predict จาก src ได้

from teenoi_data.cleaner import DEFAULT_CLEANING  # noqa: E402  ค่า clean มาตรฐาน

from teenoi_predict import collector, db, exporter, summarizer  # noqa: E402  โมดูลที่ทดสอบ

SETTINGS = {"min_mentions": 2, "positive_threshold": 0.3, "negative_threshold": -0.3, "sample_reviews_per_label": 1}  # เกณฑ์สรุปสำหรับทดสอบ


def test_db_upsert_keeps_first_found_at_and_ignores_none():
    conn = db.connect(":memory:")  # ฐานข้อมูลชั่วคราวในหน่วยความจำ
    db.upsert_branch(conn, {"branch_id": "A", "name": "x", "lat": 13.0}, found_at="2026-01-01")  # เพิ่มครั้งแรก
    db.upsert_branch(conn, {"branch_id": "A", "name": "y", "lat": None}, found_at="2026-02-01", last_scraped_at="t")  # อัปเดต
    r = conn.execute("SELECT * FROM branches").fetchone()
    assert (r["name"], r["lat"], r["found_at"], r["last_scraped_at"]) == ("y", 13.0, "2026-01-01", "t")  # ชื่อใหม่, lat เดิม, found_at แรก


def test_clean_drops_short_and_foreign():
    c = dict(DEFAULT_CLEANING)
    assert collector._clean("อร่อยมากกกกก", c) == "อร่อยมากก"  # ตัวอักษรซ้ำถูกย่อ
    assert collector._clean("ok", c) is None  # สั้นเกิน
    assert collector._clean("美味しいです", c) is None  # ภาษาอื่น


def test_summary_labels_and_samples():
    branches = [{"branch_id": "A", "name": "a", "address": None, "lat": 1, "lng": 2, "gmaps_url": "u",
                 "google_rating": 4.5, "last_scraped_at": "2026-11-18T10:00:00"}]  # สาขาเดียว
    rows = [{"branch_id": "A", "gmaps_review_id": str(i), "date_approx": "2026-11-01", "review_clean": f"r{i}",
             "aspect": a, "label": lab, "confidence": conf}
            for i, a, lab, conf in [(1, "food", "positive", .9), (2, "food", "positive", .8),
                                    (3, "food", "negative", .95), (4, "food", "negative", .5),  # ต่ำกว่า threshold
                                    (1, "service", "negative", .9), (2, "service", "none", .99)]]  # none = ไม่นับ
    b = summarizer.build_branches(branches, rows, ["food", "service"], SETTINGS, 0.6)[0]
    assert b["reviews_analyzed"] == 4  # นับทุกรีวิวที่ทำนาย
    assert b["aspects"]["food"] == {"label": "ดี", "score": 0.333, "positive": 2, "neutral": 0, "negative": 1}  # (2-1)/3
    assert b["aspects"]["service"]["label"] == "ข้อมูลไม่พอ" and b["aspects"]["service"]["score"] is None  # นับได้แค่ 1
    assert b["sample_reviews"]["food"]["positive"] == ["r1"]  # มั่นใจสุดก่อน
    assert b["last_scraped_at"] == "2026-11-18"  # ตัดเหลือวันที่


def test_export_review_predictions_wide(tmp=Path(__file__).parent / "_tmp.csv"):
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO branches (branch_id, name) VALUES ('A', 'a')")
    conn.execute("INSERT INTO reviews (gmaps_review_id, branch_id, rating, review_clean) VALUES ('1', 'A', 5, 'ดี')")
    conn.executemany("INSERT INTO predictions VALUES ('1', ?, ?, 0.9, 'v', 't')", [("food", "positive"), ("service", "none")])
    exporter.export_review_predictions({"output": {"review_predictions_file": str(tmp)}}, conn, "v", ["food", "service"])
    text = tmp.read_text(encoding="utf-8-sig").splitlines()
    tmp.unlink()  # ลบไฟล์ทดสอบ
    assert text[0] == "branch_id,branch,gmaps_review_id,date_approx,review_clean,rating,food,service,food_confidence,service_confidence"
    assert text[1] == "A,a,1,,ดี,5,positive,none,0.9,0.9"  # 1 รีวิว = 1 แถว


def test_search_url_not_rescraped_after_resolved():
    """ลิงก์ค้นหาที่ดึงแล้วต้องไม่กลับมาเป็นสาขาใหม่ในรอบถัดไป (เคยทำให้ 55 สาขากลายเป็น 119)"""
    import pandas as pd
    url = "https://www.google.com/maps/search/?api=1&query=x"
    calls = []  # จด url ที่ scraper ถูกเรียกแต่ละรอบ

    def fake(cfg, urls, known, save):  # scraper ปลอม: ลิงก์ค้นหาเปิดได้สาขา 0x1:0x2
        calls.append(list(urls))
        save({"branch_id": "0x1:0x2", "name": "a"}, [dict(gmaps_review_id="r1", rating=5, date_text="", date_approx=None,
                                                          review="อร่อย", scraped_at="t", source_url=url)])
        return pd.DataFrame()
    real, collector.scraper.scrape_many = collector.scraper.scrape_many, fake  # สลับใส่ตัวปลอม
    try:
        conn = db.connect(":memory:")
        cfg = {"scraping": {"max_reviews_first_run": 1, "skip_if_scraped_within_days": 0},
               "branches": {"extra_urls": [url]}, "browser": {}}
        collector.scrape(cfg, conn)  # รอบแรก: ดึงจากลิงก์ค้นหา
        collector.scrape(cfg, conn)  # รอบสอง: ต้องดึงจาก branch_id จริง
    finally:
        collector.scraper.scrape_many = real  # คืนตัวจริง
    assert calls[0] == [url] and len(calls[1]) == 1 and calls[1][0] != url, calls
    assert [r[0] for r in conn.execute("SELECT branch_id FROM branches")] == ["0x1:0x2"]  # ไม่มีแถวชั่วคราวค้าง


if __name__ == "__main__":  # รันทุกฟังก์ชันที่ขึ้นต้นด้วย test_
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
