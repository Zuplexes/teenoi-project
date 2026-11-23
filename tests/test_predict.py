"""ทดสอบส่วนที่ไม่ต้องใช้ browser: python tests/test_predict.py"""
import sys  # ใช้เพิ่ม path
from pathlib import Path  # จัดการ path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ให้ import teenoi จาก root ของ repo ได้

from teenoi.collect.cleaner import DEFAULT_CLEANING  # noqa: E402  ค่า clean มาตรฐาน

from teenoi.predict import collector, db, exporter, predictor, summarizer  # noqa: E402  โมดูลที่ทดสอบ

SETTINGS = {"min_mentions": 2, "positive_threshold": 0.3, "negative_threshold": -0.3, "sample_reviews_per_label": 1}  # เกณฑ์สรุปสำหรับทดสอบ


def test_db_upsert_keeps_first_found_at_and_ignores_none():  # upsert: found_at แรกคงอยู่, ค่า None ไม่ทับของเดิม
    conn = db.connect(":memory:")  # ฐานข้อมูลชั่วคราวในหน่วยความจำ
    db.upsert_branch(conn, {"branch_id": "A", "name": "x", "lat": 13.0}, found_at="2026-01-01")  # เพิ่มครั้งแรก
    db.upsert_branch(conn, {"branch_id": "A", "name": "y", "lat": None}, found_at="2026-02-01", last_scraped_at="t")  # อัปเดต
    r = conn.execute("SELECT * FROM branches").fetchone()  # อ่านแถวที่บันทึก
    assert (r["name"], r["lat"], r["found_at"], r["last_scraped_at"]) == ("y", 13.0, "2026-01-01", "t")  # ชื่อใหม่, lat เดิม, found_at แรก


def test_clean_drops_short_and_foreign():  # clean: ตัดข้อความสั้น/ภาษาอื่น
    c = dict(DEFAULT_CLEANING)  # ค่า clean มาตรฐาน
    assert collector._clean("อร่อยมากกกกก", c) == "อร่อยมากก"  # ตัวอักษรซ้ำถูกย่อ
    assert collector._clean("ok", c) is None  # สั้นเกิน
    assert collector._clean("美味しいです", c) is None  # ภาษาอื่น


def test_summary_labels_and_samples():  # สรุปรายสาขา: คำตัดสิน + ตัวอย่างรีวิว
    branches = [{"branch_id": "A", "name": "a", "address": None, "lat": 1, "lng": 2, "gmaps_url": "u",  # สาขาทดสอบ 1 สาขา:
                 "google_rating": 4.5, "last_scraped_at": "2026-11-18T10:00:00"}]  # สาขาเดียว
    rows = [{"branch_id": "A", "gmaps_review_id": str(i), "date_approx": "2026-11-01", "review_clean": f"r{i}",  # ผลทำนายปลอม:
             "aspect": a, "label": lab, "confidence": conf}  # ด้าน, ผล, ความมั่นใจ
            for i, a, lab, conf in [(1, "food", "positive", .9), (2, "food", "positive", .8),  # รายการ (รีวิว, ด้าน, ผล, ความมั่นใจ):
                                    (3, "food", "negative", .95), (4, "food", "negative", .5),  # ต่ำกว่า threshold
                                    (1, "service", "negative", .9), (2, "service", "none", .99)]]  # none = ไม่นับ
    b = summarizer.build_branches(branches, rows, ["food", "service"], SETTINGS, 0.6)[0]  # สรุปด้วยเกณฑ์ความมั่นใจ 0.6
    assert b["reviews_analyzed"] == 4  # นับทุกรีวิวที่ทำนาย
    assert b["aspects"]["food"] == {"label": "ดี", "score": 0.333, "positive": 2, "neutral": 0, "negative": 1}  # (2-1)/3
    assert b["aspects"]["service"]["label"] == "ข้อมูลไม่พอ" and b["aspects"]["service"]["score"] is None  # นับได้แค่ 1
    assert b["sample_reviews"]["food"]["positive"] == ["r1"]  # มั่นใจสุดก่อน
    assert b["last_scraped_at"] == "2026-11-18"  # ตัดเหลือวันที่


def test_export_review_predictions_wide(tmp=Path(__file__).parent / "_tmp.csv"):  # export: 1 รีวิว = 1 แถว ผลแต่ละด้านเป็นคอลัมน์
    conn = db.connect(":memory:")  # ฐานข้อมูลชั่วคราว
    conn.execute("INSERT INTO branches (branch_id, name) VALUES ('A', 'a')")  # สาขาทดสอบ
    conn.execute("INSERT INTO reviews (gmaps_review_id, branch_id, rating, review_clean) VALUES ('1', 'A', 5, 'ดี')")  # รีวิวทดสอบ
    conn.executemany("INSERT INTO predictions VALUES ('1', ?, ?, 0.9, 'v', 't')", [("food", "positive"), ("service", "none")])  # ผลทำนาย 2 ด้าน
    exporter.export_review_predictions({"output": {"review_predictions_file": str(tmp)}}, conn, "v", ["food", "service"])  # เขียนไฟล์ทดสอบ
    text = tmp.read_text(encoding="utf-8-sig").splitlines()  # อ่านกลับมาตรวจ
    tmp.unlink()  # ลบไฟล์ทดสอบ
    assert text[0] == "branch_id,branch,gmaps_review_id,date_approx,review_clean,rating,food,service,food_confidence,service_confidence"  # หัวคอลัมน์ถูกต้อง
    assert text[1] == "A,a,1,,ดี,5,positive,none,0.9,0.9"  # 1 รีวิว = 1 แถว


def test_search_url_not_rescraped_after_resolved():  # ลิงก์ค้นหาต้องไม่ถูกดึงซ้ำเป็นสาขาใหม่
    """ลิงก์ค้นหาที่ดึงแล้วต้องไม่กลับมาเป็นสาขาใหม่ในรอบถัดไป (เคยทำให้ 55 สาขากลายเป็น 119)"""
    import pandas as pd  # สร้างตารางผลปลอม
    url = "https://www.google.com/maps/search/?api=1&query=x"  # ลิงก์ค้นหาทดสอบ
    calls = []  # จด url ที่ scraper ถูกเรียกแต่ละรอบ

    def fake(cfg, urls, known, save):  # scraper ปลอม: ลิงก์ค้นหาเปิดได้สาขา 0x1:0x2
        calls.append(list(urls))  # จดลิงก์ที่ถูกสั่งดึง
        save({"branch_id": "0x1:0x2", "name": "a"}, [dict(gmaps_review_id="r1", rating=5, date_text="", date_approx=None,  # แกล้งเป็นว่าลิงก์นี้คือสาขา 0x1:0x2 ได้ 1 รีวิว:
                                                          review="อร่อย", scraped_at="t", source_url=url)])  # ข้อความรีวิว + ลิงก์ต้นทาง
        return pd.DataFrame()  # ตารางว่าง (ผู้เรียกไม่ได้ใช้)
    real, collector.scraper.scrape_many = collector.scraper.scrape_many, fake  # สลับใส่ตัวปลอม
    try:  # ลองรัน
        conn = db.connect(":memory:")  # ฐานข้อมูลชั่วคราว
        cfg = {"scraping": {"max_reviews_first_run": 1, "skip_if_scraped_within_days": 0},  # config ขั้นต่ำที่ scrape() ใช้:
               "branches": {"extra_urls": [url]}, "browser": {}}  # ลิงก์ทดสอบ 1 ลิงก์
        collector.scrape(cfg, conn)  # รอบแรก: ดึงจากลิงก์ค้นหา
        collector.scrape(cfg, conn)  # รอบสอง: ต้องดึงจาก branch_id จริง
    finally:  # ทำเสมอ แม้ assert ล้ม
        collector.scraper.scrape_many = real  # คืนตัวจริง
    assert calls[0] == [url] and len(calls[1]) == 1 and calls[1][0] != url, calls  # รอบแรกใช้ลิงก์ค้นหา รอบสองใช้ id จริง
    assert [r[0] for r in conn.execute("SELECT branch_id FROM branches")] == ["0x1:0x2"]  # ไม่มีแถวชั่วคราวค้าง


def test_predictor_uses_saved_model(tmp=Path(__file__).parent / "_tmp_model.joblib"):  # ทำนายด้วยไฟล์โมเดลจริงแบบที่ 5_train.py บันทึก
    """ทำนายด้วยไฟล์โมเดลแบบที่ 5_train.py บันทึก: ข้ามรีวิวที่ clean ไม่ผ่าน และรันซ้ำไม่ทำนายซ้ำ"""
    import joblib  # บันทึกโมเดล
    from sklearn.feature_extraction.text import CountVectorizer  # นับ n-gram
    from sklearn.naive_bayes import MultinomialNB  # Naive Bayes
    from sklearn.pipeline import make_pipeline  # ต่อเป็น pipeline

    import teenoi  # เลขเวอร์ชัน cleaner
    clf = make_pipeline(CountVectorizer(analyzer="char"), MultinomialNB(alpha=0.1)).fit(  # โมเดลเล็ก ๆ เทรนจาก 3 ข้อความ:
        ["อร่อยมาก", "ไม่อร่อยเลย", "ห้องน้ำ"], ["positive", "negative", "none"])  # ข้อความ, label
    joblib.dump({"trained_at": "v1", "aspects": {"food": "อาหาร"}, "models": {"food": clf},  # บันทึกในรูปแบบเดียวกับ 5_train.py:
                 "teenoi_data_version": teenoi.__version__}, tmp)  # เวอร์ชัน cleaner ตรงกัน (ไม่มีคำเตือน)
    conn = db.connect(":memory:")  # ฐานข้อมูลชั่วคราว
    conn.executemany("INSERT INTO reviews (gmaps_review_id, review_clean) VALUES (?, ?)",  # ใส่รีวิว 3 รีวิว:
                     [("1", "อร่อยมาก"), ("2", "ไม่อร่อยเลย"), ("3", None)])  # None = clean ไม่ผ่าน ไม่ทำนาย
    try:  # ลองรัน
        cfg = {"model": {"file": str(tmp)}}  # config ขั้นต่ำที่ predict() ใช้
        assert predictor.predict(cfg, conn) == {"pairs": 2}  # ทำนาย 2 รีวิว x 1 ด้าน
        assert predictor.predict(cfg, conn) == {"pairs": 0}  # รันซ้ำ = ไม่มีงาน
    finally:  # ทำเสมอ แม้ assert ล้ม
        tmp.unlink()  # ลบไฟล์ทดสอบ
    rows = [tuple(r) for r in conn.execute("SELECT gmaps_review_id, aspect, label, model_version FROM predictions ORDER BY 1")]  # ผลที่บันทึก
    assert rows == [("1", "food", "positive", "v1"), ("2", "food", "negative", "v1")], rows  # ทายถูกทั้ง 2 รีวิว


if __name__ == "__main__":  # รันทุกฟังก์ชันที่ขึ้นต้นด้วย test_
    for name, fn in list(globals().items()):  # ทุกชื่อในไฟล์นี้
        if name.startswith("test_"):  # เฉพาะฟังก์ชันทดสอบ
            fn()  # รัน (ล้ม = มี error พร้อมบรรทัดที่ผิด)
            print("ok", name)  # ผ่าน
