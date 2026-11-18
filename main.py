"""teenoi-predict: หาสาขา -> ดึงรีวิว -> ทำนาย -> สรุป -> export (เปิด/ปิดแต่ละขั้นที่ steps: ใน config)"""
import logging  # ระบบ log มาตรฐานของ Python
import os  # ใช้เปลี่ยนโฟลเดอร์ทำงาน
import sys  # ใช้เพิ่ม path ให้ import โมดูลใน src ได้
from collections import Counter  # นับจำนวนแต่ละค่าในรายการ
from pathlib import Path  # จัดการ path ไฟล์แบบข้ามระบบปฏิบัติการ

ROOT = Path(__file__).resolve().parent  # โฟลเดอร์ของโปรเจกต์ (ที่ไฟล์นี้อยู่)
os.chdir(ROOT)  # ให้ path สัมพัทธ์ใน config อ้างจากโฟลเดอร์โปรเจกต์เสมอ ไม่ว่าจะรันจากไหน
sys.path.insert(0, str(ROOT / "src"))  # ให้ import teenoi_predict จากโฟลเดอร์ src ได้

from teenoi_data.config import setup_logging  # noqa: E402  ตั้งค่า log ให้เขียนลงไฟล์ + หน้าจอ

from teenoi_predict import collector, config, db, exporter, predictor, summarizer  # noqa: E402  โมดูลของแต่ละขั้น

log = logging.getLogger("teenoi_predict")  # logger หลักของโปรเจกต์
VERDICTS = ["ดี", "เฉยๆ", "แย่", "ข้อมูลไม่พอ"]  # ลำดับผลสรุปที่จะแสดงบนหน้าจอ


def main():
    cfg = config.load()  # โหลด + ตรวจ config
    setup_logging(cfg.get("log_level", "INFO"), "logs", "predict")  # log ลงโฟลเดอร์ logs ชื่อไฟล์ขึ้นต้นด้วย predict
    conn = db.connect(cfg["output"]["db_file"])  # เปิดฐานข้อมูล (สร้างตารางถ้ายังไม่มี)
    run_id = db.now()  # ใช้เวลาเริ่มเป็นรหัสของการรันรอบนี้
    enabled = cfg["steps"]  # ขั้นไหนเปิด/ปิด
    results = {}  # เก็บผลลัพธ์ของแต่ละขั้น

    def run(name, fn):
        """รันขั้น name ถ้าเปิดไว้ใน config พร้อมบันทึกประวัติลงตาราง runs แล้วคืนผลลัพธ์ (ปิดไว้ = None)"""
        if enabled.get(name):  # ทำเฉพาะขั้นที่ตั้งเป็น true
            log.info("===== %s =====", name)  # แสดงหัวข้อขั้นใน log
            with db.run_step(conn, run_id, name):  # บันทึกเวลาเริ่ม/จบ/error ของขั้นนี้
                results[name] = fn()  # เรียกฟังก์ชันของขั้นแล้วเก็บผล
        return results.get(name)  # คืนผล (None ถ้าข้าม)

    run("find_branches", lambda: collector.find_branches(cfg, conn))  # 1. หาสาขา
    run("scrape", lambda: collector.scrape(cfg, conn))  # 2. ดึงรีวิว
    run("predict", lambda: predictor.predict(cfg, conn))  # 3. ทำนาย
    summary = run("summarize", lambda: summarizer.summarize(cfg, conn))  # 4. สรุปผล
    run("export", lambda: exporter.export(cfg, conn, summary or summarizer.summarize(cfg, conn)))  # 5. เขียนไฟล์ (ถ้าข้ามขั้นสรุป ให้สรุปตอนนี้)

    print_report(cfg, conn, results, summary)  # แสดงสรุปให้ผู้ใช้ตรวจ


def print_report(cfg, conn, results, summary):
    """พิมพ์สรุปผลการรันรอบนี้ออกหน้าจอ"""
    n_branches = conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0]  # จำนวนสาขาทั้งหมดในฐานข้อมูล
    print("\n===== สรุป =====")  # หัวข้อ
    print(f"สาขาในฐานข้อมูล: {n_branches}")  # จำนวนสาขา
    print(f"รีวิวใหม่รอบนี้: {(results.get('scrape') or {}).get('new_reviews', 0)}")  # รีวิวใหม่ (0 ถ้าข้ามขั้น scrape)
    print(f"คู่ที่ทำนายรอบนี้: {(results.get('predict') or {}).get('pairs', 0)}")  # จำนวนคู่ที่ทำนาย (0 ถ้าข้าม)
    if summary:  # มีผลสรุป
        print(f"สาขาที่สรุปผล: {len(summary['branches'])}")  # จำนวนสาขาที่สรุป
        for code, name in summary["aspect_names"].items():  # วนทีละด้าน (เช่น อาหาร บริการ)
            count = Counter(b["aspects"][code]["label"] for b in summary["branches"])  # นับว่ากี่สาขาได้ผลแต่ละแบบ
            print(f"  {name:<12} " + "  ".join(f"{v} {count[v]}" for v in VERDICTS))  # แสดงหนึ่งบรรทัดต่อด้าน
    if cfg["steps"].get("export"):  # ถ้าเขียนไฟล์
        print(f"ไฟล์ผล: {(ROOT / cfg['output']['branch_summary_file']).resolve()}")  # บอกตำแหน่งไฟล์ผลลัพธ์


if __name__ == "__main__":  # รันตรงด้วย python main.py
    main()  # เริ่มทำงาน
