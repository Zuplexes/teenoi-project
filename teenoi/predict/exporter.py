"""เขียน branch_summary.json / review_predictions.csv / branches.csv"""
import json  # เขียนไฟล์ json
from pathlib import Path  # จัดการ path

import pandas as pd  # อ่านตารางจาก SQL และเขียน csv

CSV_ENCODING = "utf-8-sig"  # -sig ให้ Excel อ่านภาษาไทยได้
REVIEW_COLS = ["branch_id", "branch", "gmaps_review_id", "date_approx", "review_clean", "rating"]  # คอลัมน์ข้อมูลรีวิวใน csv


def _ensure_dir(path):  # สร้างโฟลเดอร์ของไฟล์
    """สร้างโฟลเดอร์ของไฟล์ path ถ้ายังไม่มี แล้วคืน path เดิม"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ (มีแล้วไม่ error)
    return path  # คืน path เดิม ใช้ต่อได้ทันที


def export_branches(cfg, conn):  # เขียน branches.csv
    """เขียนรายชื่อสาขาทั้งหมดเป็น csv เรียงตามชื่อ"""
    df = pd.read_sql_query("SELECT * FROM branches ORDER BY name", conn)  # อ่านทั้งตาราง
    df.to_csv(_ensure_dir(cfg["output"]["branches_file"]), index=False, encoding=CSV_ENCODING)  # เขียนไฟล์


def export_review_predictions(cfg, conn, version, aspects):  # เขียน review_predictions.csv
    """เขียนผลทำนายแบบ 1 แถว = 1 รีวิว มีคอลัมน์ผลและความมั่นใจของแต่ละด้าน"""
    df = pd.read_sql_query(  # ผลทำนายแบบยาว (1 แถว = 1 รีวิว x 1 ด้าน) พร้อมข้อมูลรีวิวและชื่อสาขา
        "SELECT r.branch_id, b.name AS branch, r.gmaps_review_id, r.date_approx, r.review_clean, r.rating, "  # SQL: ข้อมูลรีวิว + ชื่อสาขา
        "p.aspect, p.label, p.confidence FROM predictions p JOIN reviews r USING (gmaps_review_id) "  # + ผลทำนายรายด้าน (ต่อตารางด้วยรหัสรีวิว)
        "LEFT JOIN branches b USING (branch_id) WHERE p.model_version = ?", conn, params=(version,))  # เฉพาะโมเดลเวอร์ชันนี้
    cols = REVIEW_COLS + aspects + [f"{a}_confidence" for a in aspects]  # ลำดับคอลัมน์ของไฟล์ผล
    if df.empty:  # ยังไม่มีผลทำนาย
        out = pd.DataFrame(columns=cols)  # เขียนแค่หัวตาราง
    else:  # มีผลทำนาย
        labels = df.pivot(index="gmaps_review_id", columns="aspect", values="label")  # ผลแต่ละด้านเป็นคอลัมน์
        conf = df.pivot(index="gmaps_review_id", columns="aspect", values="confidence").add_suffix("_confidence")  # ความมั่นใจแต่ละด้าน
        info_cols = [c for c in REVIEW_COLS if c != "gmaps_review_id"]  # ข้อมูลรีวิว (gmaps_review_id เป็น index แล้ว)
        out = (df.drop_duplicates("gmaps_review_id").set_index("gmaps_review_id")[info_cols]  # ข้อมูลรีวิวละ 1 แถว
               .join(labels).join(conf)  # ต่อคอลัมน์ผลและความมั่นใจ
               .reset_index().reindex(columns=cols)  # เอา gmaps_review_id กลับเป็นคอลัมน์ แล้วจัดลำดับ
               .sort_values(["branch", "date_approx"], ascending=[True, False]))  # เรียงตามสาขา แล้วรีวิวใหม่ก่อน
    out.to_csv(_ensure_dir(cfg["output"]["review_predictions_file"]), index=False, encoding=CSV_ENCODING)  # เขียนไฟล์


def export(cfg, conn, summary):  # เขียนไฟล์ผลทั้งหมด (6_predict.py เรียก)
    """เขียนไฟล์ผลทั้ง 3 ไฟล์"""
    out = cfg["output"]  # path ไฟล์ผลจาก config
    # ไม่มีชื่อผู้รีวิวในไฟล์นี้ (scraper ไม่ได้เก็บ) เพราะจะถูกนำไปแสดงบนเว็บ
    Path(_ensure_dir(out["branch_summary_file"])).write_text(  # เขียน branch_summary.json (เว็บอ่านไฟล์นี้)
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")  # ensure_ascii=False = เก็บไทยเป็นตัวอักษรจริง
    export_review_predictions(cfg, conn, summary["model_version"], list(summary["aspect_names"]))  # ผลรายรีวิว
    export_branches(cfg, conn)  # รายชื่อสาขา
