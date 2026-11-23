"""โหลดโมเดลจาก 5_train.py (joblib) แล้วทำนายทุกรีวิวที่ยังไม่มีผลของโมเดลเวอร์ชันนี้ ครบทุกด้าน"""
import logging  # log ความคืบหน้า
from pathlib import Path  # จัดการ path

import joblib  # โหลดโมเดล sklearn

import teenoi  # ใช้เช็กเวอร์ชัน cleaner ให้ตรงกับตอนเทรน

from . import db  # เวลาปัจจุบัน

log = logging.getLogger(__name__)  # logger ของโมดูลนี้


def load_model(cfg):  # โหลดโมเดล
    """dict ที่ 5_train.py บันทึก: models (ด้าน -> pipeline), aspects (ด้าน -> ชื่อ), trained_at (ใช้เป็นเวอร์ชัน), ..."""
    path = Path(cfg["model"]["file"])  # path ไฟล์โมเดลจาก config
    if not path.exists():  # ยังไม่ได้เทรน
        raise SystemExit(f"ไม่พบโมเดลที่ {path} รัน python 5_train.py ก่อน")  # หยุดพร้อมบอกวิธีแก้
    return joblib.load(path)  # อ่านไฟล์กลับเป็น dict


def predict(cfg, conn):  # ขั้นย่อย predict
    """ทำนายทุกรีวิวที่ยังไม่มีผลครบทุกด้านจากโมเดลเวอร์ชันนี้ แล้วบันทึกลงตาราง predictions"""
    bundle = load_model(cfg)  # โหลดโมเดล
    version, models = bundle["trained_at"], bundle["models"]  # เวอร์ชันโมเดล / ด้าน -> pipeline
    if bundle.get("teenoi_data_version") != teenoi.__version__:  # cleaner คนละเวอร์ชันกับตอนเทรน
        log.warning("โมเดลเทรนกับข้อมูลที่ clean ด้วย teenoi %s แต่ตอนนี้คือ %s ผลอาจเพี้ยนถ้า cleaner เปลี่ยน",  # เตือนใน log
                    bundle.get("teenoi_data_version"), teenoi.__version__)  # ค่าที่ใส่แทน %s ข้างบนตามลำดับ

    # รีวิวที่ยังทำนายไม่ครบทุกด้านด้วยโมเดลเวอร์ชันนี้ (เปลี่ยนโมเดล = ทำนายใหม่ทั้งหมด)
    todo = conn.execute(  # รีวิวที่ต้องทำนาย
        "SELECT gmaps_review_id, review_clean FROM reviews r WHERE review_clean IS NOT NULL AND "  # รีวิวที่ผ่านการ clean
        "(SELECT COUNT(*) FROM predictions p WHERE p.gmaps_review_id = r.gmaps_review_id "  # นับผลที่มีอยู่
        " AND p.model_version = ?) < ?", (version, len(models))).fetchall()  # น้อยกว่าจำนวนด้าน = ยังไม่ครบ
    if not todo:  # ไม่มีงาน
        log.info("ไม่มีรีวิวที่ต้องทำนาย (model_version %s)", version)  # บอกใน log
        return {"pairs": 0}  # ไม่ได้ทำนายอะไร

    ids, texts = [r[0] for r in todo], [r[1] for r in todo]  # แยกรหัสรีวิว / ข้อความ
    log.info("ต้องทำนาย %d รีวิว x %d ด้าน = %d คู่", len(todo), len(models), len(todo) * len(models))  # บอกจำนวนงาน
    ts = db.now()  # เวลาที่ทำนาย
    # ponytail: ทำนายทุกรีวิวรวดเดียวต่อด้าน (Naive Bayes เร็ว หมื่นรีวิวไม่กี่วินาที) ถ้ารีวิวเป็นล้านค่อยแบ่ง batch
    for code, model in models.items():  # ทีละด้าน
        proba = model.predict_proba(texts)  # แถว = รีวิว, คอลัมน์ = model.classes_ (เช่น negative/neutral/none/positive)
        best = proba.argmax(1)  # class ที่ความน่าจะเป็นสูงสุด
        conn.executemany("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?)", [  # บันทึกผล (มีอยู่แล้ว = ทับ)
            (rid, code, str(model.classes_[j]), round(float(p[j]), 4), version, ts) for rid, j, p in zip(ids, best, proba)])  # 1 แถว = (รีวิว, ด้าน, label, ความมั่นใจ, เวอร์ชัน, เวลา)
        conn.commit()  # บันทึกทีละด้าน
    return {"pairs": len(todo) * len(models)}  # จำนวนคู่ที่ทำนาย
