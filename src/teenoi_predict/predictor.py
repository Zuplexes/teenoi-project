"""โหลดโมเดล + ทำนายแบบ batch ด้วยคู่ (aspect_text, review_clean) แบบเดียวกับตอนเทรน"""
import json  # อ่านไฟล์ข้อมูลโมเดล
import logging  # log ความคืบหน้า
import time  # จับเวลาไว้ประมาณเวลาที่เหลือ
from pathlib import Path  # จัดการ path

import teenoi_data  # ใช้เช็กเวอร์ชันให้ตรงกับตอนเทรน

from . import db  # เวลาปัจจุบัน

log = logging.getLogger(__name__)
LOG_EVERY = 20  # แสดงความคืบหน้าทุกกี่ batch


def _read_json(path):
    """อ่านไฟล์ json (utf-8)"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_dir(cfg):
    """โฟลเดอร์โมเดล: ดาวน์โหลดจาก Hugging Face Hub หรือใช้โฟลเดอร์ในเครื่องตาม model.source"""
    model = cfg["model"]
    if model["source"] == "hf_hub":  # โหลดจาก Hub
        from huggingface_hub import snapshot_download  # import ตอนใช้ ไม่ต้องติดตั้งถ้าใช้ local
        return Path(snapshot_download(model["hf_repo_id"]))  # ดาวน์โหลด (หรือใช้ cache) แล้วคืนโฟลเดอร์
    path = Path(model["local_path"])  # โฟลเดอร์ในเครื่อง
    if not (path / "training_info.json").exists():  # ไม่มีไฟล์โมเดล
        raise SystemExit(f"ไม่พบโมเดลที่ {path} (ต้องมี config.json, model.safetensors, tokenizer, "
                         "label_map.json, training_info.json)")
    return path


def training_info(cfg):
    """ข้อมูลตอนเทรน: trained_at (ใช้เป็นเวอร์ชัน), aspects, max_length, teenoi_data_version"""
    return _read_json(model_dir(cfg) / "training_info.json")


def pick_device(setting, torch):
    """auto = ใช้ GPU ถ้ามี ไม่งั้น CPU, ค่าอื่นใช้ตามที่ตั้ง"""
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


def predict(cfg, conn):
    """ทำนายทุกรีวิวที่ยังไม่มีผลครบทุกด้านจากโมเดลเวอร์ชันนี้ แล้วบันทึกลงตาราง predictions"""
    import torch  # import ตอนใช้ (ช้า) ขั้นอื่นจะได้ไม่ต้องรอ
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # โหลดโมเดลและ tokenizer

    settings = cfg["model"]  # ตั้งค่าโมเดลจาก config
    path = model_dir(cfg)  # โฟลเดอร์โมเดล
    info = _read_json(path / "training_info.json")  # ข้อมูลตอนเทรน
    labels = _read_json(path / "label_map.json")  # เลข class -> ชื่อผล เช่น "0" -> "positive"
    version, aspects = info["trained_at"], info["aspects"]  # aspects: code -> ข้อความที่ใช้ถามโมเดล
    if info.get("teenoi_data_version") != teenoi_data.__version__:  # cleaner คนละเวอร์ชันกับตอนเทรน
        log.warning("โมเดลเทรนกับ teenoi_data %s แต่ที่ติดตั้งอยู่คือ %s ผลอาจเพี้ยนถ้า cleaner เปลี่ยน",
                    info.get("teenoi_data_version"), teenoi_data.__version__)

    # รีวิวที่ยังทำนายไม่ครบทุกด้านด้วยโมเดลเวอร์ชันนี้ (เปลี่ยนโมเดล = ทำนายใหม่ทั้งหมด)
    todo = conn.execute(
        "SELECT gmaps_review_id, review_clean FROM reviews r WHERE review_clean IS NOT NULL AND "  # รีวิวที่ผ่านการ clean
        "(SELECT COUNT(*) FROM predictions p WHERE p.gmaps_review_id = r.gmaps_review_id "  # นับผลที่มีอยู่
        " AND p.model_version = ?) < ?", (version, len(aspects))).fetchall()  # น้อยกว่าจำนวนด้าน = ยังไม่ครบ
    if not todo:  # ไม่มีงาน
        log.info("ไม่มีรีวิวที่ต้องทำนาย (model_version %s)", version)
        return {"pairs": 0}
    todo.sort(key=lambda r: len(r[1]))  # ความยาวใกล้กันอยู่ batch เดียวกัน padding น้อย เร็วขึ้นมากบน CPU

    device = pick_device(settings.get("device", "auto"), torch)  # cpu หรือ cuda
    tokenizer = AutoTokenizer.from_pretrained(path)  # ตัวตัดคำของโมเดล
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()  # โหลดโมเดล ย้ายไป device และเข้าโหมดทำนาย
    per_batch = max(1, settings["batch_size"] // len(aspects))  # จำนวนรีวิวต่อ batch (บันทึกครบทุกด้านของรีวิวพร้อมกัน)
    n_pairs = len(todo) * len(aspects)  # จำนวนคู่ทั้งหมด
    n_batches = -(-len(todo) // per_batch)  # จำนวน batch (หารปัดขึ้น)
    log.info("ต้องทำนาย %d รีวิว x %d ด้าน = %d คู่ บน %s", len(todo), len(aspects), n_pairs, device)

    start = time.time()  # เวลาเริ่ม
    for batch_no, i in enumerate(range(0, len(todo), per_batch), 1):  # วนทีละ batch (นับจาก 1)
        pairs = [(review_id, code, question, review)  # 1 รีวิว จับคู่กับทุกด้าน
                 for review_id, review in todo[i:i + per_batch] for code, question in aspects.items()]
        enc = tokenizer([p[2] for p in pairs], [p[3] for p in pairs],  # ป้อนเป็นคู่ (คำถามด้าน, รีวิว) เหมือนตอนเทรน
                        truncation="only_second", max_length=info["max_length"],  # ยาวเกินให้ตัดเฉพาะรีวิว
                        padding=True, return_tensors="pt").to(device)  # เติมให้ยาวเท่ากัน แปลงเป็น tensor
        with torch.no_grad():  # ไม่คำนวณ gradient (ทำนายอย่างเดียว เร็วและประหยัดหน่วยความจำ)
            conf, idx = model(**enc).logits.softmax(-1).max(-1)  # ความน่าจะเป็นสูงสุด และ class ที่ได้
        ts = db.now()  # เวลาที่ทำนาย
        conn.executemany("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?)", [  # บันทึกผล (มีอยู่แล้ว = ทับ)
            (p[0], p[1], labels[str(j)], round(c, 4), version, ts)
            for p, j, c in zip(pairs, idx.tolist(), conf.tolist())])
        conn.commit()  # บันทึกทุก batch โปรแกรมหยุดกลางทางก็ไม่เสียงานที่ทำไปแล้ว
        if batch_no == 1 or batch_no % LOG_EVERY == 0 or batch_no == n_batches:  # แสดงความคืบหน้าเป็นระยะ
            elapsed = time.time() - start  # เวลาที่ผ่านไป
            log.info("batch %d/%d ผ่านไป %.0f วินาที เหลืออีกประมาณ %.0f นาที",
                     batch_no, n_batches, elapsed, elapsed / batch_no * (n_batches - batch_no) / 60)  # ประมาณจากเวลาเฉลี่ยต่อ batch
    return {"pairs": n_pairs}
