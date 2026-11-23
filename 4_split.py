"""ขั้นที่ 4 แปลง Label Studio JSON export -> ตาราง (รีวิว, ด้าน, label) -> แบ่ง train/val/test

รัน    : python 4_split.py   (อ่านค่าจาก configs/split.yml)
ผลลัพธ์: <output_dir>/train.csv, val.csv, test.csv, split_info.json และ log ที่ logs/split_*.log
"""
import csv  # เขียนไฟล์ train/val/test
import json  # อ่านไฟล์ export ของ Label Studio และเขียน split_info.json
import logging  # บันทึกสถิติลงจอและ logs/
import os  # os.chdir
from collections import Counter, defaultdict  # นับจำนวน label / รวม sentiment ต่อด้าน
from pathlib import Path  # จัดการพาธไฟล์ให้ใช้ได้ทั้ง Windows และ Linux

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo

from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402  แบ่งข้อมูลโดยรีวิวเดียวกันอยู่ชุดเดียว + สัดส่วน label ใกล้กัน

from teenoi.config import load_config, setup_logging  # noqa: E402  อ่าน + ตรวจ config / ตั้ง log

CONFIG = "configs/split.yml"  # ไฟล์ config ของขั้นนี้
REQUIRED = ["seed", "input_glob", "output_dir", "teenoi_data_version", "ratios", "conflict_strategy", "aspects", "sentiments"]  # key ที่ต้องมี
LABELS = ["none", "negative", "neutral", "positive"]  # ลำดับ label = label_id 0..3 (5_train.py อ่านจาก split_info.json)
log = logging.getLogger("split")  # logger ของสคริปต์นี้


def load_rows(cfg):  # อ่านไฟล์ label ทั้งหมดเป็นแถว
    """อ่าน export ทุกไฟล์ -> 1 รีวิว ได้ 1 แถวต่อ 1 ด้าน (ด้านที่ไม่ถูก label = none)"""
    aspects, sentiments = cfg["aspects"], cfg["sentiments"]  # ตารางแปลงชื่อใน Label Studio -> รหัส
    rows, conflicts, ties, seen = [], 0, 0, set()  # แถวผลลัพธ์, จำนวนด้านที่ sentiment ขัดกัน, จำนวนที่เสมอ, review_id ที่เจอแล้ว
    files = sorted(Path().glob(cfg["input_glob"]))  # หาไฟล์ export ทั้งหมดตาม pattern เรียงชื่อให้ผลคงที่ทุกครั้ง
    if not files:  # ไม่เจอไฟล์เลย = พาธผิด หรือยังไม่ได้วางไฟล์
        raise FileNotFoundError(f"ไม่พบไฟล์ export ที่ {cfg['input_glob']}")  # หยุดพร้อมบอก pattern ที่ใช้หา
    for f in files:  # ไล่ทีละไฟล์ export
        for task in json.loads(f.read_text(encoding="utf-8")):  # 1 task = 1 รีวิวใน Label Studio
            anns = [a for a in task["annotations"] if not a.get("was_cancelled")]  # ตัด annotation ที่ถูกกด skip
            if not anns:  # รีวิวนี้ยังไม่มีใคร label
                continue  # ข้ามไปรีวิวถัดไป
            rid, review = task["data"]["review_id"], task["data"]["review"]  # รหัสรีวิวและข้อความรีวิว
            if rid in seen:  # รีวิวเดียวกันอยู่หลายไฟล์ -> ใช้อันแรกที่เจอ
                log.warning("review_id %s ซ้ำ (%s) ข้าม", rid, f.name)  # เตือนใน log
                continue  # ข้ามรีวิวนี้
            seen.add(rid)  # จำไว้ว่าเจอรีวิวนี้แล้ว
            result = anns[-1]["result"]  # ใช้ annotation ตัวสุดท้ายในรายการ (กรณีมีคน label หลายคน)
            # ใน Label Studio: span ข้อความ (type=labels) = ด้าน, คำตอบ (type=choices) = sentiment จับคู่กันด้วย region id
            span_aspect = {r["id"]: r["value"]["labels"][0] for r in result if r["type"] == "labels"}  # region id -> ชื่อด้าน
            span_sent = {r["id"]: r["value"]["choices"][0] for r in result if r["type"] == "choices"}  # region id -> sentiment
            by_aspect = defaultdict(list)  # ชื่อด้าน -> sentiment ทุก span ของด้านนั้น
            for sid, a in span_aspect.items():  # ไล่ทุก span ที่ถูกเลือกด้าน
                if a not in aspects:  # ชื่อด้านไม่อยู่ใน config -> หยุด ให้ไปเพิ่มใน config ก่อน
                    raise ValueError(f"ไม่รู้จัก aspect '{a}' (review_id {rid}) เพิ่มใน configs/split.yml")  # หยุดพร้อมบอกชื่อที่ต้องเพิ่ม
                if sid not in span_sent:  # span นี้ลืมเลือก sentiment
                    log.warning("span ไม่มี sentiment (review_id %s) ข้าม", rid)  # เตือนใน log
                    continue  # ข้าม span นี้
                s = span_sent[sid]  # sentiment ของ span นี้ (ชื่อตาม Label Studio)
                if s not in sentiments:  # ชื่อ sentiment ไม่อยู่ใน config -> หยุด
                    raise ValueError(f"ไม่รู้จัก sentiment '{s}' (review_id {rid}) เพิ่มใน configs/split.yml")  # หยุดพร้อมบอกชื่อที่ต้องเพิ่ม
                by_aspect[a].append(sentiments[s])  # เก็บเป็นรหัส negative/neutral/positive
            for a, info in aspects.items():  # สร้างครบทุกด้าน แม้ด้านนั้นไม่ถูก label
                votes = by_aspect.get(a)  # sentiment ทุก span ของด้านนี้
                if not votes:  # ไม่มี span = รีวิวไม่ได้พูดถึงด้านนี้
                    label = "none"  # none = ไม่ได้พูดถึงด้านนี้
                elif len(set(votes)) == 1:  # ทุก span ตรงกัน
                    label = votes[0]  # ใช้ค่านั้นเลย
                else:  # ด้านเดียวกันมีหลาย sentiment ขัดกัน
                    conflicts += 1  # นับไว้รายงาน
                    if cfg["conflict_strategy"] == "drop":  # drop = ตัดด้านนี้ของรีวิวนี้ทิ้ง
                        continue  # ไม่สร้างแถวของด้านนี้
                    top = Counter(votes).most_common()  # majority = นับเสียง เรียงจากมากไปน้อย
                    if top[0][1] == top[1][1]:  # เสียงเสมอ = ความเห็นผสม ไม่ใช่ neutral -> ตัดทิ้ง
                        ties += 1  # นับไว้รายงาน
                        continue  # ไม่สร้างแถวของด้านนี้
                    label = top[0][0]  # เอาเสียงข้างมาก
                rows.append({"review_id": rid, "review": review, "aspect": info["code"],  # 1 แถว = (รีวิว, ด้าน)
                             "aspect_text": info["text"], "label": label, "label_id": LABELS.index(label)})  # ข้อความด้าน, label, label แบบตัวเลข
    log.info("อ่าน %d ไฟล์ ได้ %d รีวิว %d แถว (sentiment ขัดกัน %d คู่ -> %s, เสมอตัดทิ้ง %d)",  # สรุปจำนวนที่อ่านได้
             len(files), len(seen), len(rows), conflicts, cfg["conflict_strategy"], ties)  # ค่าที่ใส่แทน %d / %s ข้างบนตามลำดับ
    return rows  # ทุกแถว (รีวิว, ด้าน, label)


def take_fold(rows, n_splits, seed):  # แบ่ง 1 ครั้ง (ใช้ 2 ครั้ง: แยก test แล้วแยก val)
    """แบ่งข้อมูลเป็น n_splits ส่วน แล้วดึงออกมา 1 ส่วน คืน (ส่วนที่เหลือ, ส่วนที่ดึงออก)"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)  # group = review_id, stratify = label
    rest_idx, fold_idx = next(sgkf.split(rows, [r["label"] for r in rows], [r["review_id"] for r in rows]))  # ใช้แค่ fold แรก
    return [rows[i] for i in rest_idx], [rows[i] for i in fold_idx]  # แปลง index กลับเป็นแถว


def main():  # จุดเริ่มของขั้นที่ 4
    cfg = load_config(CONFIG, REQUIRED)  # อ่าน + ตรวจ config
    setup_logging("INFO", name="split")  # log ลงจอ + logs/split_<เวลา>.log
    out = Path(cfg["output_dir"])  # โฟลเดอร์ผลลัพธ์
    out.mkdir(parents=True, exist_ok=True)  # สร้างถ้ายังไม่มี

    rows = load_rows(cfg)  # อ่าน export -> แถว (รีวิว, ด้าน, label)
    r = cfg["ratios"]  # สัดส่วน train/val/test
    # ponytail: ratio ถูกปัดเป็นจำนวน fold (test 0.1 -> 10 fold) จึงได้สัดส่วนใกล้เคียง ไม่ตรงเป๊ะ
    rest, test = take_fold(rows, round(1 / r["test"]), cfg["seed"])  # ดึง test ออกก่อน
    train, val = take_fold(rest, round((r["train"] + r["val"]) / r["val"]), cfg["seed"])  # แบ่งที่เหลือเป็น train/val
    splits = {"train": train, "val": val, "test": test}  # ชื่อชุด -> แถว

    ids = {n: {x["review_id"] for x in part} for n, part in splits.items()}  # review_id ของแต่ละชุด
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]), "review รั่วข้าม split"  # กันรีวิวเดียวกันอยู่หลายชุด (ข้อมูลรั่วทำให้คะแนน test สูงเกินจริง)

    for name, part in splits.items():  # เขียนไฟล์ทีละชุด
        with open(out / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig = เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน
            w = csv.DictWriter(f, fieldnames=list(part[0]))  # หัวคอลัมน์ = key ของแถว
            w.writeheader()  # เขียนหัวคอลัมน์
            w.writerows(part)  # เขียนทุกแถว
        c = Counter(x["label"] for x in part)  # นับ label ในชุดนี้
        log.info("%-5s %4d รีวิว %5d แถว | %s", name, len(ids[name]), len(part), "  ".join(f"{l}={c[l]}" for l in LABELS))  # สถิติของชุดนี้ลงจอและ log

    info = {"labels": LABELS,  # ข้อมูลที่ 5_train.py ต้องใช้
            "aspects": {v["code"]: v["text"] for v in cfg["aspects"].values()},  # รหัสด้าน -> ข้อความที่ใช้ถามโมเดล
            "teenoi_data_version": cfg["teenoi_data_version"],  # เวอร์ชันข้อมูล ติดไปกับโมเดล
            "conflict_strategy": cfg["conflict_strategy"], "seed": cfg["seed"]}  # บันทึกไว้ว่าแบ่งด้วยค่าอะไร
    (out / "split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")  # บันทึกข้อมูลการแบ่งเป็น json อ่านง่าย
    log.info("บันทึกที่ %s", out)  # บอกตำแหน่งไฟล์


if __name__ == "__main__":  # รันตรงด้วย python 4_split.py (ไม่รันตอนถูก import เช่นใน tests)
    main()  # เริ่มทำงาน
