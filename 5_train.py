"""ขั้นที่ 5 เทรน Naive Bayes จำแนก sentiment รายด้าน (aspect category sentiment) แล้ววัดผลบน test

input  : ข้อความรีวิว -> 1 โมเดลต่อ 1 ด้าน (food, service, ...) ตอบ none (ไม่พูดถึง) / negative / neutral / positive
feature: char n-gram (ไม่ต้องตัดคำไทย)
รัน    : python 4_split.py ก่อน แล้ว python 5_train.py (อ่านค่าจาก configs/train.yml)
ผลลัพธ์:
  models/aspect_sentiment.joblib  โมเดลของทุกด้าน (6_predict.py ใช้)
  reports/                        test_report.txt, test_metrics.json, test_predictions.csv,
                                  confusion_matrix.png, confusion_matrix_by_aspect.png, f1_by_aspect.png
"""
import csv  # อ่าน train/val/test.csv และเขียนผล
import json  # อ่าน split_info.json และเขียน test_metrics.json
import logging  # log ความคืบหน้า
import os  # os.chdir
from datetime import datetime  # เวลาที่เทรน ใช้เป็นเวอร์ชันโมเดล
from pathlib import Path  # จัดการพาธไฟล์

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo

import joblib  # noqa: E402  บันทึกโมเดล sklearn ลงไฟล์
import numpy as np  # noqa: E402  คำนวณ array
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402  ข้อความ -> จำนวนครั้งที่เจอแต่ละ n-gram
from sklearn.metrics import classification_report, f1_score  # noqa: E402  คำนวณคะแนน
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402  ต่อ vectorizer + โมเดล เป็นชิ้นเดียว

from teenoi.config import load_config, setup_logging  # noqa: E402  อ่าน config + ตั้ง log
from teenoi.plot import plot_test  # noqa: E402  วาดกราฟผล test

CONFIG = "configs/train.yml"
log = logging.getLogger("train")


def read_csv(path):
    """อ่าน csv เป็น list ของ dict (utf-8-sig = ตัด BOM ที่ 4_split.py ใส่ไว้ให้ Excel)"""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def report(out, test, models, labels):
    """วัดผล test รายด้าน เขียนรายงาน + กราฟลง out คืน metrics ภาพรวม"""
    y = np.array([labels.index(r["label"]) for r in test])  # label จริง -> id
    probs = np.zeros((len(test), len(labels)))  # ความน่าจะเป็นทุก label ของทุกแถว
    for a, clf in models.items():
        idx = [i for i, r in enumerate(test) if r["aspect"] == a]  # แถวของด้านนี้
        cols = [labels.index(k) for k in clf.classes_]  # ลำดับ class ของโมเดล -> ลำดับ labels (label ที่ไม่มีใน train = 0)
        probs[np.ix_(idx, cols)] = clf.predict_proba([test[i]["review"] for i in idx])
    p = probs.argmax(1)  # label ที่ทาย

    out.mkdir(parents=True, exist_ok=True)
    args = dict(labels=range(len(labels)), target_names=labels, digits=4, zero_division=0)  # ตั้งค่ารายงานร่วมกัน
    text = ["=== ภาพรวม ===", classification_report(y, p, **args)]
    for a in models:
        i = [j for j, r in enumerate(test) if r["aspect"] == a]
        text += [f"=== {a} ===", classification_report(y[i], p[i], **args)]
    (out / "test_report.txt").write_text("\n".join(text), encoding="utf-8")
    # ผลทายรายแถว: กรอง correct=0 = เคสที่ผิด, confidence ต่ำ = โมเดลไม่แน่ใจ
    with open(out / "test_predictions.csv", "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig = เปิดใน Excel ได้
        w = csv.writer(f)
        w.writerow(["review_id", "aspect", "true", "pred", "correct", "confidence", *[f"p_{l}" for l in labels], "review"])
        for r, yi, pi, pr in zip(test, y, p, probs):
            w.writerow([r["review_id"], r["aspect"], labels[yi], labels[pi], int(yi == pi), round(float(pr[pi]), 4),
                        *pr.round(4).tolist(), r["review"]])
    metrics = plot_test(test, y, p, labels, out)  # กราฟ + metrics ภาพรวม/รายด้าน
    (out / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics["overall"]


def main():
    cfg = load_config(CONFIG, ["data_dir", "model_file", "report_dir", "features.ngram_range", "features.min_df", "alpha"])
    setup_logging("INFO", name="train")
    data = Path(cfg["data_dir"])
    if not (data / "split_info.json").exists():  # ยังไม่ได้รัน 4_split.py
        raise SystemExit(f"ไม่พบ {data / 'split_info.json'} รัน python 4_split.py ก่อน")
    info = json.loads((data / "split_info.json").read_text(encoding="utf-8"))  # label, ด้าน, เวอร์ชันข้อมูล
    labels, aspects = info["labels"], info["aspects"]  # aspects: code -> ชื่อด้าน
    raw = {s: read_csv(data / f"{s}.csv") for s in ("train", "val", "test")}
    log.info("แถว train=%d val=%d test=%d", *(len(raw[s]) for s in ("train", "val", "test")))

    # ---- เทรน 1 โมเดลต่อด้าน บน train แล้วดูคะแนนบน val (ใช้ตอนปรับ alpha) ----
    f = cfg["features"]
    models = {}  # ด้าน -> pipeline (ข้อความ -> label)
    for a in aspects:
        tr, va = ([r for r in raw[s] if r["aspect"] == a] for s in ("train", "val"))  # แถวของด้านนี้
        models[a] = make_pipeline(
            CountVectorizer(analyzer="char_wb", ngram_range=tuple(f["ngram_range"]), min_df=f["min_df"]),
            MultinomialNB(alpha=cfg["alpha"]),
        ).fit([r["review"] for r in tr], [r["label"] for r in tr])
        val_f1 = f1_score([r["label"] for r in va], models[a].predict([r["review"] for r in va]), average="macro")
        log.info("%-12s val f1_macro=%.4f", a, val_f1)

    # ---- วัดผลบน test (ครั้งเดียว ตอนจบ) ----
    m = report(Path(cfg["report_dir"]), raw["test"], models, labels)
    log.info("test accuracy=%.4f f1_macro=%.4f | f1 ราย label %s", m["accuracy"], m["f1_macro"], m["f1"])

    # ---- บันทึกโมเดล (6_predict.py โหลดไฟล์นี้) ----
    path = Path(cfg["model_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": models,  # ด้าน -> pipeline
        "aspects": aspects, "labels": labels,  # รหัสด้าน -> ชื่อด้าน, ลำดับ label
        "teenoi_data_version": info["teenoi_data_version"],  # เวอร์ชัน cleaner ของข้อมูลที่ใช้เทรน
        "test_metrics": {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]},
        "trained_at": datetime.now().isoformat(timespec="seconds"),  # ใช้เป็นเวอร์ชันโมเดล (6_predict.py ทำนายใหม่เมื่อเปลี่ยน)
    }, path)
    log.info("บันทึกโมเดลที่ %s | รายงาน: %s", path, cfg["report_dir"])


if __name__ == "__main__":
    main()
