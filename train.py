"""เทรนโมเดลจำแนก sentiment รายด้าน (aspect category sentiment) แล้ววัดผลบน test

input  : คู่ประโยค (ข้อความด้าน เช่น "อาหาร", รีวิว) -> โมเดลตอบว่ารีวิวพูดถึงด้านนั้นแบบไหน
label  : none (ไม่พูดถึง) / negative / neutral / positive
รัน    : python split.py ก่อน แล้ว python train.py (อ่านค่าจาก configs/train_config.yml)
ผลลัพธ์:
  <output_dir>/<ชื่อโมเดล>/best/  โมเดลที่ดีที่สุด (เลือกจาก F1 macro บน val) พร้อมใช้งาน
  <report_dir>/<ชื่อโมเดล>/       test_report.txt, test_metrics.json, test_predictions.csv, log_history.json
                                  training_curves.png, confusion_matrix.png, confusion_matrix_by_aspect.png, f1_by_aspect.png
"""
import csv  # อ่าน train/val/test.csv และเขียน test_predictions.csv
import json  # อ่าน split_info.json และเขียนไฟล์ผล .json
import logging  # บันทึก log ลงจอและ logs/train.log
from collections import Counter  # นับจำนวนแต่ละ label เพื่อคำนวณ class weight
from datetime import datetime  # เวลาที่เทรนเสร็จ บันทึกใน training_info.json
from pathlib import Path  # จัดการพาธไฟล์

import numpy as np  # คำนวณ array / softmax
import torch  # ไลบรารี deep learning (GPU)
import yaml  # อ่าน config .yml
from sklearn.metrics import accuracy_score, classification_report, f1_score  # คำนวณคะแนน
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,  # โมเดล/tokenizer จาก Hugging Face
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from plot import plot, plot_test  # ฟังก์ชันวาดกราฟ (plot.py)

ROOT = Path(__file__).parent  # โฟลเดอร์โปรเจกต์ ทุกพาธใน config อิงจากที่นี่
log = logging.getLogger("train")  # logger ของสคริปต์นี้


def read_csv(path):
    """อ่าน csv เป็น list ของ dict (utf-8-sig = ตัด BOM ที่ split.py ใส่ไว้ให้ Excel)"""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    """เขียน dict เป็นไฟล์ json อ่านง่าย ภาษาไทยไม่ถูกแปลงเป็น \\uXXXX"""
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class WeightedTrainer(Trainer):
    """Trainer ที่ใช้ cross entropy แบบมี class weight และ label smoothing"""

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)  # ตั้งค่า Trainer ตามปกติ
        self.class_weights = class_weights  # tensor น้ำหนักต่อ label (None = ทุก label เท่ากัน)
        self.label_smoothing = label_smoothing  # 0 = ปิด

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # แยก label ออก ไม่ให้โมเดลคิด loss เอง
        outputs = model(**inputs)  # ให้โมเดลทาย -> logits
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None  # ย้าย weight ไป GPU เดียวกัน
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w, label_smoothing=self.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    """คะแนนบน val ทุก epoch (Trainer ใช้ f1_macro เลือก best)"""
    y, p = pred.label_ids, pred.predictions.argmax(-1)  # label จริง, label ที่ logits สูงสุด
    return {"accuracy": accuracy_score(y, p), "f1_macro": f1_score(y, p, average="macro")}


def main():
    cfg = yaml.safe_load((ROOT / "configs/train_config.yml").read_text(encoding="utf-8"))  # อ่าน config
    resume, eval_only = cfg.get("resume") or False, cfg.get("eval_only", False)  # โหมดรัน
    t, m = cfg["training"], cfg["model"]  # ค่าการเทรน, ค่าโมเดล
    set_seed(cfg["seed"])  # ล็อก random ให้ผลซ้ำได้
    name = Path(m["name"]).name  # ชื่อโมเดลสั้นๆ เช่น xlm-roberta-base ใช้เป็นชื่อโฟลเดอร์ย่อย
    data_dir = ROOT / cfg["data_dir"]  # โฟลเดอร์ผลของ split.py
    out_dir = ROOT / cfg["output_dir"] / name  # ที่เก็บ checkpoint และ best/
    report_dir = ROOT / cfg["report_dir"] / name  # ที่เก็บรายงานและกราฟ
    if not (data_dir / "split_info.json").exists():  # ยังไม่ได้รัน split.py
        raise FileNotFoundError(f"ไม่พบ {data_dir / 'split_info.json'} รัน python split.py ก่อน")
    info = json.loads((data_dir / "split_info.json").read_text(encoding="utf-8"))  # label, ด้าน, เวอร์ชันข้อมูล
    labels = info["labels"]  # ["none", "negative", "neutral", "positive"]
    id2label = dict(enumerate(labels))  # 0 -> "none", ...

    tok = AutoTokenizer.from_pretrained(m["name"])  # tokenizer ของโมเดลฐาน (ดาวน์โหลดครั้งแรก)

    def encode(rows):
        """แปลงแถวเป็น token: ประโยคแรก = ข้อความด้าน, ประโยคที่สอง = รีวิว"""
        enc = tok([r["aspect_text"] for r in rows], [r["review"] for r in rows],
                  truncation="only_second", max_length=m["max_length"])  # ตัดเฉพาะรีวิวถ้ายาวเกิน
        return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                 "labels": int(r["label_id"])} for i, r in enumerate(rows)]

    raw = {s: read_csv(data_dir / f"{s}.csv") for s in ("train", "val", "test")}  # แถวดิบ (เก็บไว้เขียนรายงาน)
    ds = {s: encode(rows) for s, rows in raw.items()}  # แถวที่ tokenize แล้ว
    log.info("แถว train=%d val=%d test=%d", *(len(ds[s]) for s in ("train", "val", "test")))

    weights = None  # ค่าเริ่มต้น: ไม่ใช้ class weight
    if t["use_class_weights"]:  # label ที่มีน้อยได้น้ำหนักมาก โมเดลจะไม่มองข้าม
        c = Counter(x["labels"] for x in ds["train"])  # จำนวนแถวต่อ label ใน train
        n = len(ds["train"])  # จำนวนแถวทั้งหมด
        p = t.get("class_weight_power", 1.0)  # 1 = ตามสัดส่วนเต็ม, 0.5 = รากที่สอง (นุ่มลง)
        weights = torch.tensor([(n / (len(labels) * c[i])) ** p if c[i] else 0.0 for i in range(len(labels))])  # n/(k*count)
        log.info("class weights %s", dict(zip(labels, weights.round(decimals=2).tolist())))

    model = AutoModelForSequenceClassification.from_pretrained(  # eval_only = โหลด best/ ที่เทรนไว้, ไม่งั้น = โมเดลฐาน
        str(out_dir / "best") if eval_only else m["name"],
        num_labels=len(labels), id2label=id2label, label2id={v: k for k, v in id2label.items()})

    args = TrainingArguments(
        output_dir=str(out_dir), seed=cfg["seed"],  # ที่เก็บ checkpoint
        learning_rate=t["learning_rate"], num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"], per_device_eval_batch_size=t["batch_size"] * 2,  # ตอน eval ไม่เก็บ gradient ใช้ batch ใหญ่ได้
        gradient_accumulation_steps=t["grad_accum_steps"],  # สะสม gradient หลาย batch ก่อนอัปเดต
        warmup_steps=t["warmup_ratio"], weight_decay=t["weight_decay"],  # warmup_steps เป็นทศนิยม < 1 = สัดส่วนของ step ทั้งหมด
        fp16=t["fp16"] and torch.cuda.is_available(),  # fp16 ใช้ได้เฉพาะบน GPU
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,  # วัด val และเซฟทุก epoch เก็บไว้ 2 อันล่าสุด (+best)
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,  # จบแล้วโหลด epoch ที่ F1 macro สูงสุด
        logging_steps=50, report_to="none",  # log loss ทุก 50 step, ไม่ส่งไป wandb/tensorboard
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),  # เติม padding ให้ยาวเท่ากันในแต่ละ batch
        compute_metrics=compute_metrics, class_weights=weights, label_smoothing=t.get("label_smoothing", 0.0),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])],  # หยุดเมื่อ val ไม่ดีขึ้น N epoch
    )
    report_dir.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์รายงาน
    if not eval_only:  # โหมดเทรน
        trainer.train(resume_from_checkpoint=resume)  # False = ใหม่, True = ต่อ checkpoint ล่าสุด, "path" = ต่อจาก checkpoint ที่ระบุ
        write_json(report_dir / "log_history.json", trainer.state.log_history)  # ประวัติ loss/คะแนนทุก step
        plot(trainer.state.log_history, report_dir / "training_curves.png")  # กราฟการเทรน

    # ---- วัดผลบน test (ครั้งเดียว ตอนจบ ห้ามใช้เลือกโมเดล) ----
    pred = trainer.predict(ds["test"])  # ให้โมเดลทาย test ทั้งหมด
    y, p = pred.label_ids, pred.predictions.argmax(-1)  # label จริง, label ที่ทาย
    logits = pred.predictions - pred.predictions.max(-1, keepdims=True)  # ลบค่าสูงสุดก่อน exp กันตัวเลขล้น
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)  # softmax -> ความน่าจะเป็นแต่ละ label
    test_metrics = {"accuracy": round(accuracy_score(y, p), 4), "f1_macro": round(f1_score(y, p, average="macro"), 4)}
    report_args = dict(labels=range(len(labels)), target_names=labels, digits=4, zero_division=0)  # ตั้งค่ารายงานร่วมกัน
    report = ["=== ภาพรวม ===", classification_report(y, p, **report_args)]  # precision/recall/F1 ภาพรวม
    for code in info["aspects"]:  # รายงานแยกรายด้าน
        idx = [i for i, r in enumerate(raw["test"]) if r["aspect"] == code]  # แถวของด้านนี้
        report += [f"=== {code} ===", classification_report(y[idx], p[idx], **report_args)]
    (report_dir / "test_report.txt").write_text("\n".join(report), encoding="utf-8")
    log.info("\n%s", report[1])  # แสดงรายงานภาพรวมบนจอ/ใน log

    # ผลทายรายแถว: กรอง correct=0 = เคสที่ผิด, confidence ต่ำ = โมเดลไม่แน่ใจ
    with open(report_dir / "test_predictions.csv", "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig = เปิดใน Excel ได้
        w = csv.writer(f)
        w.writerow(["review_id", "aspect", "true", "pred", "correct", "confidence", *[f"p_{l}" for l in labels], "review"])
        for r, yi, pi, pr in zip(raw["test"], y, p, probs):  # 1 แถวต่อ (รีวิว, ด้าน)
            w.writerow([r["review_id"], r["aspect"], labels[yi], labels[pi], int(yi == pi), round(float(pr[pi]), 4),
                        *pr.round(4).tolist(), r["review"]])
    write_json(report_dir / "test_metrics.json", plot_test(raw["test"], y, p, labels, report_dir))  # วาดกราฟ test + บันทึก metrics รายด้าน
    if eval_only:  # โหมดวัดผลอย่างเดียว ไม่ต้องเซฟโมเดลซ้ำ
        log.info("test %s | รายงาน: %s", test_metrics, report_dir)
        return

    # ---- เซฟโมเดลที่ดีที่สุดไว้ที่ best/ สำหรับนำไปใช้งาน ----
    best = out_dir / "best"
    trainer.save_model(str(best))  # น้ำหนักโมเดล + config
    tok.save_pretrained(str(best))  # tokenizer
    write_json(best / "label_map.json", {str(k): v for k, v in id2label.items()})  # id -> ชื่อ label
    write_json(best / "training_info.json", {  # ข้อมูลประกอบ: โมเดลนี้มาจากอะไร ได้คะแนนเท่าไร
        "task": "aspect_category", "aspects": info["aspects"], "include_none": True,
        "max_length": m["max_length"], "base_model": m["name"],
        "teenoi_data_version": info["teenoi_data_version"], "test_metrics": test_metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds")})
    log.info("test %s | โมเดล: %s | รายงาน: %s", test_metrics, best, report_dir / "test_report.txt")


if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)  # โฟลเดอร์ log
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s",  # log ไลบรารีอื่นเฉพาะ WARNING ขึ้นไป
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "logs/train.log", encoding="utf-8")])  # ออกจอ + ไฟล์
    log.setLevel(logging.INFO)  # log ของสคริปต์นี้แสดงระดับ INFO ด้วย (ไม่เอา log HTTP ของ Hugging Face)
    main()
