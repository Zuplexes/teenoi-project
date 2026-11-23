"""Baseline สำหรับเทียบกับ xlm-roberta: Multinomial Naive Bayes บน bag of char n-gram แยก 1 โมเดลต่อด้าน

ใช้ข้อมูล split เดียวกับ train.py (data/final) และ plot_test เดียวกัน -> test_metrics.json รูปแบบเดียวกัน เทียบกันตรงๆ ได้
รัน    : python split.py ก่อน แล้ว python baseline.py
ผลลัพธ์: reports/baseline/naive_bayes/  test_report.txt, test_metrics.json, confusion_matrix*.png, f1_by_aspect.png
         แล้วพิมพ์ตารางเทียบกับทุกรอบใน reports/*/*/test_metrics.json
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from plot import plot_test  # ใช้ฟังก์ชันวาดกราฟเดียวกับ train.py ให้รายงานหน้าตาเหมือนกัน

ROOT = Path(__file__).parent
DATA = ROOT / "data/final"


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def naive_bayes(alpha):
    # ponytail: นับ char n-gram แทนการตัดคำไทย (ไม่ต้องลง pythainlp)
    return make_pipeline(CountVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2), MultinomialNB(alpha=alpha))


def fit_predict(make, raw, aspect):
    """เทรนบน train ของด้านนี้ คืนผลทาย (val, test)"""
    tr, va, te = ([r for r in raw[s] if r["aspect"] == aspect] for s in ("train", "val", "test"))
    clf = make().fit([r["review"] for r in tr], [int(r["label_id"]) for r in tr])
    return clf.predict([r["review"] for r in va]), clf.predict([r["review"] for r in te])


def run(out, makers, raw, info):
    """makers = ตัวเลือก hyperparameter หลายตัว เลือกตัวที่ F1 macro บน val ดีสุด รายด้าน (test ไม่ใช้เลือก)"""
    labels, aspects = info["labels"], list(info["aspects"])
    y = np.array([int(r["label_id"]) for r in raw["test"]])
    p = np.zeros_like(y)
    chosen = {}
    for a in aspects:
        y_val = [int(r["label_id"]) for r in raw["val"] if r["aspect"] == a]
        best = max(((k, *fit_predict(m, raw, a)) for k, m in makers.items()),
                   key=lambda t: f1_score(y_val, t[1], average="macro"))
        chosen[a] = best[0]
        p[[i for i, r in enumerate(raw["test"]) if r["aspect"] == a]] = best[2]

    out.mkdir(parents=True, exist_ok=True)
    args = dict(labels=range(len(labels)), target_names=labels, digits=4, zero_division=0)
    report = ["=== ภาพรวม ===", classification_report(y, p, **args)]
    for a in aspects:
        i = [j for j, r in enumerate(raw["test"]) if r["aspect"] == a]
        report += [f"=== {a} (เลือก {chosen[a]}) ===", classification_report(y[i], p[i], **args)]
    (out / "test_report.txt").write_text("\n".join(report), encoding="utf-8")
    metrics = plot_test(raw["test"], y, p, labels, out)
    (out / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if not (DATA / "split_info.json").exists():
        raise FileNotFoundError(f"ไม่พบ {DATA / 'split_info.json'} รัน python split.py ก่อน")
    info = json.loads((DATA / "split_info.json").read_text(encoding="utf-8"))
    raw = {s: read_csv(DATA / f"{s}.csv") for s in ("train", "val", "test")}
    run(ROOT / "reports/baseline/naive_bayes", {f"alpha={a}": (lambda a=a: naive_bayes(a)) for a in (0.1, 0.3, 1)}, raw, info)

    # เทียบ baseline กับทุกรอบที่มี test_metrics.json
    print(f"\n{'model':32} {'acc':>7} {'f1_macro':>9}  f1 per label")
    for f in sorted(ROOT.glob("reports/*/*/test_metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))["overall"]
        print(f"{f.parent.parent.name + '/' + f.parent.name:32} {m['accuracy']:7.4f} {m['f1_macro']:9.4f}  {m['f1']}")


if __name__ == "__main__":
    main()
