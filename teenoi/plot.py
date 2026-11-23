"""วาดกราฟผล test ลง reports/ (5_train.py เรียก plot_test)"""
from pathlib import Path  # จัดการพาธไฟล์

import matplotlib  # ไลบรารีวาดกราฟ
matplotlib.use("Agg")  # วาดลงไฟล์อย่างเดียว ไม่เปิดหน้าต่าง (ต้องตั้งก่อน import pyplot)
import matplotlib.pyplot as plt  # noqa: E402  ฟังก์ชันวาดกราฟ
import numpy as np  # noqa: E402  คำนวณ array
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score  # noqa: E402  วาด confusion matrix + คำนวณคะแนน


def save(fig, path):
    """บันทึกกราฟเป็น .png แล้วปิด figure คืนหน่วยความจำ"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี
    fig.tight_layout()  # จัดระยะไม่ให้กราฟย่อยทับกัน
    fig.savefig(path, dpi=120, bbox_inches="tight")  # bbox_inches="tight" = ไม่ตัดหัวกราฟ/ป้ายแกนที่ล้นขอบ
    plt.close(fig)  # ปิด figure


def score_of(y, p):
    """ข้อความ 'acc=... f1_macro=...' F1 macro คิดเฉพาะ label ที่มีอยู่จริงใน y (ตรงกับ f1_by_aspect.png)"""
    f1 = f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)  # label ที่ไม่มีตัวอย่าง ไม่นับเป็น 0
    return f"acc={accuracy_score(y, p):.4f}  f1_macro={f1:.4f}"


def plot_test(rows, y, p, labels, out_dir):
    """วาดกราฟผล test 3 ไฟล์ และคืนค่า metrics รายกลุ่ม (overall + รายด้าน) สำหรับ test_metrics.json
    rows = แถว test (ต้องมีคอลัมน์ aspect), y = label จริง, p = label ที่ทาย (เป็น id), labels = ชื่อ label"""
    ids = list(range(len(labels)))  # label id ทั้งหมด ให้ทุกกราฟมีครบ 4 แถว/คอลัมน์ แม้บางด้านไม่มีบาง label
    groups = {"overall": np.arange(len(y))}  # กลุ่ม -> index แถว เริ่มจากภาพรวม
    for a in dict.fromkeys(r["aspect"] for r in rows):  # รายชื่อด้าน ไม่ซ้ำ เรียงตามที่เจอ
        groups[a] = np.array([i for i, r in enumerate(rows) if r["aspect"] == a])  # แถวของด้านนั้น
    cm = dict(labels=ids, display_labels=labels, cmap="Blues", colorbar=False, xticks_rotation=45)  # ตั้งค่า confusion matrix ร่วมกัน

    # 1) confusion_matrix.png: ซ้าย = จำนวนแถว, ขวา = สัดส่วนต่อ label จริง (แนวทแยง = recall)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y, p, values_format="d", ax=axes[0], **cm)  # จำนวน
    ConfusionMatrixDisplay.from_predictions(y, p, normalize="true", values_format=".2f", ax=axes[1], **cm)  # สัดส่วนต่อแถว
    axes[0].set_title("Test confusion matrix (count)")
    axes[1].set_title("Test confusion matrix (row-normalized)")
    fig.suptitle(f"Test: {score_of(y, p)}", fontweight="bold")  # หัวกราฟ: accuracy + F1 macro รวม
    save(fig, Path(out_dir) / "confusion_matrix.png")

    # 2) confusion_matrix_by_aspect.png: แยกรายด้าน แสดงจำนวน (บางด้านบาง label มีน้อยมาก สัดส่วนจะหลอกตา)
    aspects = [g for g in groups if g != "overall"]  # เฉพาะรายด้าน
    fig, axes = plt.subplots(1, len(aspects), figsize=(4.3 * len(aspects), 4.6))  # 1 กราฟต่อด้าน เรียงแนวนอน
    for ax, a in zip(np.atleast_1d(axes), aspects):  # atleast_1d = รองรับกรณีมีด้านเดียว
        i = groups[a]  # แถวของด้านนี้
        ConfusionMatrixDisplay.from_predictions(y[i], p[i], values_format="d", ax=ax, **cm)
        ax.set_title(f"{a} (n={len(i)})\n{score_of(y[i], p[i])}")  # ชื่อด้าน + จำนวนแถว + acc/F1 ของด้านนี้
    fig.suptitle(f"Overall: {score_of(y, p)}", fontweight="bold", y=1.04)  # หัวกราฟรวม (y=1.04 ไม่ทับชื่อกราฟย่อย)
    save(fig, Path(out_dir) / "confusion_matrix_by_aspect.png")

    # 3) f1_by_aspect.png: ตารางสี F1 ของแต่ละ label x แต่ละด้าน + macro + accuracy
    metrics, table = {}, []  # metrics สำหรับ json, table สำหรับวาด
    for g, i in groups.items():  # ไล่ทุกกลุ่ม
        support = np.bincount(y[i], minlength=len(labels))  # จำนวนแถวจริงของแต่ละ label
        # label ที่ไม่มีในกลุ่มนี้ = วัดไม่ได้ ใส่ NaN (ไม่นับเป็น 0 ตอนเฉลี่ย macro)
        f1 = np.where(support > 0, f1_score(y[i], p[i], labels=ids, average=None, zero_division=0), np.nan)
        metrics[g] = {"n": len(i), "accuracy": round(accuracy_score(y[i], p[i]), 4),  # จำนวนแถว, accuracy
                      "f1_macro": round(float(np.nanmean(f1)), 4),  # F1 เฉลี่ยเฉพาะ label ที่วัดได้
                      "f1": {l: None if np.isnan(v) else round(float(v), 4) for l, v in zip(labels, f1)},  # F1 ราย label
                      "support": dict(zip(labels, support.tolist()))}  # จำนวนแถวจริงราย label
        table.append([*f1, np.nanmean(f1), metrics[g]["accuracy"]])  # 1 แถวในตาราง
    table = np.array(table)
    fig, ax = plt.subplots(figsize=(1.4 * table.shape[1] + 2, 0.6 * len(groups) + 1.5))  # ขนาดตามจำนวนแถว/คอลัมน์
    ax.imshow(table, cmap="RdYlGn", vmin=0, vmax=1)  # แดง = 0, เขียว = 1 (NaN = ช่องว่าง)
    ax.set_xticks(range(table.shape[1]), [*labels, "macro", "accuracy"])  # ชื่อคอลัมน์
    ax.axvline(table.shape[1] - 1.5, color="black", lw=2)  # เส้นคั่นก่อนคอลัมน์ accuracy (คนละ metric กับ F1)
    ax.set_yticks(range(len(groups)), list(groups))  # ชื่อแถว
    for r in range(table.shape[0]):  # ใส่ตัวเลขทุกช่อง
        for c in range(table.shape[1]):
            ax.text(c, r, "-" if np.isnan(table[r, c]) else f"{table[r, c]:.2f}", ha="center", va="center")
    ax.set_title("Test F1 by aspect + accuracy (- = no samples)")
    save(fig, Path(out_dir) / "f1_by_aspect.png")
    return metrics
