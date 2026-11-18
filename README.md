# teenoi-train

เทรนโมเดลจำแนกความรู้สึกรายด้าน (Aspect Category Sentiment) ของรีวิวร้านอาหารภาษาไทย
ด้วย `xlm-roberta-base`

ทุกรีวิวจะถูกถาม 5 ด้าน แต่ละด้านตอบ 1 ใน 4 แบบ

| ด้าน (aspect) | label |
|---|---|
| อาหาร `food` · บริการ `service` · บรรยากาศ `environment` · เวลาในการรอ `waiting` · ความสะอาด `cleanliness` | `none` (ไม่พูดถึง) · `negative` · `neutral` · `positive` |

ตัวอย่าง: รีวิว "อาหารอร่อย แต่รอนานมาก" → food = positive, waiting = negative, ด้านอื่น = none

---

## โครงสร้างโปรเจกต์

```
teenoi-train/
├── split.py                 # 1) แปลงไฟล์ Label Studio -> train/val/test
├── train.py                 # 2) เทรนโมเดล + วัดผล test + สร้างรายงาน
├── plot.py                  # วาดกราฟรายงาน (train.py เรียกใช้ / รันเองเพื่อวาดใหม่ได้)
├── requirements.txt         # ไลบรารีที่ต้องติดตั้ง
├── configs/
│   ├── split_config.yml     # ค่าของ split.py
│   └── train_config.yml     # ค่าของ train.py
├── dataset/
│   └── label/*.json         # ไฟล์ export จาก Label Studio (ข้อมูลต้นฉบับ)
├── data/final/              # ผลของ split.py: train.csv, val.csv, test.csv, split_info.json
├── models/<รอบ>/            # โมเดลที่เทรนแล้ว (ไม่ขึ้น git เพราะไฟล์ใหญ่)
├── reports/<รอบ>/           # รายงานและกราฟของแต่ละรอบ
└── logs/                    # split.log, train.log (ไม่ขึ้น git)
```

---

## 1. ติดตั้ง (ทำครั้งเดียว)

ต้องมี Python 3.12 และการ์ดจอ NVIDIA (ทดสอบบน GPU 6GB) ถ้าไม่มี GPU ก็รันได้แต่ช้ามาก

```powershell
cd D:\NLP\teenoi-train
python -m venv .venv
.venv\Scripts\activate

# ติดตั้ง torch แบบใช้ GPU ก่อน (เลือก cuXXX ให้ตรงกับไดรเวอร์ ดูได้จาก nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu134

# ติดตั้งที่เหลือ
pip install -r requirements.txt
```

เช็กว่าเห็น GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"   # ต้องได้ True
```

---

## 2. เตรียมข้อมูล

1. ใน Label Studio เลือกโปรเจกต์ → **Export** → **JSON**
2. วางไฟล์ทั้งหมดไว้ใน `dataset/label/` (กี่ไฟล์ก็ได้ ถ้ารีวิวซ้ำกันข้ามไฟล์จะใช้อันแรกที่เจอ)

รูปแบบ labeling ใน Label Studio ที่สคริปต์รองรับ:
- ลากเลือกข้อความแล้วติดป้าย **ด้าน** (เช่น `อาหาร (Food)`)
- แล้วเลือก **sentiment** ของ span นั้น (เช่น `ดี (Positive)`)
- ชื่อด้าน/sentiment ต้องตรงกับใน `configs/split_config.yml` ถ้าเจอชื่อที่ไม่รู้จักสคริปต์จะหยุดและบอกว่าต้องเพิ่มอะไร
- เจอ label ผิด → แก้ใน Label Studio แล้ว export ใหม่มาทับไฟล์เดิมใน `dataset/label/`

---

## 3. แบ่งข้อมูล train / val / test

```powershell
python split.py
```

สิ่งที่สคริปต์ทำ:
1. อ่านทุกไฟล์ใน `dataset/label/`
2. แปลง 1 รีวิว → 5 แถว (1 แถวต่อด้าน) ด้านที่ไม่ถูก label = `none`
3. ด้านเดียวกันมีหลาย sentiment ขัดกัน → ใช้เสียงข้างมาก ถ้าเสมอตัดทิ้ง (`conflict_strategy`)
4. แบ่ง 80/10/10 โดย **รีวิวเดียวกันอยู่ชุดเดียว** และสัดส่วน label แต่ละชุดใกล้กัน
5. เขียนผลที่ `data/final/` และสถิติที่ `logs/split.log`

ตัวอย่างผลบนจอ:

```
อ่าน 7 ไฟล์ ได้ 2983 รีวิว 14680 แถว (sentiment ขัดกัน 383 คู่ -> majority, เสมอตัดทิ้ง 235)
train 2387 รีวิว 11744 แถว | none=7724  negative=1388  neutral=278  positive=2354
val    298 รีวิว  1469 แถว | none=967  negative=174  neutral=34  positive=294
test   298 รีวิว  1467 แถว | none=964  negative=173  neutral=34  positive=296
```

`seed` เดิม + ไฟล์ label เดิม = ได้ข้อมูลชุดเดิมทุกครั้ง

---

## 4. เทรน

1. เปิด `configs/train_config.yml` **ตั้งชื่อรอบใหม่ทุกครั้ง** ไม่งั้นจะเขียนทับรอบเก่า

   ```yaml
   output_dir: models/teenoi_5
   report_dir: reports/teenoi_5
   ```

2. รัน

   ```powershell
   python train.py
   ```

ใช้เวลาประมาณ 15–35 นาทีบน GPU 6GB (ขึ้นกับว่า early stopping หยุดที่ epoch ไหน)
ระหว่างเทรนวัดผลบน val ทุก epoch และเก็บ epoch ที่ **F1 macro บน val สูงสุด** เป็นโมเดลที่ดีที่สุด
จบแล้ววัดผลบน test ครั้งเดียว

> ห้ามปิด terminal ระหว่างเทรน กราฟและรายงานจะถูกสร้างตอนจบเท่านั้น
> ถ้าหลุดกลางทาง ตั้ง `resume: true` แล้วรันใหม่ จะเทรนต่อจาก checkpoint ล่าสุด (ถ้ามี)

### โหมดอื่นใน train_config.yml

| ต้องการ | ตั้งค่า |
|---|---|
| เทรนใหม่ตั้งแต่ต้น | `resume: false`, `eval_only: false` |
| เทรนต่อจาก checkpoint ล่าสุด | `resume: true` |
| เทรนต่อจาก checkpoint ที่ระบุ | `resume: "models/teenoi_5/xlm-roberta-base/checkpoint-500"` |
| วัดโมเดลที่เทรนแล้วบน test ปัจจุบัน (ไม่เทรน) เช่นหลัง export label ใหม่ | `eval_only: true` (ใช้ `output_dir` ชี้ไปที่รอบนั้น, `report_dir` ตั้งเป็นโฟลเดอร์ใหม่ถ้าไม่อยากทับรายงานเก่า) |

ค่าการเทรนอื่นๆ (learning rate, batch size, class weight, ...) มีคำอธิบายทุกบรรทัดใน `configs/train_config.yml`

---

## 5. ดูผล

### โมเดล — `models/<รอบ>/xlm-roberta-base/best/`

| ไฟล์ | คือ |
|---|---|
| `model.safetensors`, `config.json` | น้ำหนักโมเดล |
| `tokenizer.json`, `tokenizer_config.json` | tokenizer |
| `label_map.json` | id → ชื่อ label |
| `training_info.json` | ด้าน, เวอร์ชันข้อมูล, คะแนน test, เวลาที่เทรน |

### รายงาน — `reports/<รอบ>/xlm-roberta-base/`

| ไฟล์ | ใช้ดูอะไร |
|---|---|
| `training_curves.png` | loss และ F1/accuracy บน val ต่อ epoch ดาวแดง = ค่าดีที่สุด, เส้นประ = epoch ที่ถูกเลือก |
| `confusion_matrix.png` | ภาพรวมทาย label ไหนผิดเป็นอะไร + accuracy / F1 macro ของ test |
| `confusion_matrix_by_aspect.png` | confusion matrix แยกรายด้าน + accuracy / F1 รายด้านและ Overall |
| `f1_by_aspect.png` | ตารางสี F1 ราย label × รายด้าน + accuracy ดูว่าด้าน/label ไหนอ่อน |
| `test_report.txt` | precision / recall / F1 ละเอียด ภาพรวมและรายด้าน |
| `test_metrics.json` | ตัวเลขเดียวกับ f1_by_aspect.png แบบอ่านด้วยโปรแกรม |
| `test_predictions.csv` | ผลทายทุกแถว กรอง `correct = 0` ดูเคสที่ผิด, `confidence` ต่ำ = โมเดลไม่แน่ใจ |
| `log_history.json` | ประวัติการเทรนดิบ (ใช้วาด training_curves.png ใหม่) |

### อ่านคะแนนยังไง

- **accuracy** ดูสูงได้ง่ายเพราะ `none` มีมากกว่า 60% ของข้อมูล
- **F1 macro** = เฉลี่ย F1 ทุก label เท่าๆ กัน → ใช้เทียบรอบเป็นหลัก แต่ `neutral` ใน test มีน้อย (~20–35 แถว) ทายถูก/ผิดเพิ่ม 1–2 แถวก็ทำให้ F1 macro ขยับมาก ดู F1 ของ negative/positive ประกอบด้วย
- เทียบรอบที่ใช้ **test ชุดเดียวกัน** เท่านั้น (ถ้า label ใน test เปลี่ยน ให้ใช้ `eval_only` วัดรอบเก่าใหม่)

### วาดกราฟใหม่ (ไม่ต้องเทรน)

ถ้าแก้หน้าตากราฟใน `plot.py` แล้วอยากให้รอบเก่าได้กราฟแบบใหม่:

```powershell
python plot.py reports/teenoi_1/xlm-roberta-base reports/teenoi_2/xlm-roberta-base
```

---

## 6. ขั้นตอนสรุป (รอบถัดไป)

```powershell
.venv\Scripts\activate
# (ถ้ามี label ใหม่) วางไฟล์ export ใน dataset/label/
python split.py
# แก้ configs/train_config.yml: output_dir / report_dir = ชื่อรอบใหม่
python train.py
# ดูผลที่ reports/<รอบใหม่>/xlm-roberta-base/
```

## ปัญหาที่พบบ่อย

| อาการ | แก้ |
|---|---|
| `ไม่พบ data/final/split_info.json` | ยังไม่ได้รัน `python split.py` |
| `ไม่รู้จัก aspect '...'` / `ไม่รู้จัก sentiment '...'` | เพิ่มชื่อนั้นใน `configs/split_config.yml` |
| `CUDA out of memory` | ลด `batch_size` เป็น 4 แล้วเพิ่ม `grad_accum_steps` เป็น 8 (effective batch เท่าเดิม) |
| `torch.cuda.is_available()` ได้ False | ติดตั้ง torch แบบ CUDA ใหม่ตามข้อ 1 |
| โฟลเดอร์ reports ว่าง | เทรนไม่จบ (ปิด terminal / Ctrl+C) รันใหม่ หรือ `resume: true` |
