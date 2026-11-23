# ML-teenoi

วิเคราะห์รีวิวร้านสุกี้ตี๋น้อยจาก Google Maps แยกรายด้านด้วย **Naive Bayes**
ตั้งแต่ดึงรีวิว → ให้คน label → เทรน → สรุปผลทุกสาขา → แสดงบนเว็บ อยู่ใน repo เดียว

ทุกรีวิวถูกถาม 5 ด้าน แต่ละด้านตอบ 1 ใน 4 แบบ

| ด้าน (aspect) | label |
|---|---|
| อาหาร `food` · บริการ `service` · บรรยากาศ `environment` · เวลาในการรอ `waiting` · ความสะอาด `cleanliness` | `none` (ไม่พูดถึง) · `negative` · `neutral` · `positive` |

ตัวอย่าง: "อาหารอร่อย แต่รอนานมาก" → food = positive, waiting = negative, ด้านอื่น = none

## โครงสร้าง

ทุกสคริปต์รันด้วย `python <ไฟล์>.py` **ไม่มี argument** ตั้งค่าทุกอย่างในไฟล์ `configs/` ของขั้นนั้น

```
ML-teenoi/
├── login.py                    # 0) ล็อกอิน Google ครั้งแรกครั้งเดียว (ใช้ร่วมกันขั้น 1 และ 6)
├── 1_collect_reviews.py        # 1) ดึงรีวิวจาก Google Maps + clean           -> data/raw/, data/clean/
├── 2_upload_labelstudio.py     # 2) upload รีวิวขึ้น Label Studio ให้คน label
├── 3_fetch_labels.py           # 3) ดึงผล label กลับมา                         -> data/label/
├── 4_split.py                  # 4) แปลง label -> train/val/test               -> data/final/
├── 5_train.py                  # 5) เทรน Naive Bayes + วัดผล test              -> models/, reports/
├── 6_predict.py                # 6) ดึงรีวิวทุกสาขา -> ทำนาย -> สรุป            -> web/output/, data/predict/
├── configs/
│   ├── collect.yml             #    ของขั้น 1 (+ login.py)
│   ├── labelstudio.yml         #    ของขั้น 2 (upload:) และ 3 (fetch:)
│   ├── labelstudio_interface.xml  # Labeling Interface ของ Label Studio
│   ├── split.yml               #    ของขั้น 4
│   ├── train.yml               #    ของขั้น 5
│   └── predict.yml             #    ของขั้น 6
├── teenoi/                     # โค้ดที่สคริปต์ด้านบนเรียกใช้
│   ├── config.py               #    อ่าน config + ตั้ง log (ใช้ทุกขั้น)
│   ├── labelstudio.py          #    เรียก Label Studio API (ขั้น 2, 3)
│   ├── plot.py                 #    กราฟรายงาน (ขั้น 5)
│   ├── collect/                #    browser, scraper, branch_finder, cleaner (ขั้น 1 และ 6 ใช้ตัวเดียวกัน)
│   └── predict/                #    collector, db, predictor, summarizer, exporter (ขั้น 6)
├── web/                        # 7) เว็บหาสาขาใกล้ฉัน + สรุป 5 ด้าน (static ล้วน ไม่มี build)
├── tests/                      # ทดสอบส่วนที่ไม่ต้องเปิด browser
├── data/
│   ├── raw/, clean/            #    รีวิวที่ดึงมา (ไม่ขึ้น git)
│   ├── label/                  #    ไฟล์ export จาก Label Studio
│   ├── final/                  #    train.csv, val.csv, test.csv, split_info.json
│   └── predict/                #    predict.db + output/ ของขั้น 6 (ไม่ขึ้น git)
├── models/                     # โมเดลที่เทรนแล้ว (ไม่ขึ้น git เทรนใหม่ได้ในไม่กี่วินาที)
└── reports/                    # รายงานและกราฟผล test ของขั้น 5
```

## ติดตั้ง (ครั้งเดียว)

ต้องมี Python 3.10+ และ Google Chrome ในเครื่อง (ไม่ต้องมี GPU)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

ขั้น 2 และ 3 ต้องมีไฟล์ `.env` ที่ root (ไม่ขึ้น git):

```
LABEL_STUDIO_API_KEY=<Label Studio -> Account & Settings -> Personal Access Token หรือ Access Token>
```

## วิธีรัน

```powershell
python login.py                 # ครั้งแรก: ล็อกอิน Google ในหน้าต่าง Chrome แล้วปิดหน้าต่าง
python 1_collect_reviews.py     # ดึงรีวิว + clean
python 2_upload_labelstudio.py  # ส่งขึ้น Label Studio -> ให้คน label ในเว็บ
python 3_fetch_labels.py        # ดึงผล label กลับมา
python 4_split.py               # แบ่ง train/val/test
python 5_train.py               # เทรน + รายงาน
python 6_predict.py             # ใช้งานจริงกับทุกสาขา -> อัปเดตข้อมูลให้เว็บ
```

ไม่ต้องรันครบทุกขั้นทุกครั้ง เช่น label เพิ่ม → รัน 3, 4, 5 / อยากอัปเดตผลสาขาบนเว็บ → รัน 6 แล้ว deploy เว็บ

### 1) ดึงรีวิว — `configs/collect.yml`

- ใส่ลิงก์สาขาใน `scraper.urls` หรือตั้ง `steps.find_branches: true` ให้ค้นหาสาขาเอง เปิด/ปิดแต่ละขั้นที่ `steps:`
- รันซ้ำได้: รีวิวที่อยู่ใน `data/raw/reviews_raw.csv` แล้วจะไม่ถูกดึงซ้ำ อยากดึงใหม่หมดให้ลบไฟล์นั้น
- ขั้น clean: ตัดคำแปลของ Google, ลบ URL/HTML, `pythainlp.util.normalize`, ลดตัวอักษรซ้ำ, ตัดรีวิวสั้น/ภาษาอื่น/ซ้ำ
  ไม่ตัดคำและไม่ลบ stopwords (จะทำให้ "ไม่อร่อย" กลายเป็น "อร่อย")
- แก้ `teenoi/collect/cleaner.py` เมื่อไร ให้เพิ่ม `__version__` ใน `teenoi/__init__.py`

| ไฟล์ | คือ |
|---|---|
| `data/raw/reviews_raw.csv` | รีวิวดิบทุกสาขา |
| `data/clean/reviews_clean.csv` | `review_id, review` สำหรับ Label Studio |
| `data/clean/reviews_clean_full.csv` | + คอลัมน์ของ raw |
| `data/raw/<สาขา>.csv`, `data/clean/<สาขา>.csv` | แยกต่อสาขา |

### 2–3) Label Studio — `configs/labelstudio.yml`

- `2_upload_labelstudio.py` สร้างโปรเจกต์ให้ถ้ายังไม่มี (ใช้ `labelstudio_interface.xml`) รีวิวที่ upload แล้วจะไม่ถูกส่งซ้ำ
- `3_fetch_labels.py` ดึงทุกโปรเจกต์ที่ชื่อขึ้นต้นด้วย `fetch.project_prefix` ลง `data/label/` (หรือ Export → JSON เองแล้ววางไว้ที่นั่น)
- วิธี label: ลากเลือกข้อความแล้วติดป้าย **ด้าน** จากนั้นเลือก **sentiment** ของ span นั้น ชื่อต้องตรงกับใน `configs/split.yml`

### 4) แบ่งข้อมูล — `configs/split.yml`

1 รีวิว → 5 แถว (1 แถวต่อด้าน) ด้านที่ไม่ถูก label = `none` ด้านเดียวกันมีหลาย sentiment ขัดกัน → ใช้เสียงข้างมาก เสมอตัดทิ้ง
แบ่ง 80/10/10 โดย **รีวิวเดียวกันอยู่ชุดเดียว**

> scikit-learn คนละเวอร์ชันอาจแบ่งได้ไม่เหมือนเดิมเป๊ะ (ลองกับ 1.7.2 ได้ต่างจากไฟล์ใน repo เล็กน้อย)
> ถ้าไม่ได้เพิ่ม label ใหม่ ใช้ `data/final/` ที่อยู่ใน repo ไปเลย ไม่ต้องรัน 4 ใหม่

### 5) เทรน — `configs/train.yml`

- **Multinomial Naive Bayes** แยก 1 โมเดลต่อ 1 ด้าน feature = **char n-gram** (ชิ้นตัวอักษร 1–4 ตัว ไม่ต้องตัดคำไทย)
- เทรนบน train ดูคะแนนแต่ละด้านบน val ใน log (ใช้ตอนปรับ `alpha`) แล้ววัดผลบน test ครั้งเดียวตอนจบ
- ใช้เวลาไม่กี่วินาทีบน CPU

ผลบน test (1,467 แถว): **accuracy 0.756, F1 macro 0.532** (F1: none 0.85 · negative 0.56 · neutral 0.08 · positive 0.64)

| ไฟล์ | คือ |
|---|---|
| `models/aspect_sentiment.joblib` | โมเดลของทุกด้าน (ขั้น 6 ใช้) |
| `reports/test_report.txt` | precision / recall / F1 ภาพรวมและรายด้าน |
| `reports/test_metrics.json` | ตัวเลขเดียวกับ `f1_by_aspect.png` แบบอ่านด้วยโปรแกรม |
| `reports/test_predictions.csv` | ผลทายทุกแถว กรอง `correct = 0` ดูเคสที่ผิด |
| `reports/confusion_matrix.png`, `confusion_matrix_by_aspect.png`, `f1_by_aspect.png` | กราฟผล test |

อ่านคะแนน: accuracy ดูสูงง่ายเพราะ `none` มีกว่า 60% ให้ดู **F1 macro** เป็นหลัก `neutral` มีน้อยมาก (~30 แถวใน test) จึงทายได้ยากที่สุด

### 6) ใช้งาน — `configs/predict.yml`

- ทดสอบกับ 2–3 สาขาก่อน: `steps.find_branches: false` แล้วใส่ลิงก์ใน `branches.extra_urls`
- รันซ้ำได้: ข้อมูลสะสมใน `data/predict/predict.db` ดึงเฉพาะรีวิวใหม่ ทำนายเฉพาะรีวิวที่ยังไม่มีผลของโมเดลนี้ เทรนใหม่ = ทำนายใหม่ทั้งหมดอัตโนมัติ
- สรุปรายด้าน: คะแนน = (บวก − ลบ) / จำนวนที่พูดถึง นับเฉพาะผลที่มั่นใจ ≥ `model.confidence_threshold`
  (Naive Bayes มั่นใจสูงเกินจริง ค่า 0.6 แทบไม่ได้ตัดอะไรออก)

| ไฟล์ | คือ |
|---|---|
| `web/output/branch_summary.json` | สรุปรายสาขา รายด้าน + ตัวอย่างรีวิว **เว็บอ่านไฟล์นี้ตรง ๆ** (ขึ้น git) |
| `data/predict/output/review_predictions.csv` | ผลทำนายรายรีวิว |
| `data/predict/output/branches.csv` | รายชื่อสาขา |

### 7) เว็บ — `web/`

เว็บหาสาขาใกล้ตัว พร้อมผลวิเคราะห์รีวิว 5 ด้าน (แผนที่ OpenStreetMap + Leaflet, static ล้วน ไม่มี build)

```
web/index.html      โครงหน้า
web/css/style.css   หน้าตา
web/js/config.js    ค่าคงที่ (path ข้อมูล, จุดเริ่มแผนที่, จำนวนสาขาที่แสดง, สีตาม label, เกณฑ์ ±0.3 ต้องตรงกับ configs/predict.yml)
web/js/geo.js       ระยะทาง, GPS, ค้นที่อยู่ (Nominatim), ลิงก์นำทาง Google Maps
web/js/map.js       แผนที่ Leaflet และหมุด
web/js/ui.js        รายการสาขา, popup, ข้อความสถานะ
web/js/main.js      จุดเริ่ม: โหลดข้อมูลและผูก event
web/output/         branch_summary.json ที่ 6_predict.py เขียนให้
```

รันในเครื่อง (ดับเบิลคลิกเปิดไฟล์ตรง ๆ ไม่ได้ เพราะ ES module และ fetch ใช้กับ `file://` ไม่ได้):

```powershell
cd web
python -m http.server 8000      # แล้วเปิด http://localhost:8000
```

Deploy (Vercel): `cd web` แล้ว `vercel login` → `vercel --prod` (Framework = **Other**, build command เว้นว่าง)
ถ้าต่อ Vercel กับ GitHub ให้ตั้ง **Root Directory = `web`**

อัปเดตข้อมูลบนเว็บ: `python 6_predict.py` → `cd web` → `vercel --prod`

## ทดสอบ

```powershell
python tests/test_collect.py
python tests/test_predict.py
node web/js/ui.test.mjs
```

## ข้อควรระวัง

- การ scrape ขัดกับ Terms of Service ของ Google ควรใช้บัญชีสำรองและดึงในปริมาณพอเหมาะ
- `chrome_profile/` มี session Google ที่ล็อกอินอยู่ และ `.env` มี API key — ห้าม commit หรือแชร์ (อยู่ใน `.gitignore` แล้ว)
- log เตือน "มุมมองแบบจำกัด" = session หลุด ให้รัน `python login.py` ใหม่
- ดึงได้ 0 รีวิวทุกสาขา = Google เปลี่ยน class name ให้แก้ dict `SEL` ใน `teenoi/collect/scraper.py`

| อาการ | แก้ |
|---|---|
| `ไม่พบ data/final/split_info.json` | รัน `python 4_split.py` |
| `ไม่พบโมเดลที่ models/...` | รัน `python 5_train.py` |
| `ไม่รู้จัก aspect/sentiment '...'` | เพิ่มชื่อนั้นใน `configs/split.yml` |
| `ยังไม่ได้ใส่ LABEL_STUDIO_API_KEY` | สร้างไฟล์ `.env` ตามหัวข้อติดตั้ง |
