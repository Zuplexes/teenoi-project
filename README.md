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
ทุกบรรทัดของโค้ดและ config มีคำอธิบายภาษาไทยกำกับ (ท้ายบรรทัด หรือบรรทัดบน)

```text
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
│   ├── labelstudio_interface.xml  # หน้าจอ label ของ Label Studio
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
│   ├── raw/                    #    รีวิวดิบที่ดึงมา
│   ├── clean/                  #    รีวิวหลัง clean (ส่งเข้า Label Studio)
│   ├── label/                  #    ไฟล์ export จาก Label Studio
│   ├── final/                  #    train.csv, val.csv, test.csv, split_info.json
│   └── predict/                #    predict.db + output/ ของขั้น 6
├── models/                     # โมเดลที่เทรนแล้ว
└── reports/                    # รายงานและกราฟผล test ของขั้น 5
```

ข้อมูลใน `data/` และโมเดลใน `models/` ขึ้น git ด้วย clone ไปแล้วรันขั้น 6 หรือเปิดเว็บได้เลย
ที่ไม่ขึ้น git: `.env` (API key), `chrome_profile/` (session Google), `logs/`, `.venv/`

## ติดตั้ง (ครั้งเดียว)

ต้องมี Python 3.10+ และ Google Chrome ในเครื่อง (ไม่ต้องมี GPU)

```powershell
cd D:\ML69\ML-teenoi
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

ทุกครั้งที่เปิด terminal ใหม่ ให้รัน `.venv\Scripts\activate` ก่อน (หน้าบรรทัดจะขึ้น `(.venv)`)

ขั้น 2 และ 3 ต้องมีไฟล์ `.env` ที่ root ของ repo (ไม่ขึ้น git):

```text
LABEL_STUDIO_API_KEY=<Label Studio -> Account & Settings -> Personal Access Token หรือ Access Token>
```

## ขั้นตอนการรัน

### ภาพรวม

```text
login.py ─┬─> 1 ดึงรีวิว ─> 2 upload ─> [คน label บนเว็บ Label Studio] ─> 3 ดึง label ─> 4 แบ่งข้อมูล ─> 5 เทรน ─┐
          │                                                                                                    │
          └──────────────────────────────────────────────────────────> 6 ดึงรีวิวทุกสาขา + ทำนาย + สรุป <──────┘
                                                                                     │
                                                                                     v
                                                                        7 เว็บ (web/output/branch_summary.json)
```

| ขั้น | คำสั่ง | ทำอะไร | ต้องมีก่อน | ได้อะไร | ใช้เวลา |
|---|---|---|---|---|---|
| 0 | `python login.py` | เปิด Chrome ให้ล็อกอิน Google แล้วปิดหน้าต่าง | Google Chrome | `chrome_profile/` | 1 นาที |
| 1 | `python 1_collect_reviews.py` | ดึงรีวิวสาขาใน `collect.yml` + clean | ขั้น 0 | `data/raw/`, `data/clean/` | หลายนาที–ชั่วโมง |
| 2 | `python 2_upload_labelstudio.py` | ส่งรีวิวขึ้น Label Studio | ขั้น 1, ไฟล์ `.env` | task ใน Label Studio | ไม่กี่วินาที |
| – | (ทำบนเว็บ Label Studio) | คน label ด้าน + ความรู้สึก | ขั้น 2 | ผล label บน server | แล้วแต่คน |
| 3 | `python 3_fetch_labels.py` | ดึงผล label กลับมา | ไฟล์ `.env` | `data/label/*.json` | ไม่กี่วินาที |
| 4 | `python 4_split.py` | แปลง label เป็นตาราง แบ่ง train/val/test | ขั้น 3 | `data/final/` | ไม่กี่วินาที |
| 5 | `python 5_train.py` | เทรน Naive Bayes + วัดผล test | ขั้น 4 | `models/`, `reports/` | ไม่กี่วินาที |
| 6 | `python 6_predict.py` | ดึงรีวิวทุกสาขา → ทำนาย → สรุปรายสาขา | ขั้น 5 (+ ขั้น 0 ถ้าดึงรีวิว) | `web/output/branch_summary.json`, `data/predict/` | ไม่กี่วินาที–ชั่วโมง |
| 7 | `cd web` → `python -m http.server 8000` | เปิดเว็บในเครื่องที่ http://localhost:8000 | ขั้น 6 | หน้าเว็บ | ทันที |

### เลือกรันตามงาน

ไม่ต้องรันครบทุกขั้นทุกครั้ง ข้อมูลและโมเดลใน repo พร้อมใช้แล้ว

**ก. แค่ดูเว็บ** (ใช้ผลที่มีอยู่)

```powershell
cd web
python -m http.server 8000      # แล้วเปิด http://localhost:8000
```

**ข. อัปเดตผลบนเว็บจากรีวิวที่มีอยู่แล้ว** (ไม่เปิด browser ไม่ต้องล็อกอิน)

1. ใน `configs/predict.yml` ตั้ง `steps.scrape: false`
2. `python 6_predict.py`
3. ดูเว็บตามข้อ ก.

**ค. ดึงรีวิวใหม่ของทุกสาขาแล้วอัปเดตเว็บ**

1. `python login.py` (ครั้งแรก หรือเมื่อ log เตือน "มุมมองแบบจำกัด")
2. ใน `configs/predict.yml` ตั้ง `steps.scrape: true` และแนะนำ `scraping.incremental: true`, `scraping.skip_if_scraped_within_days: 7` (ดึงเฉพาะรีวิวใหม่ เร็วกว่ามาก)
3. `python 6_predict.py`

**ง. มี label เพิ่มใน Label Studio → เทรนโมเดลใหม่**

```powershell
python 3_fetch_labels.py
python 4_split.py
python 5_train.py
python 6_predict.py       # ทำนายใหม่ทุกรีวิวด้วยโมเดลใหม่อัตโนมัติ
```

**จ. เก็บรีวิวชุดใหม่ไปให้คน label**

1. ใส่ลิงก์สาขาใน `configs/collect.yml` → `scraper.urls`
2. `python 1_collect_reviews.py`
3. `python 2_upload_labelstudio.py` แล้วให้คน label บนเว็บ Label Studio ต่อด้วยข้อ ง.

**ฉ. ตรวจว่าโค้ดยังทำงาน**

```powershell
python tests/test_collect.py
python tests/test_predict.py
node web/js/ui.test.mjs        # ต้องมี Node.js
```

## รายละเอียดแต่ละขั้น

### 1) ดึงรีวิว — `configs/collect.yml`

- ใส่ลิงก์สาขาใน `scraper.urls` หรือตั้ง `steps.find_branches: true` ให้ค้นหาสาขาเอง เปิด/ปิดขั้นย่อยที่ `steps:`
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
| `web/output/branch_summary.json` | สรุปรายสาขา รายด้าน + ตัวอย่างรีวิว **เว็บอ่านไฟล์นี้ตรง ๆ** |
| `data/predict/output/review_predictions.csv` | ผลทำนายรายรีวิว |
| `data/predict/output/branches.csv` | รายชื่อสาขา |

### 7) เว็บ — `web/`

เว็บหาสาขาใกล้ตัว พร้อมผลวิเคราะห์รีวิว 5 ด้าน (แผนที่ OpenStreetMap + Leaflet, static ล้วน ไม่มี build)

```text
web/index.html      โครงหน้า
web/css/style.css   หน้าตา
web/js/config.js    ค่าคงที่ (path ข้อมูล, จุดเริ่มแผนที่, จำนวนสาขาที่แสดง, สีตาม label, เกณฑ์ ±0.3 ต้องตรงกับ configs/predict.yml)
web/js/geo.js       ระยะทาง, GPS, ค้นที่อยู่ (Nominatim), ลิงก์นำทาง Google Maps
web/js/map.js       แผนที่ Leaflet และหมุด
web/js/ui.js        รายการสาขา, popup, ข้อความสถานะ
web/js/main.js      จุดเริ่ม: โหลดข้อมูลและผูก event
web/output/         branch_summary.json ที่ 6_predict.py เขียนให้
```

- เปิดในเครื่อง: `cd web` → `python -m http.server 8000` → http://localhost:8000
  (ดับเบิลคลิกเปิดไฟล์ตรง ๆ ไม่ได้ เพราะ ES module และ fetch ใช้กับ `file://` ไม่ได้)
- Deploy (Vercel): `cd web` → `vercel login` → `vercel --prod` (Framework = **Other**, build command เว้นว่าง)
  ถ้าต่อ Vercel กับ GitHub ให้ตั้ง **Root Directory = `web`**

## ข้อควรระวัง

- การ scrape ขัดกับ Terms of Service ของ Google ควรใช้บัญชีสำรองและดึงในปริมาณพอเหมาะ
- `chrome_profile/` มี session Google ที่ล็อกอินอยู่ และ `.env` มี API key — ห้าม commit หรือแชร์ (อยู่ใน `.gitignore` แล้ว)
- `data/` มีข้อความรีวิวจาก Google Maps (ไม่มีชื่อผู้รีวิว) ถ้า repo เป็น public ข้อความเหล่านี้จะเปิดเผยด้วย
- log เตือน "มุมมองแบบจำกัด" = session หลุด ให้รัน `python login.py` ใหม่
- ดึงได้ 0 รีวิวทุกสาขา = Google เปลี่ยน class name ให้แก้ dict `SEL` ใน `teenoi/collect/scraper.py`

| อาการ | แก้ |
|---|---|
| `ModuleNotFoundError` | ยังไม่ได้ `.venv\Scripts\activate` หรือยังไม่ได้ `pip install -r requirements.txt` |
| `ไม่พบ data/final/split_info.json` | รัน `python 4_split.py` |
| `ไม่พบโมเดลที่ models/...` | รัน `python 5_train.py` |
| `ไม่รู้จัก aspect/sentiment '...'` | เพิ่มชื่อนั้นใน `configs/split.yml` |
| `ยังไม่ได้ใส่ LABEL_STUDIO_API_KEY` | สร้างไฟล์ `.env` ตามหัวข้อติดตั้ง |
| เว็บขึ้น "โหลดข้อมูลไม่ได้" | เปิดผ่าน `python -m http.server` ไม่ใช่ดับเบิลคลิกไฟล์ |
