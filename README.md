# teenoi-predict

หาทุกสาขาสุกี้ตี๋น้อย → ดึงรีวิว → clean → ทำนาย 5 ด้าน → สรุปผลต่อสาขา → `data/output/branch_summary.json`

## ติดตั้ง

```bash
pip install -r requirements.txt     # ติดตั้ง teenoi-data จาก ../teenoi-data ด้วย
playwright install chromium
```

วางโมเดลไว้ที่ `model/best/` (หรือตั้ง `model.source: hf_hub`) แล้วแก้ค่าใน `configs/Predict_config.yml` ตามต้องการ

## รัน

```bash
python login.py     # ครั้งแรกครั้งเดียว
python main.py
python tests/test_offline.py   # ทดสอบส่วนที่ไม่ต้องใช้ browser/โมเดล
```

ทดสอบกับ 2-3 สาขาก่อน: ตั้ง `steps.find_branches: false` แล้วใส่ลิงก์สาขาใน `branches.extra_urls`

รันซ้ำได้อย่างปลอดภัย: ข้อมูลสะสมใน `data/predict.db` ดึงเฉพาะรีวิวใหม่ ทำนายเฉพาะรีวิวที่ยังไม่มีผลของโมเดลเวอร์ชันนี้ ถ้าเปลี่ยนโมเดล (`trained_at` เปลี่ยน) จะทำนายใหม่ทั้งหมด
