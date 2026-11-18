"""โหลด Predict_config.yml (ใช้ตัวตรวจ key ของ teenoi_data ซึ่ง error เป็นภาษาไทยอยู่แล้ว)"""
from teenoi_data.config import load_config  # อ่าน yaml + แจ้ง error ถ้าขาด key

REQUIRED = [  # key ที่ต้องมีใน config ไม่งั้นหยุดทำงานตั้งแต่ต้น
    "steps", "browser.profile_dir", "branches.search_queries", "scraping.max_reviews_first_run",  # ขั้นตอน/browser/สาขา/การดึง
    "model.source", "model.batch_size", "model.confidence_threshold",  # โมเดล
    "summary.min_mentions", "summary.positive_threshold", "summary.negative_threshold",  # เกณฑ์สรุป
    "summary.sample_reviews_per_label",  # จำนวนตัวอย่างรีวิว
    "output.db_file", "output.review_predictions_file", "output.branch_summary_file", "output.branches_file",  # ไฟล์ผล
]
MODEL_SOURCE_KEY = {"local": "local_path", "hf_hub": "hf_repo_id"}  # model.source แต่ละแบบต้องมี key ไหนเพิ่ม


def load(path="configs/Predict_config.yml"):
    """โหลด config แล้วตรวจว่า model.source ถูกต้องและมี key ที่ต้องใช้คู่กัน"""
    cfg = load_config(path, REQUIRED)  # โหลด + ตรวจ key ใน REQUIRED
    model = cfg["model"]  # ส่วนตั้งค่าโมเดล
    key = MODEL_SOURCE_KEY.get(model["source"])  # key ที่ต้องมีตาม source
    if not key:  # source ไม่ใช่ local หรือ hf_hub
        raise SystemExit(f"model.source ต้องเป็น local หรือ hf_hub (ได้ {model['source']!r})")
    if not model.get(key):  # ไม่ได้ใส่ local_path / hf_repo_id
        raise SystemExit(f"ไฟล์ config {path} ขาด key ที่จำเป็น: model.{key}")
    return cfg  # config ที่ผ่านการตรวจแล้ว
