"""รันครั้งแรกครั้งเดียว: เปิด Chrome ให้ล็อกอิน Google เอง session เก็บใน browser.profile_dir
ใช้ร่วมกันทั้ง 1_collect_reviews.py และ 6_predict.py (browser.profile_dir ใน configs/collect.yml กับ configs/predict.yml ต้องตรงกัน)"""
import os  # os.chdir
from pathlib import Path  # หาโฟลเดอร์ของไฟล์นี้

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo

from teenoi.collect.browser import login  # noqa: E402  เปิด Chrome ให้ล็อกอินเอง
from teenoi.config import load_config  # noqa: E402  อ่าน config

if __name__ == "__main__":  # รันด้วย python login.py
    login(load_config("configs/collect.yml", ["browser.profile_dir"]))  # ต้องมี profile_dir จึงรู้ว่าจะเก็บ session ที่ไหน
