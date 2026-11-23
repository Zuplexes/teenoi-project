"""สรุปผลต่อสาขา ต่อด้าน"""
from collections import Counter, defaultdict  # นับจำนวน / dict ที่มีค่าเริ่มต้น
from datetime import date, timedelta  # คำนวณวันเริ่มช่วงเวลาที่สนใจ

from . import db, predictor  # เวลาปัจจุบัน / ข้อมูลโมเดล

SENTIMENTS = ["positive", "neutral", "negative"]  # ผลทำนายที่นับ ("none" = ไม่ได้พูดถึงด้านนั้น ไม่นับ)
NOT_ENOUGH = "ข้อมูลไม่พอ"  # ผลเมื่อพูดถึงน้อยเกินไป


def verdict(score, mentions, settings):
    """แปลงคะแนน (-1 ถึง 1) เป็นคำตัดสิน ดี/เฉยๆ/แย่/ข้อมูลไม่พอ"""
    if mentions < settings["min_mentions"]:  # พูดถึงน้อยเกิน เชื่อถือไม่ได้
        return NOT_ENOUGH
    if score >= settings["positive_threshold"]:  # บวกมากพอ
        return "ดี"
    if score <= settings["negative_threshold"]:  # ลบมากพอ
        return "แย่"
    return "เฉยๆ"  # อยู่ระหว่างกลาง


def build_branches(branches, rows, aspect_codes, settings, threshold):
    """branches: แถวจากตาราง branches, rows: ผลทำนายที่อยู่ในช่วงเวลาที่สนใจแล้ว
    threshold: ความมั่นใจขั้นต่ำที่นับ คืน list ของ dict สรุปรายสาขา"""
    analyzed = defaultdict(set)  # branch_id -> รหัสรีวิวที่ผ่านโมเดล
    counted = defaultdict(list)  # (branch_id, aspect) -> ผลทำนายที่นับได้
    for r in rows:
        analyzed[r["branch_id"]].add(r["gmaps_review_id"])  # นับทุกรีวิวที่ทำนาย
        if r["label"] != "none" and r["confidence"] >= threshold:  # พูดถึงด้านนี้ และโมเดลมั่นใจพอ
            counted[r["branch_id"], r["aspect"]].append(r)

    out = []  # ผลสรุปทุกสาขา
    for b in branches:
        aspects, samples = {}, {}  # ผลต่อด้าน / ตัวอย่างรีวิวต่อด้าน
        for code in aspect_codes:
            rs = counted[b["branch_id"], code]  # ผลของสาขานี้ ด้านนี้
            n = Counter(r["label"] for r in rs)  # จำนวน positive/neutral/negative
            raw = (n["positive"] - n["negative"]) / len(rs) if rs else 0.0  # คะแนน = (บวก - ลบ) / ทั้งหมด
            label = verdict(raw, len(rs), settings)  # คำตัดสิน
            score = round(raw, 3) if rs and label != NOT_ENOUGH else None  # ข้อมูลไม่พอ = ไม่แสดงคะแนน
            aspects[code] = {"label": label, "score": score, **{k: n[k] for k in SENTIMENTS}}  # ผลด้านนี้
            best = sorted(rs, key=lambda r: (r["confidence"], r["date_approx"] or ""), reverse=True)  # มั่นใจสุด/ใหม่สุดก่อน
            samples[code] = {k: [r["review_clean"] for r in best if r["label"] == k][:settings["sample_reviews_per_label"]]
                             for k in SENTIMENTS}  # ตัวอย่างรีวิวแยกตามผล
        out.append({  # สรุปของสาขานี้
            "branch_id": b["branch_id"], "name": b["name"], "address": b["address"],  # ข้อมูลสาขา
            "lat": b["lat"], "lng": b["lng"], "gmaps_url": b["gmaps_url"], "google_rating": b["google_rating"],  # พิกัด/ลิงก์/ดาว Google
            "reviews_analyzed": len(analyzed[b["branch_id"]]),  # จำนวนรีวิวที่ผ่านโมเดล
            "last_scraped_at": (b["last_scraped_at"] or "")[:10] or None,  # เก็บแค่วันที่ YYYY-MM-DD
            "aspects": aspects, "sample_reviews": samples,  # ผลต่อด้าน และตัวอย่างรีวิว
        })
    return out


def star_counts(conn, since):
    """นับดาวทุกรีวิวในช่วงเวลา รวมรีวิวที่ให้แต่ดาว (ซึ่งไม่ได้ผ่านโมเดล) คืน branch_id -> Counter{ดาว: จำนวน}"""
    stars = defaultdict(Counter)
    for r in conn.execute("SELECT branch_id, rating FROM reviews WHERE rating BETWEEN 1 AND 5 "
                          "AND (? IS NULL OR date_approx >= ?)", (since, since)):  # since = None คือทุกช่วงเวลา
        stars[r["branch_id"]][r["rating"]] += 1
    return stars


def summarize(cfg, conn):
    """สรุปผลทุกสาขาที่ดึงรีวิวแล้ว คืน dict ที่จะเขียนเป็น branch_summary.json"""
    settings = cfg["summary"]  # เกณฑ์สรุปจาก config
    info = predictor.load_model(cfg)  # ข้อมูลโมเดล (เวอร์ชัน + รายชื่อด้าน)
    months = settings.get("recent_months")  # นับย้อนกี่เดือน (None = ทั้งหมด)
    since = (date.today() - timedelta(days=30 * months)).isoformat() if months else None  # วันเริ่มนับ (เดือนละ 30 วันโดยประมาณ)
    rows = conn.execute(  # ผลทำนายของโมเดลเวอร์ชันนี้ ในช่วงเวลา
        "SELECT r.branch_id, r.gmaps_review_id, r.date_approx, r.review_clean, p.aspect, p.label, p.confidence "
        "FROM predictions p JOIN reviews r USING (gmaps_review_id) "
        "WHERE p.model_version = ? AND (? IS NULL OR r.date_approx >= ?)",
        (info["trained_at"], since, since)).fetchall()
    excluded = set(cfg["branches"].get("exclude_branch_ids") or [])  # สาขาที่ไม่เอา
    branches = [b for b in conn.execute("SELECT * FROM branches WHERE last_scraped_at IS NOT NULL ORDER BY name")
                if b["branch_id"] not in excluded]  # เฉพาะสาขาที่ดึงแล้ว และไม่ถูกตัดออก
    out = build_branches(branches, rows, list(info["aspects"]), settings, cfg["model"]["confidence_threshold"])  # สรุปรายสาขา
    stars = star_counts(conn, since)  # นับดาว
    for b in out:
        c = stars[b["branch_id"]]  # ดาวของสาขานี้
        n = sum(c.values())  # จำนวนรีวิวที่มีดาว
        b["stars"] = {"count": n, "avg": round(sum(k * v for k, v in c.items()) / n, 2) if n else None,  # ดาวเฉลี่ย
                      **{str(k): c[k] for k in range(1, 6)}}  # จำนวนรีวิวแยกตามดาว 1-5
    return {
        "generated_at": db.now(),  # เวลาที่สรุป
        "model_version": info["trained_at"],  # เวอร์ชันโมเดล
        "settings": {"recent_months": months, "min_mentions": settings["min_mentions"]},  # เกณฑ์ที่ใช้ (ให้เว็บแสดงได้)
        "aspect_names": info["aspects"],  # รหัสด้าน -> ชื่อด้าน
        "branches": out,  # ผลรายสาขา
    }
