import json
import os
import re
import random
import hashlib

DATA_FILES = ["ChnSentiCorp_lite.json", "ChnSentiCorp_lite_300_full.json"]
OUTPUT_DIR = os.path.join("data", "chnsenticorp_lite")

PII_PATTERNS = {
    "PHONE": re.compile(r"1[3-9][0-9]{9}"),
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "PERSON": re.compile(r"[\u4e00-\u9fa5]{2,4}") # 简单处理，按模型实际场景可换NER工具
}

# 合成PII模板
def synthesize_pii(text, pii_entities):
    result = {}
    if "PERSON" in pii_entities:
        m = re.findall("[\u4e00-\u9fa5]{2,4}", text)
        if m:
            result["person_name"] = m[0]
            result["generation_template"] = "name_entity"
    if "PHONE" in pii_entities:
        m = re.findall(PII_PATTERNS["PHONE"], text)
        if m:
            masked = m[0][:3] + "****" + m[0][-4:]
            result["phone_masked"] = masked
            result["generation_template"] = result.get("generation_template", "") + ",phone_entity"
    if "EMAIL" in pii_entities:
        m = re.findall(PII_PATTERNS["EMAIL"], text)
        if m:
            masked = m[0].split("@")[0][:2] + "***@" + m[0].split("@")[-1]
            result["email_masked"] = masked
            result["generation_template"] = result.get("generation_template", "") + ",email_entity"
    if result:
        result["generation_template"] = result["generation_template"].strip(',')
        return result
    return None

def detect_pii(text):
    entities = []
    if PII_PATTERNS["PHONE"].search(text):
        entities.append("PHONE")
    if PII_PATTERNS["EMAIL"].search(text):
        entities.append("EMAIL")
    # 人名检测：只在文本含有两个以上汉字并含联系方式时，认为是泄漏
    if PII_PATTERNS["PERSON"].search(text):
        if "PHONE" in entities or "EMAIL" in entities:
            entities.append("PERSON")
    return entities

def load_data():
    for fn in DATA_FILES:
        if os.path.exists(fn):
            with open(fn, encoding="utf8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    rows = data["data"]
                elif isinstance(data, list):
                    rows = data
                else:
                    continue
                out = []
                for i, row in enumerate(rows):
                    text = row["text"] if isinstance(row, dict) else row[0]
                    label = row.get("label", 1) if isinstance(row, dict) else int(row[1])
                    out.append({"text": text, "label": label})
                return out
    raise FileNotFoundError("数据文件不存在")

def build_entries(raw_data):
    entries = []
    for idx, item in enumerate(raw_data):
        text = item["text"]
        label = item["label"]
        sample_id = f"chnsc_{idx:06d}"
        pii_entities = detect_pii(text)
        privacy_level = "high_sensitivity" if pii_entities else "general"
        synthetic_pii = synthesize_pii(text, pii_entities) if privacy_level == "high_sensitivity" else None
        pipl_cross_border = False if privacy_level=="high_sensitivity" else True
        privacy_budget_cost = 1.2 if privacy_level=="high_sensitivity" else 0.0
        entry = {
            "sample_id": sample_id,
            "text": text,
            "label": "positive" if label==1 else "negative",
            "privacy_level": privacy_level,
            "pii_entities": pii_entities,
            "pipl_cross_border": pipl_cross_border,
            "synthetic_pii": synthetic_pii,
            "privacy_budget_cost": privacy_budget_cost,
            "metadata": {
                "source": "ChnSentiCorp_lite",
                "domain": "general",
                "length": len(text),
                # MIA攻击评测子集：前10%的高敏感即true
                "mia_test_subset": False
            }
        }
        entries.append(entry)
    # mia_test_subset 规则：前10%高敏感分配
    idxs = [i for i, e in enumerate(entries) if e["privacy_level"]=="high_sensitivity"]
    k = max(1, int(0.1*len(idxs)))
    for i in idxs[:k]:
        entries[i]["metadata"]["mia_test_subset"] = True
    return entries

def split_save(entries):
    random.seed(42)
    random.shuffle(entries)
    train_end = int(0.666 * len(entries))
    val_end = int(0.833 * len(entries))
    sets = [
        ("train.jsonl", entries[:train_end]),
        ("val.jsonl", entries[train_end:val_end]),
        ("test.jsonl", entries[val_end:])
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, subset in sets:
        with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf8") as f:
            for entry in subset:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
    print("已生成：", [os.path.join(OUTPUT_DIR, x[0]) for x in sets])

def main():
    data = load_data()
    entries = build_entries(data)
    split_save(entries)

if __name__ == "__main__":
    main()
