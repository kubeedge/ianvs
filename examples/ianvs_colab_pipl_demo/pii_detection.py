import re
from typing import List, Dict

def detect_pii(text: str) -> List[Dict]:
    """Detect PII in plain text using regex (sample: phone & email)."""
    pii_patterns = [
        (r'\b\d{11}\b', 'PHONE'),
        (r'\b[\w.-]+@[\w.-]+\.\w+\b', 'EMAIL')
    ]
    findings = []
    for pat, tag in pii_patterns:
        for m in re.finditer(pat, text):
            findings.append({"start": m.start(), "end": m.end(), "label": tag, "text": m.group()})
    return findings

def mask_pii(text: str, findings: List[Dict]) -> str:
    """Mask all detected PII entities in the text."""
    spans = sorted([(f["start"], f["end"]) for f in findings], reverse=True)
    text_out = text
    for start, end in spans:
        text_out = text_out[:start] + '[MASKED]' + text_out[end:]
    return text_out

# Example usage
if __name__ == "__main__":
    sample = "Contact me at 13912345678 or demo@example.com."
    found = detect_pii(sample)
    print("Detected:", found)
    print("Masked:", mask_pii(sample, found))
