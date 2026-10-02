#!/usr/bin/env python3
"""Generate the activity_classification.jsonl dataset for the llm-agent example.

Run from repo root:
    python examples/llm-agent/dataset/generate.py
"""
import json
import os

SAMPLES = [
    ("User is moving rapidly on foot across the track.", "Running"),
    ("User is sitting perfectly still in a chair.", "Resting"),
    ("User is pedaling a two-wheeled vehicle.", "Cycling"),
    ("User is walking at a leisurely pace.", "Walking"),
    ("User is lifting heavy dumbbells.", "Exercising"),
    ("User is horizontally positioned with eyes closed.", "Sleeping"),
    ("User is moving through water using their arms and legs.", "Swimming"),
    ("User is typing rapidly on a keyboard.", "Working"),
    ("User is chopping vegetables in the kitchen.", "Cooking"),
    ("User is steering a four-wheeled vehicle on the highway.", "Driving"),
]

OUT_PATH = os.path.join(os.path.dirname(__file__), "activity_classification.jsonl")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for q_body, answer in SAMPLES:
        record = {
            "question": f"What activity is the user performing? {q_body}",
            "answer": answer,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote {len(SAMPLES)} samples to {OUT_PATH}")