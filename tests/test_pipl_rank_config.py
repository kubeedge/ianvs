"""Regression check for the PIPL leaderboard configuration."""

from pathlib import Path

import yaml


def test_pipl_rank_receives_curated_columns():
    config_file = (
        Path(__file__).resolve().parents[1]
        / "examples/PIPL/edge-cloud_collaborative_learning_bench/benchmarkingjob.yaml"
    )
    job = yaml.safe_load(config_file.read_text(encoding="utf-8"))["benchmarkingjob"]
    rank = job["rank"]

    assert "visualization" not in job
    assert "selected_dataitem" not in job
    assert "save_mode" not in job
    assert rank["visualization"] == {
        "mode": "selected_only",
        "method": "print_table",
    }
    assert rank["save_mode"] == "selected_and_all"
    assert rank["selected_dataitem"]["modules"] == [
        "privacy_preserving_llm",
        "privacy_detection",
        "privacy_encryption",
    ]
    assert len(rank["selected_dataitem"]["metrics"]) == 9
