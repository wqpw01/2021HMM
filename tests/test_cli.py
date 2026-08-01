from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .test_pipeline import _make_queries, _write_gallery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "run_reproduction.py"


def test_run_command_executes_synthetic_end_to_end(tmp_path: Path):
    inputs = __import__("ramalhinho2021.inputs", fromlist=["load_gallery_database"])
    manifest = _write_gallery(tmp_path / "case_2" / "gallery", count=3)
    gallery = inputs.load_gallery_database(
        manifest, PROJECT_ROOT / "registration" / "2021.py"
    )
    eus_root = tmp_path / "eus"
    _make_queries(eus_root, gallery.module, count=6)
    output_dir = tmp_path / "results"

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "run",
            "--gallery-jsonl",
            str(manifest),
            "--eus-root",
            str(eus_root),
            "--output-dir",
            str(output_dir),
            "--k",
            "2",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
    assert metadata["parameters"] == {
        "search_range": 2,
        "k": 2,
        "hmm_sigma_x": 0.6,
        "hmm_sigma_y": 0.6,
        "hmm_sigma_z": 3.0,
        "hmm_sigma_theta": 2.0,
        "hmm_window_size": 6,
        "timestamp_mode": "equal_unit_intervals",
    }
    assert metadata["input_checks"]["gallery_vector_count"] == 3
    assert metadata["input_checks"]["query_frame_count"] == 6
    assert "完成" in completed.stdout


def test_validate_reports_counts_without_creating_results(tmp_path: Path):
    inputs = __import__("ramalhinho2021.inputs", fromlist=["load_gallery_database"])
    manifest = _write_gallery(tmp_path / "case_2" / "gallery", count=2)
    gallery = inputs.load_gallery_database(
        manifest, PROJECT_ROOT / "registration" / "2021.py"
    )
    eus_root = tmp_path / "eus"
    _make_queries(eus_root, gallery.module, count=3, unindexed_index=1)

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "validate",
            "--gallery-jsonl",
            str(manifest),
            "--eus-root",
            str(eus_root),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["gallery_vector_count"] == 2
    assert report["query_frame_count"] == 3
    assert report["valid_query_count"] == 2
    assert report["unindexed_query_count"] == 1


def test_missing_gallery_fails_before_creating_output(tmp_path: Path):
    eus_root = tmp_path / "eus"
    eus_root.mkdir()
    output_dir = tmp_path / "results"

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "run",
            "--gallery-jsonl",
            str(tmp_path / "missing" / "gallery.jsonl"),
            "--eus-root",
            str(eus_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "gallery.jsonl 不存在" in completed.stderr
    assert not output_dir.exists()
