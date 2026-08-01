from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_MODULE = PROJECT_ROOT / "registration" / "2021.py"


def gallery_record(slice_id: str = "slice-001") -> dict:
    return {
        "frame_id": "case_2",
        "slice_id": slice_id,
        "status": "gallery",
        "organ": "liver",
        "center_world": [10.0, 20.0, 30.0],
        "u_axis_world": [1.0, 0.0, 0.0],
        "v_axis_world": [0.0, 1.0, 0.0],
        "normal_world": [0.0, 0.0, 1.0],
        "ct_png": f"ct/{slice_id}.png",
        "boundary_only_png": f"boundary_only/{slice_id}.png",
        "ct_overlay_png": f"ct_overlay/{slice_id}.png",
        "features": [
            {"label": "artery", "x_mm": 12.0, "y_mm": 18.0, "area_mm2": 3.5},
            {"label": "vein", "x_mm": 30.0, "y_mm": 40.0, "area_mm2": 5.0},
        ],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_gallery_database_recovers_features_pose_and_source_record(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    manifest = tmp_path / "case_2" / "gallery" / "gallery.jsonl"
    record = gallery_record()
    write_jsonl(manifest, [record])

    gallery = inputs.load_gallery_database(manifest, REGISTRATION_MODULE)

    assert len(gallery.features) == 1
    vector = gallery.features[0]
    assert [triplet.label for triplet in vector.triplets] == ["artery", "vein"]
    np.testing.assert_allclose(vector.pose.surface_point, [10.0, 20.0, 30.0])
    assert gallery.database.keys() == {"artery:1_vein:1"}
    assert gallery.records_by_pose_id[id(vector.pose)] == record
    assert gallery.gallery_root == manifest.parent
    assert gallery.max_rotation_error < 1e-12


def test_load_gallery_database_reports_manifest_line_for_invalid_basis(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    manifest = tmp_path / "gallery.jsonl"
    record = gallery_record()
    record["v_axis_world"] = [1.0, 0.0, 0.0]
    write_jsonl(manifest, [record])

    with pytest.raises(ValueError, match=r"gallery\.jsonl 第 1 行.*正交"):
        inputs.load_gallery_database(manifest, REGISTRATION_MODULE)


def test_load_eus_queries_sorts_numeric_ids_and_preserves_unindexed(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_one_gallery(tmp_path / "gallery"), REGISTRATION_MODULE
    )
    eus_root = tmp_path / "eus"
    for frame_number, status, features in (
        (10, "gallery", gallery_record()["features"][:1]),
        (2, "unindexed", []),
        (3, "gallery", gallery_record()["features"][1:]),
    ):
        frame_id = f"frame_{frame_number:08d}"
        record = {
            "frame_id": frame_id,
            "slice_id": f"{frame_id}_cropped",
            "status": status,
            "features": features,
            "patient_world_pose": False,
        }
        write_jsonl(
            eus_root / frame_id / f"{frame_id}_cropped_gallery.jsonl", [record]
        )

    queries = inputs.load_eus_queries(eus_root, gallery.module)

    assert [query.numeric_frame_id for query in queries] == [2, 3, 10]
    assert queries[0].feature_vector is None
    assert queries[0].status == "unindexed"
    assert queries[1].feature_vector.triplets[0].label == "vein"


def test_timestamp_csv_requires_all_requested_frames_and_increasing_values(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    timestamp_csv = tmp_path / "timestamps.csv"
    with timestamp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "timestamp_seconds"])
        writer.writeheader()
        writer.writerow({"frame_id": "frame_00000001", "timestamp_seconds": "0.025"})
        writer.writerow({"frame_id": "frame_00000002", "timestamp_seconds": "0.050"})

    timestamps = inputs.load_timestamps_csv(timestamp_csv)

    assert timestamps == {"frame_00000001": 0.025, "frame_00000002": 0.05}
    assert inputs.timestamps_for_frames(
        ["frame_00000001", "frame_00000002"], timestamps
    ) == [0.025, 0.05]
    with pytest.raises(ValueError, match="缺少时间戳"):
        inputs.timestamps_for_frames(["frame_00000003"], timestamps)
    with pytest.raises(ValueError, match="严格递增"):
        inputs.timestamps_for_frames(
            ["frame_00000002", "frame_00000001"], timestamps
        )


def _write_one_gallery(root: Path) -> Path:
    manifest = root / "gallery.jsonl"
    write_jsonl(manifest, [gallery_record()])
    return manifest
