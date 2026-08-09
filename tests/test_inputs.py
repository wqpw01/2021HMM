from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from .test_organs import polygon, write_annotation_tar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_MODULE = PROJECT_ROOT / "registration" / "2021.py"


def formal_plane_fields() -> dict:
    return {
        "width_mm": 100.0,
        "length_mm": 100.0,
        "pixel_spacing_mm": [100.0 / 959.0, 100.0 / 959.0],
        "pose_coordinate_system": "synthetic_2d_10cm_crop",
        "patient_world_pose": False,
    }


def gallery_record(slice_id: str = "slice-001") -> dict:
    return {
        "frame_id": "case_2",
        "slice_id": slice_id,
        "status": "gallery",
        "organ": "liver",
        "organ_labels": ["liver"],
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
        **formal_plane_fields(),
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
    assert gallery.organ_labels_by_pose_id[id(vector.pose)] == ("liver",)
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
            "organ_labels": ["liver"],
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
    assert queries[1].organ_labels == ("liver",)
    assert queries[1].organ_label_source == "jsonl"


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


def test_load_eus_queries_falls_back_to_active_organ_polygons_in_tar(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_one_gallery(tmp_path / "gallery"), REGISTRATION_MODULE
    )
    frame_id = "frame_00000001"
    frame_root = tmp_path / "eus" / frame_id
    record = {
        "frame_id": frame_id,
        "slice_id": f"{frame_id}_cropped",
        "status": "gallery",
        "features": gallery_record()["features"][:1],
        **formal_plane_fields(),
    }
    write_jsonl(frame_root / f"{frame_id}_cropped_gallery.jsonl", [record])
    write_annotation_tar(
        frame_root / f"{frame_id}_cropped_jpg_Label.tar",
        [polygon(15), polygon(24)],
    )

    query = inputs.load_eus_queries(
        tmp_path / "eus", gallery.module, require_formal_contract=True
    )[0]

    assert query.organ_labels == ("gallbladder", "kidney_left")
    assert query.organ_label_source == "tar_active_polygons"
    assert query.organ_label_source_path == (
        frame_root / f"{frame_id}_cropped_jpg_Label.tar"
    ).resolve()


def test_explicit_eus_organ_labels_take_precedence_over_tar(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_one_gallery(tmp_path / "gallery"), REGISTRATION_MODULE
    )
    frame_id = "frame_00000001"
    frame_root = tmp_path / "eus" / frame_id
    write_jsonl(
        frame_root / f"{frame_id}_cropped_gallery.jsonl",
        [
            {
                "frame_id": frame_id,
                "slice_id": f"{frame_id}_cropped",
                "status": "gallery",
                "features": gallery_record()["features"][:1],
                "organ_labels": ["spleen"],
                **formal_plane_fields(),
            }
        ],
    )
    write_annotation_tar(
        frame_root / f"{frame_id}_cropped_jpg_Label.tar", [polygon(15)]
    )

    query = inputs.load_eus_queries(
        tmp_path / "eus", gallery.module, require_formal_contract=True
    )[0]

    assert query.organ_labels == ("spleen",)
    assert query.organ_label_source == "jsonl"


def test_eus_organ_metadata_is_required_when_filtering_is_enabled(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_one_gallery(tmp_path / "gallery"), REGISTRATION_MODULE
    )
    frame_id = "frame_00000001"
    write_jsonl(
        tmp_path / "eus" / frame_id / f"{frame_id}_cropped_gallery.jsonl",
        [
            {
                "frame_id": frame_id,
                "status": "gallery",
                "features": gallery_record()["features"][:1],
            }
        ],
    )

    with pytest.raises(ValueError, match="缺少 organ_labels.*TAR 不存在"):
        inputs.load_eus_queries(tmp_path / "eus", gallery.module)

    query = inputs.load_eus_queries(
        tmp_path / "eus", gallery.module, require_organ_labels=False
    )[0]
    assert query.organ_labels == ()
    assert query.organ_label_source == "unavailable"


def test_formal_contract_accepts_shared_100mm_ct_and_eus_records(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery_record_value = gallery_record()
    gallery_record_value.update(formal_plane_fields())
    gallery_manifest = tmp_path / "gallery" / "gallery.jsonl"
    write_jsonl(gallery_manifest, [gallery_record_value])
    gallery = inputs.load_gallery_database(
        gallery_manifest,
        REGISTRATION_MODULE,
        require_formal_contract=True,
    )

    frame_id = "frame_00000001"
    query_record = {
        "frame_id": frame_id,
        "slice_id": f"{frame_id}_cropped",
        "status": "gallery",
        "features": gallery_record_value["features"][:1],
        "organ_labels": ["liver"],
        **formal_plane_fields(),
    }
    write_jsonl(
        tmp_path / "eus" / frame_id / f"{frame_id}_cropped_gallery.jsonl",
        [query_record],
    )

    queries = inputs.load_eus_queries(
        tmp_path / "eus",
        gallery.module,
        require_formal_contract=True,
    )

    assert queries[0].feature_vector is not None


def test_formal_contract_rejects_patient_world_eus_pose(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_one_gallery(tmp_path / "gallery"), REGISTRATION_MODULE
    )
    frame_id = "frame_00000001"
    record = {
        "frame_id": frame_id,
        "slice_id": f"{frame_id}_cropped",
        "status": "gallery",
        "features": gallery_record()["features"][:1],
        "organ_labels": ["liver"],
        **formal_plane_fields(),
        "patient_world_pose": True,
    }
    write_jsonl(
        tmp_path / "eus" / frame_id / f"{frame_id}_cropped_gallery.jsonl",
        [record],
    )

    with pytest.raises(ValueError, match="patient_world_pose 必须为 false"):
        inputs.load_eus_queries(
            tmp_path / "eus",
            gallery.module,
            require_formal_contract=True,
        )


def test_formal_contract_rejects_inconsistent_status_and_features(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_one_gallery(tmp_path / "gallery"), REGISTRATION_MODULE
    )
    frame_id = "frame_00000001"
    record = {
        "frame_id": frame_id,
        "slice_id": f"{frame_id}_cropped",
        "status": "unindexed",
        "features": gallery_record()["features"][:1],
        "organ_labels": ["liver"],
        **formal_plane_fields(),
    }
    write_jsonl(
        tmp_path / "eus" / frame_id / f"{frame_id}_cropped_gallery.jsonl",
        [record],
    )

    with pytest.raises(ValueError, match="status=unindexed 时 features 必须为空"):
        inputs.load_eus_queries(
            tmp_path / "eus",
            gallery.module,
            require_formal_contract=True,
        )


def test_formal_contract_rejects_out_of_plane_eus_centroid(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_one_gallery(tmp_path / "gallery"), REGISTRATION_MODULE
    )
    frame_id = "frame_00000001"
    feature = dict(gallery_record()["features"][0])
    feature["x_mm"] = 100.1
    record = {
        "frame_id": frame_id,
        "slice_id": f"{frame_id}_cropped",
        "status": "gallery",
        "features": [feature],
        "organ_labels": ["liver"],
        **formal_plane_fields(),
    }
    write_jsonl(
        tmp_path / "eus" / frame_id / f"{frame_id}_cropped_gallery.jsonl",
        [record],
    )

    with pytest.raises(ValueError, match="x_mm/y_mm 必须位于 100 mm 平面内"):
        inputs.load_eus_queries(
            tmp_path / "eus",
            gallery.module,
            require_formal_contract=True,
        )


def test_formal_contract_rejects_non_100mm_ct_plane(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    record = gallery_record()
    record.update(formal_plane_fields())
    record["width_mm"] = 80.0
    manifest = tmp_path / "gallery.jsonl"
    write_jsonl(manifest, [record])

    with pytest.raises(ValueError, match=r"gallery\.jsonl.*100 mm"):
        inputs.load_gallery_database(
            manifest,
            REGISTRATION_MODULE,
            require_formal_contract=True,
        )


def test_formal_contract_rejects_missing_ct_slice_id(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    record = gallery_record()
    record.pop("slice_id")
    manifest = tmp_path / "gallery.jsonl"
    write_jsonl(manifest, [record])

    with pytest.raises(ValueError, match="slice_id 必须是非空字符串"):
        inputs.load_gallery_database(
            manifest,
            REGISTRATION_MODULE,
            require_formal_contract=True,
        )


def test_formal_contract_rejects_non_object_ct_record(tmp_path: Path):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    manifest = tmp_path / "gallery.jsonl"
    write_jsonl(manifest, [[]])

    with pytest.raises(ValueError, match="记录必须是 JSON 对象"):
        inputs.load_gallery_database(
            manifest,
            REGISTRATION_MODULE,
            require_organ_labels=False,
            require_formal_contract=True,
        )


def _write_one_gallery(root: Path) -> Path:
    manifest = root / "gallery.jsonl"
    write_jsonl(manifest, [gallery_record()])
    return manifest
