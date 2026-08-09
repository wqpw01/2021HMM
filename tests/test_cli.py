from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from .test_inputs import formal_plane_fields
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
            "--search-range",
            "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
    assert metadata["workflow_contract"] == "ramalhinho2021-formal-run/v1"
    assert metadata["mode"] == "organ_overlap_prefilter_then_vascular_cbir"
    assert metadata["parameters"] == {
        "search_range": 0,
        "k": 2,
        "organ_filter_mode": "overlap",
        "organ_match_rule": "any_overlap",
        "hmm_sigma_x": 0.6,
        "hmm_sigma_y": 0.6,
        "hmm_sigma_z": 3.0,
        "hmm_sigma_theta": 2.0,
        "hmm_window_size": 6,
        "timestamp_mode": "equal_unit_intervals",
    }
    assert metadata["input_checks"]["gallery_vector_count"] == 3
    assert metadata["input_checks"]["query_frame_count"] == 6
    assert metadata["input_checks"]["filtered_query_count"] == 6
    assert metadata["input_checks"]["query_organ_source_counts"] == {"jsonl": 6}
    assert "检索参数: K=2, r=0" in completed.stdout
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
    assert report["query_with_organs_count"] == 3
    assert report["valid_query_with_organs_count"] == 2
    assert report["gallery_organ_label_counts"] == {"liver": 2}


def test_eus_organ_source_hash_includes_tar_bytes(tmp_path: Path):
    from .test_inputs import gallery_record, write_jsonl
    from .test_organs import polygon, write_annotation_tar

    cli = __import__("ramalhinho2021.cli", fromlist=["sha256_eus_organ_sources"])
    inputs = __import__("ramalhinho2021.inputs", fromlist=["load_eus_queries"])
    module = inputs.load_registration_module(PROJECT_ROOT / "registration" / "2021.py")
    frame_id = "frame_00000001"
    frame_root = tmp_path / "eus" / frame_id
    write_jsonl(
        frame_root / f"{frame_id}_cropped_gallery.jsonl",
        [
            {
                "frame_id": frame_id,
                "status": "gallery",
                "features": gallery_record()["features"][:1],
            }
        ],
    )
    annotation = frame_root / f"{frame_id}_cropped_jpg_Label.tar"
    write_annotation_tar(annotation, [polygon(15)])
    queries = inputs.load_eus_queries(tmp_path / "eus", module)

    first_digest = cli.sha256_eus_organ_sources(queries)
    write_annotation_tar(annotation, [polygon(24)])
    second_digest = cli.sha256_eus_organ_sources(queries)

    assert len(first_digest) == 64
    assert first_digest != second_digest


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


def test_run_rejects_invalid_formal_eus_contract_before_creating_output(
    tmp_path: Path,
):
    from .test_inputs import write_jsonl

    inputs = __import__("ramalhinho2021.inputs", fromlist=["load_registration_module"])
    manifest = _write_gallery(tmp_path / "case_2" / "gallery", count=1)
    module = inputs.load_registration_module(PROJECT_ROOT / "registration" / "2021.py")
    eus_root = tmp_path / "eus"
    _make_queries(eus_root, module, count=6)
    frame_id = "frame_00000000"
    eus_manifest = eus_root / frame_id / f"{frame_id}_cropped_gallery.jsonl"
    record = json.loads(eus_manifest.read_text("utf-8"))
    record["patient_world_pose"] = True
    write_jsonl(eus_manifest, [record])
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
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "patient_world_pose 必须为 false" in completed.stderr
    assert not output_dir.exists()


def test_run_rejects_missing_ct_slice_id_before_creating_output(tmp_path: Path):
    from .test_inputs import gallery_record, write_jsonl

    inputs = __import__("ramalhinho2021.inputs", fromlist=["load_registration_module"])
    gallery = gallery_record()
    gallery.pop("slice_id")
    manifest = tmp_path / "gallery" / "gallery.jsonl"
    write_jsonl(manifest, [gallery])
    module = inputs.load_registration_module(PROJECT_ROOT / "registration" / "2021.py")
    eus_root = tmp_path / "eus"
    _make_queries(eus_root, module, count=6)
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
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "slice_id 必须是非空字符串" in completed.stderr
    assert not output_dir.exists()


def test_run_records_timestamp_file_hash(tmp_path: Path):
    inputs = __import__("ramalhinho2021.inputs", fromlist=["load_gallery_database"])
    manifest = _write_gallery(tmp_path / "gallery", count=1)
    gallery = inputs.load_gallery_database(
        manifest, PROJECT_ROOT / "registration" / "2021.py"
    )
    eus_root = tmp_path / "eus"
    _make_queries(eus_root, gallery.module, count=6)
    timestamps_csv = tmp_path / "timestamps.csv"
    timestamps_csv.write_text(
        "frame_id,timestamp_seconds\n"
        + "".join(
            f"frame_{index:08d},{index * 0.025}\n" for index in range(6)
        ),
        encoding="utf-8",
    )
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
            "--timestamps-csv",
            str(timestamps_csv),
            "--k",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
    assert metadata["timestamps_csv_sha256"] == hashlib.sha256(
        timestamps_csv.read_bytes()
    ).hexdigest()


def test_run_parser_rejects_nonfinite_sigma():
    cli = __import__("ramalhinho2021.cli", fromlist=["build_parser"])

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "run",
                "--gallery-jsonl",
                "gallery.jsonl",
                "--eus-root",
                "eus",
                "--output-dir",
                "results",
                "--sigma-x",
                "nan",
            ]
        )


def test_organ_filter_off_accepts_legacy_inputs_without_organ_metadata(
    tmp_path: Path,
):
    from .test_inputs import gallery_record, write_jsonl

    gallery = gallery_record()
    gallery.pop("organ_labels")
    manifest = tmp_path / "gallery" / "gallery.jsonl"
    write_jsonl(manifest, [gallery])
    frame_id = "frame_00000001"
    eus_root = tmp_path / "eus"
    write_jsonl(
        eus_root / frame_id / f"{frame_id}_cropped_gallery.jsonl",
        [
            {
                "frame_id": frame_id,
                "slice_id": f"{frame_id}_cropped",
                "status": "gallery",
                "features": gallery["features"],
                **formal_plane_fields(),
            }
        ],
    )
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
            "--organ-filter-mode",
            "off",
            "--k",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
    assert metadata["mode"] == "vascular_cbir_baseline"
    result = json.loads(
        (output_dir / "single_frame_results.jsonl").read_text("utf-8").splitlines()[
            0
        ]
    )
    assert result["organ_label_source"] == "unavailable"
    assert result["single_frame"]["organ_filter"]["fallback_reason"] == "disabled"
