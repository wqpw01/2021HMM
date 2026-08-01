from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

from .test_inputs import gallery_record, write_jsonl


def test_build_hmm_windows_uses_six_frames_and_tail_anchors(tmp_path: Path):
    pipeline = importlib.import_module("ramalhinho2021.pipeline")
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_gallery(tmp_path / "gallery", count=1),
        Path(__file__).resolve().parents[1] / "registration" / "2021.py",
    )
    queries = _make_queries(tmp_path / "eus", gallery.module, count=8)

    windows, assignments = pipeline.build_hmm_window_assignments(
        queries, window_size=6
    )

    assert len(windows) == 3
    assert windows[0].frame_ids == tuple(f"frame_{i:08d}" for i in range(6))
    assert windows[-1].frame_ids == tuple(f"frame_{i:08d}" for i in range(2, 8))
    assert assignments["frame_00000000"].local_position == 0
    assert assignments["frame_00000001"].window_index == 1
    assert assignments["frame_00000002"].local_position == 0
    assert assignments["frame_00000007"].local_position == 5


def test_unindexed_frame_splits_hmm_runs(tmp_path: Path):
    pipeline = importlib.import_module("ramalhinho2021.pipeline")
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_gallery(tmp_path / "gallery", count=1),
        Path(__file__).resolve().parents[1] / "registration" / "2021.py",
    )
    queries = _make_queries(
        tmp_path / "eus", gallery.module, count=8, unindexed_index=3
    )

    windows, assignments = pipeline.build_hmm_window_assignments(
        queries, window_size=3
    )

    assert len(windows) == 3
    assert "frame_00000003" not in assignments
    for window in windows:
        numeric_ids = [query.numeric_frame_id for query in window.queries]
        assert max(numeric_ids) < 3 or min(numeric_ids) > 3


def test_single_frame_retrieval_keeps_sorted_candidates_and_records(tmp_path: Path):
    pipeline = importlib.import_module("ramalhinho2021.pipeline")
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery_manifest = _write_gallery(tmp_path / "gallery", count=3)
    gallery = inputs.load_gallery_database(
        gallery_manifest,
        Path(__file__).resolve().parents[1] / "registration" / "2021.py",
    )
    queries = _make_queries(tmp_path / "eus", gallery.module, count=1)

    results = pipeline.run_single_frame_retrieval(
        gallery, queries, k=2, search_range=2
    )

    result = results["frame_00000000"]
    assert len(result.candidates) == 2
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    assert result.candidates[0].distance <= result.candidates[1].distance
    assert result.candidates[0].record["slice_id"].startswith("slice-")
    assert result.retrieval_result is not None


def test_run_hmm_diagnostics_returns_complete_six_frame_path(tmp_path: Path):
    pipeline = importlib.import_module("ramalhinho2021.pipeline")
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_gallery(tmp_path / "gallery", count=3),
        Path(__file__).resolve().parents[1] / "registration" / "2021.py",
    )
    queries = _make_queries(tmp_path / "eus", gallery.module, count=6)
    windows, assignments = pipeline.build_hmm_window_assignments(queries)
    single_results = pipeline.run_single_frame_retrieval(
        gallery, queries, k=2, search_range=2
    )
    timestamps = {
        query.frame_id: index * 0.025 for index, query in enumerate(queries)
    }

    frame_results, window_results = pipeline.run_hmm_diagnostics(
        gallery.module,
        windows,
        assignments,
        single_results,
        timestamps_by_frame=timestamps,
    )

    assert len(frame_results) == 6
    assert len(window_results) == 1
    window = window_results[0]
    assert window.timestamps == tuple(index * 0.025 for index in range(6))
    assert len(window.selected) == 6
    assert len(window.transition_costs) == 5
    assert all(math.isfinite(cost) for cost in window.transition_costs)
    for frame_id, hmm_result in frame_results.items():
        candidate_ranks = {
            candidate.rank for candidate in single_results[frame_id].candidates
        }
        assert hmm_result.selected.rank in candidate_ranks


def test_run_hmm_diagnostics_defaults_to_equal_unit_intervals(tmp_path: Path):
    pipeline = importlib.import_module("ramalhinho2021.pipeline")
    inputs = importlib.import_module("ramalhinho2021.inputs")
    gallery = inputs.load_gallery_database(
        _write_gallery(tmp_path / "gallery", count=2),
        Path(__file__).resolve().parents[1] / "registration" / "2021.py",
    )
    queries = _make_queries(tmp_path / "eus", gallery.module, count=6)
    windows, assignments = pipeline.build_hmm_window_assignments(queries)
    single_results = pipeline.run_single_frame_retrieval(
        gallery, queries, k=2, search_range=2
    )

    _, window_results = pipeline.run_hmm_diagnostics(
        gallery.module, windows, assignments, single_results
    )

    assert window_results[0].timestamps == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def _write_gallery(root: Path, count: int) -> Path:
    records = []
    for index in range(count):
        record = gallery_record(f"slice-{index:03d}")
        for feature in record["features"]:
            feature["x_mm"] += index * 0.2
        records.append(record)
    manifest = root / "gallery.jsonl"
    write_jsonl(manifest, records)
    return manifest


def _make_queries(
    root: Path, module, count: int, unindexed_index: int | None = None
):
    eus_root = root
    for index in range(count):
        frame_id = f"frame_{index:08d}"
        if index == unindexed_index:
            status = "unindexed"
            features = []
        else:
            status = "gallery"
            features = gallery_record()["features"][:1]
        write_jsonl(
            eus_root / frame_id / f"{frame_id}_cropped_gallery.jsonl",
            [
                {
                    "frame_id": frame_id,
                    "slice_id": f"{frame_id}_cropped",
                    "status": status,
                    "features": features,
                }
            ],
        )
    return importlib.import_module("ramalhinho2021.inputs").load_eus_queries(
        eus_root, module
    )
