from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

from PIL import Image

from .test_pipeline import _make_queries, _write_gallery


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_write_result_bundle_creates_auditable_numeric_and_visual_outputs(
    tmp_path: Path,
):
    inputs = importlib.import_module("ramalhinho2021.inputs")
    pipeline = importlib.import_module("ramalhinho2021.pipeline")
    outputs = importlib.import_module("ramalhinho2021.outputs")
    gallery = inputs.load_gallery_database(
        _write_gallery(tmp_path / "case_2" / "gallery", count=3),
        PROJECT_ROOT / "registration" / "2021.py",
    )
    _write_gallery_images(gallery)
    queries = _make_queries(tmp_path / "eus", gallery.module, count=6)
    _write_eus_images(queries)
    windows, assignments = pipeline.build_hmm_window_assignments(queries)
    single_results = pipeline.run_single_frame_retrieval(
        gallery, queries, k=2, search_range=2
    )
    hmm_frames, hmm_windows = pipeline.run_hmm_diagnostics(
        gallery.module, windows, assignments, single_results
    )
    output_dir = tmp_path / "results"

    outputs.write_result_bundle(
        output_dir=output_dir,
        gallery=gallery,
        queries=queries,
        single_frame_results=single_results,
        hmm_frame_results=hmm_frames,
        hmm_window_results=hmm_windows,
        metadata={"parameters": {"k": 2, "hmm_window_size": 6}},
    )

    expected = {
        "single_frame_results.jsonl",
        "single_frame_summary.csv",
        "hmm_diagnostic_windows.jsonl",
        "run_metadata.json",
        "README.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    result_rows = _read_jsonl(output_dir / "single_frame_results.jsonl")
    assert len(result_rows) == 6
    assert result_rows[0]["single_frame"]["candidate_count"] == 2
    assert result_rows[0]["single_frame"]["top_k"][0]["slice_id"]
    assert result_rows[0]["hmm_status"] == "diagnostic_only"
    assert result_rows[0]["hmm_diagnostic"]["selected"]["rank"] in (1, 2)
    window_rows = _read_jsonl(output_dir / "hmm_diagnostic_windows.jsonl")
    assert len(window_rows) == 1
    assert len(window_rows[0]["frame_ids"]) == 6
    assert len(window_rows[0]["transition_costs"]) == 5
    metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
    assert metadata["query_frame_count"] == 6
    assert metadata["hmm_window_count"] == 1
    assert metadata["visualization_count"] == 6
    assert len(list((output_dir / "visualizations").glob("*.png"))) == 6
    assert len(list((output_dir / "contact_sheets").glob("*.png"))) == 1
    with (output_dir / "single_frame_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 6


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _write_gallery_images(gallery) -> None:
    for record in gallery.records_by_pose_id.values():
        path = gallery.gallery_root / record["ct_overlay_png"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (80, 100, 120)).save(path)


def _write_eus_images(queries) -> None:
    for query in queries:
        path = query.manifest_path.parent / f"{query.frame_id}_cropped_overlay.png"
        Image.new("RGB", (64, 64), (120, 80, 100)).save(path)
