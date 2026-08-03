from __future__ import annotations

import csv
import json
from math import ceil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .inputs import GalleryDatabase, QueryRecord
from .pipeline import (
    CandidateResult,
    HMMFrameResult,
    HMMWindowResult,
    SingleFrameResult,
)


PANEL_SIZE = (360, 300)
CAPTION_HEIGHT = 34


def _candidate_payload(candidate: CandidateResult) -> dict[str, Any]:
    record = candidate.record
    return {
        "rank": candidate.rank,
        "distance": candidate.distance,
        "slice_id": record["slice_id"],
        "organ": record.get("organ"),
        "organ_labels": record.get("organ_labels", []),
        "features": record.get("features", []),
        "center_world": record.get("center_world"),
        "u_axis_world": record.get("u_axis_world"),
        "v_axis_world": record.get("v_axis_world"),
        "normal_world": record.get("normal_world"),
        "ct_png": record.get("ct_png"),
        "boundary_only_png": record.get("boundary_only_png"),
        "ct_overlay_png": record.get("ct_overlay_png"),
    }


def _open_or_placeholder(path: Path | None, message: str) -> Image.Image:
    if path is not None and path.is_file():
        with Image.open(path) as image:
            return image.convert("RGB")
    image = Image.new("RGB", PANEL_SIZE, "white")
    ImageDraw.Draw(image).text((12, 12), message, fill="black")
    return image


def _panel(image: Image.Image, caption: str) -> Image.Image:
    canvas = Image.new(
        "RGB", (PANEL_SIZE[0], PANEL_SIZE[1] + CAPTION_HEIGHT), "white"
    )
    image.thumbnail(PANEL_SIZE, Image.Resampling.LANCZOS)
    offset_x = (PANEL_SIZE[0] - image.width) // 2
    offset_y = (PANEL_SIZE[1] - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    ImageDraw.Draw(canvas).text((8, PANEL_SIZE[1] + 9), caption, fill="black")
    return canvas


def _render_frame_visualization(
    output_path: Path,
    gallery: GalleryDatabase,
    query: QueryRecord,
    single_frame: SingleFrameResult,
    hmm_frame: HMMFrameResult | None,
) -> None:
    query_image = _open_or_placeholder(
        query.manifest_path.parent / f"{query.frame_id}_cropped_overlay.png",
        f"Missing EUS overlay: {query.frame_id}",
    )
    single_candidate = single_frame.candidates[0] if single_frame.candidates else None
    single_image = _open_or_placeholder(
        gallery.gallery_root / single_candidate.record["ct_overlay_png"]
        if single_candidate and single_candidate.record.get("ct_overlay_png")
        else None,
        "No single-frame candidate",
    )
    hmm_candidate = hmm_frame.selected if hmm_frame else None
    hmm_image = _open_or_placeholder(
        gallery.gallery_root / hmm_candidate.record["ct_overlay_png"]
        if hmm_candidate and hmm_candidate.record.get("ct_overlay_png")
        else None,
        "No diagnostic HMM result",
    )
    captions = (
        f"EUS {query.frame_id} ({query.status})",
        "Single: none"
        if single_candidate is None
        else f"Single r{single_candidate.rank} d={single_candidate.distance:.6g}",
        "HMM: unavailable"
        if hmm_candidate is None
        else f"HMM r{hmm_candidate.rank} d={hmm_candidate.distance:.6g}",
    )
    panels = [
        _panel(image, caption)
        for image, caption in zip(
            (query_image, single_image, hmm_image), captions, strict=True
        )
    ]
    result = Image.new(
        "RGB", (sum(panel.width for panel in panels), panels[0].height), "white"
    )
    x_offset = 0
    for panel in panels:
        result.paste(panel, (x_offset, 0))
        x_offset += panel.width
    result.save(output_path)


def _write_contact_sheets(visualizations: list[Path], output_dir: Path) -> None:
    if not visualizations:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = 2
    items_per_page = 8
    for page_index, start in enumerate(
        range(0, len(visualizations), items_per_page), start=1
    ):
        page_items = visualizations[start : start + items_per_page]
        with Image.open(page_items[0]) as first:
            item_size = first.size
        rows = ceil(len(page_items) / columns)
        page = Image.new(
            "RGB", (columns * item_size[0], rows * item_size[1]), "white"
        )
        for item_index, visualization in enumerate(page_items):
            with Image.open(visualization) as image:
                x_offset = (item_index % columns) * item_size[0]
                y_offset = (item_index // columns) * item_size[1]
                page.paste(image.convert("RGB"), (x_offset, y_offset))
        page.save(output_dir / f"page_{page_index:03d}.png")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_result_bundle(
    *,
    output_dir: str | Path,
    gallery: GalleryDatabase,
    queries: list[QueryRecord],
    single_frame_results: dict[str, SingleFrameResult],
    hmm_frame_results: dict[str, HMMFrameResult],
    hmm_window_results: list[HMMWindowResult],
    metadata: dict[str, Any],
    organ_filter_mode: str = "overlap",
) -> None:
    output_dir = Path(output_dir)
    visualizations_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    visualizations: list[Path] = []
    for query in queries:
        single_frame = single_frame_results[query.frame_id]
        hmm_frame = hmm_frame_results.get(query.frame_id)
        top_k = [_candidate_payload(candidate) for candidate in single_frame.candidates]
        hmm_payload = None
        if hmm_frame is not None:
            hmm_payload = {
                "diagnostic_only": True,
                "window_index": hmm_frame.window_index,
                "local_position": hmm_frame.local_position,
                "selected": _candidate_payload(hmm_frame.selected),
            }
        hmm_status = (
            "diagnostic_only"
            if hmm_frame is not None
            else "unindexed"
            if query.feature_vector is None
            else "insufficient_contiguous_valid_frames"
        )
        result_rows.append(
            {
                "frame_id": query.frame_id,
                "numeric_frame_id": query.numeric_frame_id,
                "status": query.status,
                "query_features": query.record.get("features", []),
                "query_organ_labels": list(query.organ_labels),
                "organ_label_source": query.organ_label_source,
                "single_frame": {
                    "retrieval_status": single_frame.retrieval_status,
                    "candidate_count": len(top_k),
                    "organ_filter": {
                        "mode": organ_filter_mode,
                        "match_rule": "any_overlap"
                        if organ_filter_mode == "overlap"
                        else None,
                        "filter_applied": single_frame.filter_applied,
                        "gallery_count_before": single_frame.gallery_count_before,
                        "eligible_gallery_count": single_frame.eligible_gallery_count,
                        "fallback_reason": single_frame.fallback_reason,
                    },
                    "top_k": top_k,
                },
                "hmm_status": hmm_status,
                "hmm_diagnostic": hmm_payload,
            }
        )
        top_one = top_k[0] if top_k else {}
        hmm_selected = hmm_payload["selected"] if hmm_payload else {}
        summary_rows.append(
            {
                "frame_id": query.frame_id,
                "status": query.status,
                "feature_count": len(query.record.get("features", [])),
                "query_organ_labels": "+".join(query.organ_labels),
                "organ_label_source": query.organ_label_source,
                "organ_filter_applied": single_frame.filter_applied,
                "eligible_gallery_count": single_frame.eligible_gallery_count,
                "organ_filter_fallback_reason": single_frame.fallback_reason or "",
                "retrieval_status": single_frame.retrieval_status,
                "single_candidate_count": len(top_k),
                "single_top1_slice_id": top_one.get("slice_id", ""),
                "single_top1_distance": top_one.get("distance", ""),
                "hmm_status": hmm_status,
                "hmm_window_index": hmm_payload["window_index"]
                if hmm_payload
                else "",
                "hmm_slice_id": hmm_selected.get("slice_id", ""),
                "hmm_rank": hmm_selected.get("rank", ""),
                "hmm_distance": hmm_selected.get("distance", ""),
            }
        )
        visualization = visualizations_dir / f"{query.frame_id}.png"
        _render_frame_visualization(
            visualization, gallery, query, single_frame, hmm_frame
        )
        visualizations.append(visualization)

    _write_jsonl(output_dir / "single_frame_results.jsonl", result_rows)
    with (output_dir / "single_frame_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(summary_rows[0]) if summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    window_rows = [
        {
            "diagnostic_only": True,
            "window_index": window.window_index,
            "frame_ids": window.frame_ids,
            "timestamps": window.timestamps,
            "selected": [_candidate_payload(candidate) for candidate in window.selected],
            "transition_costs": window.transition_costs,
        }
        for window in hmm_window_results
    ]
    _write_jsonl(output_dir / "hmm_diagnostic_windows.jsonl", window_rows)
    run_metadata = dict(metadata)
    run_metadata.update(
        {
            "diagnostic_only": True,
            "query_frame_count": len(queries),
            "visualization_count": len(visualizations),
            "hmm_window_count": len(hmm_window_results),
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Ramalhinho 2021 检索结果\n\n"
        "本目录包含器官预筛选后的单帧血管 CBIR 结果和诊断性 HMM 路径。"
        "器官标签只用于缩小 CT 候选范围，不参与血管距离或 HMM 转移代价。"
        "EUS 输入没有患者世界坐标或配准真值，因此不报告 TRE 或临床成功率。\n",
        encoding="utf-8",
    )
    _write_contact_sheets(visualizations, output_dir / "contact_sheets")
