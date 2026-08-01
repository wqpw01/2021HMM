from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .inputs import (
    GalleryDatabase,
    QueryRecord,
    load_eus_queries,
    load_gallery_database,
    load_registration_module,
    load_timestamps_csv,
)
from .outputs import write_result_bundle
from .pipeline import (
    build_hmm_window_assignments,
    run_hmm_diagnostics,
    run_single_frame_retrieval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRATION_MODULE = PROJECT_ROOT / "registration" / "2021.py"
ORIGINAL_REGISTRATION_SHA256 = (
    "cd60f299d30d8cb9cfdf63820ed4b092e45df67ec1a7bcc69bdf960b43e1171b"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_eus_manifests(queries: list[QueryRecord]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.frame_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(query.manifest_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def summarize_eus(queries: list[QueryRecord]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    valid_count = 0
    for query in queries:
        if query.feature_vector is None:
            continue
        valid_count += 1
        label_counts.update(triplet.label for triplet in query.feature_vector.triplets)
    return {
        "query_frame_count": len(queries),
        "valid_query_count": valid_count,
        "unindexed_query_count": len(queries) - valid_count,
        "query_label_counts": dict(sorted(label_counts.items())),
    }


def summarize_inputs(
    gallery: GalleryDatabase,
    queries: list[QueryRecord],
    hmm_window_count: int,
) -> dict[str, Any]:
    gallery_label_counts: Counter[str] = Counter()
    for feature_vector in gallery.features:
        gallery_label_counts.update(
            triplet.label for triplet in feature_vector.triplets
        )
    return {
        "gallery_vector_count": len(gallery.features),
        "gallery_database_key_count": len(gallery.database),
        "gallery_label_counts": dict(sorted(gallery_label_counts.items())),
        **summarize_eus(queries),
        "hmm_window_count": hmm_window_count,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能为负数")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def _add_registration_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registration-module",
        type=Path,
        default=DEFAULT_REGISTRATION_MODULE,
        help="修正后的 2021.py 路径",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ramalhinho 2021 多标签 CBIR 与 HMM 本地复现"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_eus = subparsers.add_parser(
        "validate-eus", help="只验证 EUS 特征目录"
    )
    validate_eus.add_argument("--eus-root", type=Path, required=True)
    _add_registration_argument(validate_eus)

    validate = subparsers.add_parser("validate", help="验证图库和 EUS 输入")
    validate.add_argument("--gallery-jsonl", type=Path, required=True)
    validate.add_argument("--eus-root", type=Path, required=True)
    validate.add_argument("--hmm-window-size", type=_positive_int, default=6)
    _add_registration_argument(validate)

    run = subparsers.add_parser("run", help="执行单帧 CBIR 和多帧 HMM")
    run.add_argument("--gallery-jsonl", type=Path, required=True)
    run.add_argument("--eus-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--timestamps-csv", type=Path)
    run.add_argument("--k", type=_positive_int, default=200)
    run.add_argument("--search-range", type=_nonnegative_int, default=2)
    run.add_argument("--hmm-window-size", type=_positive_int, default=6)
    run.add_argument("--sigma-x", type=_positive_float, default=0.6)
    run.add_argument("--sigma-y", type=_positive_float, default=0.6)
    run.add_argument("--sigma-z", type=_positive_float, default=3.0)
    run.add_argument("--sigma-theta", type=_positive_float, default=2.0)
    _add_registration_argument(run)
    return parser


def _validate_eus_command(args: argparse.Namespace) -> int:
    module = load_registration_module(args.registration_module)
    queries = load_eus_queries(args.eus_root, module)
    print(json.dumps(summarize_eus(queries), ensure_ascii=False, indent=2))
    return 0


def _load_inputs(args: argparse.Namespace) -> tuple[GalleryDatabase, list[QueryRecord]]:
    gallery = load_gallery_database(args.gallery_jsonl, args.registration_module)
    queries = load_eus_queries(args.eus_root, gallery.module)
    return gallery, queries


def _validate_command(args: argparse.Namespace) -> int:
    gallery, queries = _load_inputs(args)
    windows, _ = build_hmm_window_assignments(
        queries, window_size=args.hmm_window_size
    )
    print(
        json.dumps(
            summarize_inputs(gallery, queries, len(windows)),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    if args.hmm_window_size < 2:
        raise ValueError("HMM 窗口大小必须至少为 2")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录不是空目录: {output_dir}")

    gallery, queries = _load_inputs(args)
    timestamps_by_frame = (
        load_timestamps_csv(args.timestamps_csv) if args.timestamps_csv else None
    )
    windows, assignments = build_hmm_window_assignments(
        queries, window_size=args.hmm_window_size
    )
    print(
        f"已加载 {len(gallery.features)} 个 CT 候选和 {len(queries)} 个 EUS 帧。",
        flush=True,
    )
    single_results = run_single_frame_retrieval(
        gallery, queries, k=args.k, search_range=args.search_range
    )
    hmm_frame_results, hmm_window_results = run_hmm_diagnostics(
        gallery.module,
        windows,
        assignments,
        single_results,
        sigma_x=args.sigma_x,
        sigma_y=args.sigma_y,
        sigma_z=args.sigma_z,
        sigma_theta=args.sigma_theta,
        timestamps_by_frame=timestamps_by_frame,
    )
    input_checks = summarize_inputs(gallery, queries, len(windows))
    indexed_results = [
        result
        for result in single_results.values()
        if result.query.feature_vector is not None
    ]
    candidate_shortfalls = sum(
        len(result.candidates) < args.k for result in indexed_results
    )
    timestamp_mode = (
        "timestamps_csv" if timestamps_by_frame is not None else "equal_unit_intervals"
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "paper_defaults",
        "registration_module": str(Path(args.registration_module).resolve()),
        "registration_module_sha256": sha256_file(args.registration_module),
        "original_registration_module_sha256": ORIGINAL_REGISTRATION_SHA256,
        "gallery_jsonl": str(Path(args.gallery_jsonl).resolve()),
        "gallery_jsonl_sha256": sha256_file(args.gallery_jsonl),
        "eus_root": str(Path(args.eus_root).resolve()),
        "eus_manifests_sha256": sha256_eus_manifests(queries),
        "timestamps_csv": str(Path(args.timestamps_csv).resolve())
        if args.timestamps_csv
        else None,
        "parameters": {
            "search_range": args.search_range,
            "k": args.k,
            "hmm_sigma_x": args.sigma_x,
            "hmm_sigma_y": args.sigma_y,
            "hmm_sigma_z": args.sigma_z,
            "hmm_sigma_theta": args.sigma_theta,
            "hmm_window_size": args.hmm_window_size,
            "timestamp_mode": timestamp_mode,
        },
        "assumptions": [
            "EUS frames are ordered by ascending numeric frame ID.",
            "Unindexed EUS frames split contiguous HMM runs.",
            "EUS patient_world_pose is false, so HMM output is diagnostic only.",
            "No TRE or clinical success rate is reported without ground truth.",
        ],
        "input_checks": input_checks,
        "candidate_shortfall_frame_count": candidate_shortfalls,
        "max_pose_rotation_reconstruction_error": gallery.max_rotation_error,
    }
    write_result_bundle(
        output_dir=output_dir,
        gallery=gallery,
        queries=queries,
        single_frame_results=single_results,
        hmm_frame_results=hmm_frame_results,
        hmm_window_results=hmm_window_results,
        metadata=metadata,
    )
    print(f"完成。结果目录: {output_dir}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-eus":
            return _validate_eus_command(args)
        if args.command == "validate":
            return _validate_command(args)
        return _run_command(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 2
