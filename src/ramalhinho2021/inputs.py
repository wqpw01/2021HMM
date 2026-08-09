from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np

from .organs import load_organ_labels_from_tar, normalize_organ_labels


FORMAL_PLANE_WIDTH_MM = 100.0
FORMAL_PLANE_LENGTH_MM = 100.0
FORMAL_EUS_COORDINATE_SYSTEM = "synthetic_2d_10cm_crop"
FORMAL_VESSEL_LABELS = frozenset({"artery", "vein"})


@dataclass(frozen=True)
class GalleryDatabase:
    module: ModuleType
    database: dict[str, list[Any]]
    features: list[Any]
    records_by_pose_id: dict[int, dict[str, Any]]
    organ_labels_by_pose_id: dict[int, tuple[str, ...]]
    gallery_root: Path
    manifest_path: Path
    max_rotation_error: float

    def create_cbir(
        self,
        search_range: int = 2,
        database: dict[str, list[Any]] | None = None,
    ):
        return self.module.MultiLabelledCBIR(
            database=self.database if database is None else database,
            search_range=search_range,
        )

    def create_hmm(self, **kwargs):
        return self.module.HMMPoseEstimator(**kwargs)


@dataclass(frozen=True)
class QueryRecord:
    numeric_frame_id: int
    frame_id: str
    status: str
    feature_vector: Any | None
    organ_labels: tuple[str, ...]
    organ_label_source: str
    organ_label_source_path: Path | None
    record: dict[str, Any]
    manifest_path: Path


def load_registration_module(registration_module_path: str | Path) -> ModuleType:
    source = Path(registration_module_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"2021.py 模块不存在: {source}")
    module_name = f"ramalhinho_2021_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 2021.py: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    required = (
        "VesselTriplet",
        "FeatureVector",
        "ProbePose",
        "DatabaseGenerator",
        "MultiLabelledCBIR",
        "HMMPoseEstimator",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(f"2021.py 缺少所需对象: {', '.join(missing)}")
    return module


def _basis_and_angles(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        basis = np.column_stack(
            [record["u_axis_world"], record["v_axis_world"], record["normal_world"]]
        ).astype(float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"位姿方向轴无效: {error}") from error
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("位姿方向轴必须是有限的三个三维向量")
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6):
        raise ValueError("位姿方向轴必须正交且为单位向量")
    if np.linalg.det(basis) <= 0.0:
        raise ValueError("位姿方向轴必须构成右手坐标系")

    cos_y = np.hypot(basis[0, 0], basis[1, 0])
    ry = np.arctan2(-basis[2, 0], cos_y)
    if cos_y > 1e-8:
        rx = np.arctan2(basis[2, 1], basis[2, 2])
        rz = np.arctan2(basis[1, 0], basis[0, 0])
    else:
        rx = np.arctan2(-basis[1, 2], basis[1, 1])
        rz = 0.0
    return basis, np.degrees([rx, ry, rz])


def _feature_vector_from_gallery_record(
    record: dict[str, Any], module: ModuleType
) -> tuple[Any, float]:
    if record.get("status") != "gallery":
        raise ValueError("图库记录 status 必须为 gallery")
    basis, angles = _basis_and_angles(record)
    center = np.asarray(record["center_world"], dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_world 必须是三个有限数值")
    pose = module.ProbePose(
        surface_point=center,
        rx=float(angles[0]),
        ry=float(angles[1]),
        rz=float(angles[2]),
        depth=0.0,
    )
    reconstructed = module.HMMPoseEstimator._rotation_matrix(
        pose.rx, pose.ry, pose.rz
    )
    rotation_error = float(np.max(np.abs(reconstructed - basis)))
    triplets = _triplets_from_features(record.get("features", []), module)
    if not triplets:
        raise ValueError("gallery 记录必须至少包含一个血管特征")
    return module.FeatureVector(triplets=triplets, pose=pose), rotation_error


def _triplets_from_features(features: Any, module: ModuleType) -> list[Any]:
    if not isinstance(features, list):
        raise ValueError("features 必须是列表")
    triplets = []
    for item in features:
        triplet = module.VesselTriplet(
            x=float(item["x_mm"]),
            y=float(item["y_mm"]),
            area=float(item["area_mm2"]),
            label=str(item["label"]),
        )
        values = (triplet.x, triplet.y, triplet.area)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("血管特征数值必须有限")
        if triplet.area < 0.0 or not triplet.label:
            raise ValueError("血管特征必须具有非负面积和非空标签")
        triplets.append(triplet)
    return triplets


def _formal_plane_dimensions(record: dict[str, Any], context: str) -> tuple[float, float]:
    try:
        width_mm = float(record["width_mm"])
        length_mm = float(record["length_mm"])
        spacing = record["pixel_spacing_mm"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{context} 必须包含 width_mm、length_mm 和 pixel_spacing_mm"
        ) from error
    if not all(math.isfinite(value) for value in (width_mm, length_mm)):
        raise ValueError(f"{context} 的平面尺寸必须是有限数值")
    if not math.isclose(
        width_mm, FORMAL_PLANE_WIDTH_MM, rel_tol=0.0, abs_tol=1e-6
    ) or not math.isclose(
        length_mm, FORMAL_PLANE_LENGTH_MM, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"{context} 平面尺寸必须为 100 mm x 100 mm")
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 2:
        raise ValueError(f"{context} pixel_spacing_mm 必须是两个数值")
    try:
        spacing_values = [float(value) for value in spacing]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} pixel_spacing_mm 必须是两个数值") from error
    if any(not math.isfinite(value) or value <= 0.0 for value in spacing_values):
        raise ValueError(f"{context} pixel_spacing_mm 必须是有限正数")
    return width_mm, length_mm


def _validate_formal_features(
    features: Any,
    *,
    width_mm: float,
    length_mm: float,
    context: str,
) -> None:
    if not isinstance(features, list):
        raise ValueError(f"{context} features 必须是列表")
    for index, item in enumerate(features):
        if not isinstance(item, dict):
            raise ValueError(f"{context} features[{index}] 必须是对象")
        label = item.get("label")
        if label not in FORMAL_VESSEL_LABELS:
            raise ValueError(
                f"{context} features[{index}] label 必须是 artery 或 vein"
            )
        try:
            x_mm = float(item["x_mm"])
            y_mm = float(item["y_mm"])
            area_mm2 = float(item["area_mm2"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{context} features[{index}] 必须包含数值 x_mm、y_mm、area_mm2"
            ) from error
        if not all(math.isfinite(value) for value in (x_mm, y_mm, area_mm2)):
            raise ValueError(f"{context} features[{index}] 数值必须有限")
        if not (0.0 <= x_mm <= width_mm and 0.0 <= y_mm <= length_mm):
            raise ValueError(f"{context} x_mm/y_mm 必须位于 100 mm 平面内")
        if area_mm2 < 0.0 or area_mm2 > width_mm * length_mm:
            raise ValueError(f"{context} area_mm2 必须位于平面面积范围内")


def _validate_formal_gallery_record(record: dict[str, Any], context: str) -> None:
    width_mm, length_mm = _formal_plane_dimensions(record, context)
    slice_id = record.get("slice_id")
    if not isinstance(slice_id, str) or not slice_id.strip():
        raise ValueError(f"{context} slice_id 必须是非空字符串")
    if record.get("status") != "gallery":
        raise ValueError(f"{context} status 必须为 gallery")
    features = record.get("features")
    _validate_formal_features(
        features,
        width_mm=width_mm,
        length_mm=length_mm,
        context=context,
    )
    if not features:
        raise ValueError(f"{context} gallery 记录必须至少包含一个血管特征")


def _validate_formal_eus_record(
    record: dict[str, Any], manifest_path: Path, context: str
) -> None:
    frame_id = record.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id.startswith("frame_"):
        raise ValueError(f"{context} frame_id 必须是 frame_ 开头的帧号")
    try:
        int(frame_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"{context} frame_id 必须包含数字后缀") from error
    if manifest_path.parent.name != frame_id:
        raise ValueError(f"{context} frame_id 必须与父目录一致")
    expected_name = f"{frame_id}_cropped_gallery.jsonl"
    if manifest_path.name != expected_name:
        raise ValueError(f"{context} 文件名必须为 {expected_name}")
    if record.get("slice_id") != f"{frame_id}_cropped":
        raise ValueError(f"{context} slice_id 与 frame_id 不一致")
    if record.get("pose_coordinate_system") != FORMAL_EUS_COORDINATE_SYSTEM:
        raise ValueError(
            f"{context} pose_coordinate_system 必须为 {FORMAL_EUS_COORDINATE_SYSTEM}"
        )
    if record.get("patient_world_pose") is not False:
        raise ValueError(f"{context} patient_world_pose 必须为 false")
    width_mm, length_mm = _formal_plane_dimensions(record, context)
    status = record.get("status")
    features = record.get("features")
    _validate_formal_features(
        features,
        width_mm=width_mm,
        length_mm=length_mm,
        context=context,
    )
    if status not in {"gallery", "unindexed"}:
        raise ValueError(f"{context} status 必须为 gallery 或 unindexed")
    if status == "gallery" and not features:
        raise ValueError(f"{context} status=gallery 时 features 不能为空")
    if status == "unindexed" and features:
        raise ValueError(f"{context} status=unindexed 时 features 必须为空")


def load_gallery_database(
    manifest_path: str | Path,
    registration_module_path: str | Path,
    *,
    require_organ_labels: bool = True,
    require_formal_contract: bool = False,
) -> GalleryDatabase:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"gallery.jsonl 不存在: {manifest_path}")
    module = load_registration_module(registration_module_path)
    database: dict[str, list[Any]] = defaultdict(list)
    features: list[Any] = []
    records_by_pose_id: dict[int, dict[str, Any]] = {}
    organ_labels_by_pose_id: dict[int, tuple[str, ...]] = {}
    max_rotation_error = 0.0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("记录必须是 JSON 对象")
                if "organ_labels" in record:
                    organ_labels = normalize_organ_labels(
                        record["organ_labels"],
                        context=f"{manifest_path.name} 第 {line_number} 行",
                    )
                elif require_organ_labels:
                    raise ValueError("缺少 organ_labels")
                else:
                    organ_labels = ()
                if require_formal_contract:
                    _validate_formal_gallery_record(
                        record, f"{manifest_path.name} 第 {line_number} 行"
                    )
                feature_vector, rotation_error = _feature_vector_from_gallery_record(
                    record, module
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{manifest_path.name} 第 {line_number} 行无效: {error}"
                ) from error
            key = module.DatabaseGenerator._make_db_key(feature_vector)
            database[key].append(feature_vector)
            features.append(feature_vector)
            records_by_pose_id[id(feature_vector.pose)] = record
            organ_labels_by_pose_id[id(feature_vector.pose)] = organ_labels
            max_rotation_error = max(max_rotation_error, rotation_error)

    if not features:
        raise ValueError(f"gallery.jsonl 没有可检索记录: {manifest_path}")
    return GalleryDatabase(
        module=module,
        database=dict(database),
        features=features,
        records_by_pose_id=records_by_pose_id,
        organ_labels_by_pose_id=organ_labels_by_pose_id,
        gallery_root=manifest_path.parent,
        manifest_path=manifest_path,
        max_rotation_error=max_rotation_error,
    )


def _query_feature_vector(record: dict[str, Any], module: ModuleType) -> Any | None:
    if record.get("status") != "gallery" or not record.get("features"):
        return None
    return module.FeatureVector(
        triplets=_triplets_from_features(record["features"], module)
    )


def _query_organ_labels(
    record: dict[str, Any],
    frame_id: str,
    manifest_path: Path,
    *,
    require_organ_labels: bool,
) -> tuple[tuple[str, ...], str, Path | None]:
    if "organ_labels" in record:
        return (
            normalize_organ_labels(record["organ_labels"], context=f"EUS {frame_id}"),
            "jsonl",
            manifest_path,
        )
    annotation_path = manifest_path.parent / f"{frame_id}_cropped_jpg_Label.tar"
    if annotation_path.is_file():
        return (
            load_organ_labels_from_tar(annotation_path),
            "tar_active_polygons",
            annotation_path.resolve(),
        )
    if not require_organ_labels:
        return (), "unavailable", None
    raise ValueError(
        f"EUS {frame_id} 缺少 organ_labels，且标注 TAR 不存在: {annotation_path}"
    )


def load_eus_queries(
    eus_root: str | Path,
    module: ModuleType,
    *,
    require_organ_labels: bool = True,
    require_formal_contract: bool = False,
) -> list[QueryRecord]:
    eus_root = Path(eus_root).resolve()
    if not eus_root.is_dir():
        raise FileNotFoundError(f"EUS 根目录不存在: {eus_root}")
    queries: list[QueryRecord] = []
    manifests = sorted(eus_root.glob("frame_*/*_cropped_gallery.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"EUS 目录中未找到 *_cropped_gallery.jsonl: {eus_root}")
    for manifest_path in manifests:
        lines = [
            line
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != 1:
            raise ValueError(f"EUS 清单必须恰好包含一条记录: {manifest_path}")
        try:
            record = json.loads(lines[0])
            frame_id = str(record["frame_id"])
            numeric_frame_id = int(frame_id.rsplit("_", 1)[1])
            if require_formal_contract:
                _validate_formal_eus_record(
                    record,
                    manifest_path,
                    f"EUS {manifest_path.name}",
                )
            feature_vector = _query_feature_vector(record, module)
            organ_labels, organ_label_source, organ_label_source_path = (
                _query_organ_labels(
                    record,
                    frame_id,
                    manifest_path,
                    require_organ_labels=require_organ_labels,
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"EUS 清单无效: {manifest_path}: {error}") from error
        queries.append(
            QueryRecord(
                numeric_frame_id=numeric_frame_id,
                frame_id=frame_id,
                status=str(record.get("status", "")),
                feature_vector=feature_vector,
                organ_labels=organ_labels,
                organ_label_source=organ_label_source,
                organ_label_source_path=organ_label_source_path,
                record=record,
                manifest_path=manifest_path,
            )
        )
    return sorted(queries, key=lambda query: query.numeric_frame_id)


def load_timestamps_csv(path: str | Path) -> dict[str, float]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"时间戳 CSV 不存在: {path}")
    result: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_id", "timestamp_seconds"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("时间戳 CSV 必须包含 frame_id,timestamp_seconds")
        for line_number, row in enumerate(reader, start=2):
            frame_id = str(row["frame_id"]).strip()
            if not frame_id or frame_id in result:
                raise ValueError(f"时间戳 CSV 第 {line_number} 行帧号为空或重复")
            try:
                timestamp = float(row["timestamp_seconds"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"时间戳 CSV 第 {line_number} 行不是数值") from error
            if not math.isfinite(timestamp):
                raise ValueError(f"时间戳 CSV 第 {line_number} 行必须是有限数值")
            result[frame_id] = timestamp
    if not result:
        raise ValueError("时间戳 CSV 没有数据")
    return result


def timestamps_for_frames(
    frame_ids: list[str] | tuple[str, ...], timestamps: dict[str, float]
) -> list[float]:
    missing = [frame_id for frame_id in frame_ids if frame_id not in timestamps]
    if missing:
        raise ValueError(f"缺少时间戳: {', '.join(missing)}")
    selected = [timestamps[frame_id] for frame_id in frame_ids]
    if any(current <= previous for previous, current in zip(selected, selected[1:])):
        raise ValueError("所选帧的时间戳必须严格递增")
    return selected
