from __future__ import annotations

import json
import math
from pathlib import Path
import tarfile
from typing import Any


DIRECT_ORGAN_LABELS = {
    14: "liver",
    15: "gallbladder",
    18: "spleen",
    19: "pancreas",
    22: "adrenal_gland_left",
    23: "adrenal_gland_right",
    24: "kidney_left",
    25: "kidney_right",
    41: "duodenum",
}
VALID_ORGAN_LABELS = frozenset(
    {
        "adrenal_gland_left",
        "adrenal_gland_right",
        "duodenum",
        "esophagus",
        "gallbladder",
        "kidney_left",
        "kidney_right",
        "liver",
        "pancreas",
        "spleen",
        "stomach",
    }
)


def normalize_organ_labels(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(label, str) or not label for label in value
    ):
        raise ValueError(f"{context} organ_labels 必须是字符串列表")
    normalized = tuple(sorted(set(value)))
    if list(normalized) != value:
        raise ValueError(f"{context} organ_labels 必须排序且不重复")
    unknown = sorted(set(normalized) - VALID_ORGAN_LABELS)
    if unknown:
        raise ValueError(f"{context} organ_labels 包含未知器官: {', '.join(unknown)}")
    return normalized


def _load_tar_metadata(annotation_path: Path) -> dict[str, Any]:
    try:
        with tarfile.open(annotation_path) as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".json")
            ]
            if len(members) != 1:
                raise ValueError(
                    f"EUS 标注 TAR 必须恰好包含一个 JSON: {annotation_path}"
                )
            source = archive.extractfile(members[0])
            if source is None:
                raise ValueError(f"无法读取 EUS 标注 JSON: {annotation_path}")
            metadata = json.load(source)
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"无法打开 EUS 标注 TAR: {annotation_path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"EUS 标注 JSON 必须是对象: {annotation_path}")
    return metadata


def _validate_organ_polygon(shape: dict[str, Any], annotation_path: Path) -> None:
    points: list[tuple[float, float]] = []
    for point in shape.get("Points") or []:
        position = point.get("Pos") if isinstance(point, dict) else None
        if (
            not isinstance(position, list)
            or len(position) < 2
            or not all(isinstance(value, (int, float)) for value in position[:2])
        ):
            raise ValueError(f"EUS 器官轮廓坐标无效: {annotation_path}")
        x, y = float(position[0]), float(position[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"EUS 器官轮廓坐标必须有限: {annotation_path}")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError(f"EUS 器官轮廓必须至少包含三个点: {annotation_path}")
    doubled_area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    )
    if doubled_area <= 0.0:
        raise ValueError(f"EUS 器官轮廓面积必须大于 0: {annotation_path}")


def load_organ_labels_from_tar(annotation_path: str | Path) -> tuple[str, ...]:
    annotation_path = Path(annotation_path)
    metadata = _load_tar_metadata(annotation_path)
    labels: set[str] = set()
    for polygon_group in metadata.get("Polys") or []:
        if not isinstance(polygon_group, dict):
            continue
        for shape in polygon_group.get("Shapes") or []:
            if not isinstance(shape, dict) or not shape.get("Actived", True):
                continue
            label = DIRECT_ORGAN_LABELS.get(shape.get("labelType"))
            if label is not None:
                _validate_organ_polygon(shape, annotation_path)
                labels.add(label)
    return tuple(sorted(labels))
