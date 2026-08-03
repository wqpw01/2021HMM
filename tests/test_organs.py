from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest


def write_annotation_tar(path: Path, shapes: list[dict]) -> None:
    metadata = {
        "Models": {
            "ColorLabelTableModel": [
                {"ID": 15, "Desc": "胆囊"},
                {"ID": 18, "Desc": "脾脏"},
                {"ID": 24, "Desc": "左侧肾脏"},
                {"ID": 26, "Desc": "门静脉（包括分支"},
            ]
        },
        "Polys": [{"Shapes": shapes}],
    }
    payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo(path.with_suffix(".json").name)
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))


def polygon(label_id: int, *, active: bool = True) -> dict:
    return {
        "Actived": active,
        "labelType": label_id,
        "Points": [
            {"Pos": [0.0, 0.0]},
            {"Pos": [10.0, 0.0]},
            {"Pos": [0.0, 10.0]},
        ],
    }


def test_tar_parser_uses_only_active_direct_organ_polygons(tmp_path: Path):
    organs = __import__(
        "ramalhinho2021.organs", fromlist=["load_organ_labels_from_tar"]
    )
    annotation = tmp_path / "frame_00000001_cropped_jpg_Label.tar"
    write_annotation_tar(
        annotation,
        [
            polygon(24),
            polygon(15),
            polygon(18, active=False),
            polygon(26),
        ],
    )

    result = organs.load_organ_labels_from_tar(annotation)

    assert result == ("gallbladder", "kidney_left")


def test_tar_parser_rejects_degenerate_active_organ_polygon(tmp_path: Path):
    organs = __import__(
        "ramalhinho2021.organs", fromlist=["load_organ_labels_from_tar"]
    )
    annotation = tmp_path / "frame_00000001_cropped_jpg_Label.tar"
    shape = polygon(15)
    shape["Points"] = [
        {"Pos": [0.0, 0.0]},
        {"Pos": [1.0, 1.0]},
        {"Pos": [2.0, 2.0]},
    ]
    write_annotation_tar(annotation, [shape])

    with pytest.raises(ValueError, match="器官轮廓.*面积"):
        organs.load_organ_labels_from_tar(annotation)


def test_normalize_organ_labels_requires_sorted_unique_canonical_list():
    organs = __import__(
        "ramalhinho2021.organs", fromlist=["normalize_organ_labels"]
    )

    assert organs.normalize_organ_labels(
        ["kidney_left", "spleen"], context="EUS"
    ) == ("kidney_left", "spleen")
    assert organs.normalize_organ_labels([], context="EUS") == ()
    with pytest.raises(ValueError, match="排序且不重复"):
        organs.normalize_organ_labels(["spleen", "kidney_left"], context="EUS")
    with pytest.raises(ValueError, match="未知器官"):
        organs.normalize_organ_labels(["left_kidney"], context="EUS")


def test_tar_parser_reports_corrupt_archive_as_input_error(tmp_path: Path):
    organs = __import__(
        "ramalhinho2021.organs", fromlist=["load_organ_labels_from_tar"]
    )
    annotation = tmp_path / "frame_00000001_cropped_jpg_Label.tar"
    annotation.write_bytes(b"not a tar archive")

    with pytest.raises(ValueError, match="无法打开 EUS 标注 TAR"):
        organs.load_organ_labels_from_tar(annotation)
