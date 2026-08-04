from pathlib import Path
import re


PROJECT_GUIDE = Path(__file__).resolve().parents[1] / "PROJECT_GUIDE_zh.md"


def _mermaid_blocks(document: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", document, flags=re.DOTALL)


def test_mermaid_node_labels_are_quoted_for_version_9_compatibility():
    blocks = _mermaid_blocks(PROJECT_GUIDE.read_text(encoding="utf-8"))

    assert blocks
    for block in blocks:
        assert not re.search(r"\b[A-Z][A-Z0-9]*\[(?!\")", block)
        assert not re.search(r"\b[A-Z][A-Z0-9]*\{(?!\")", block)


def test_eus_retrieval_gate_is_explicitly_distinguished_from_ct_filtering():
    document = PROJECT_GUIDE.read_text(encoding="utf-8")

    assert "EUS 查询帧准入" in document
    assert "这一步检查的是 EUS" in document
    assert "不是对 CT 检索库做筛选" in document


def test_unindexed_frames_distinguish_missing_from_rejected_vessel_features():
    document = PROJECT_GUIDE.read_text(encoding="utf-8")

    assert "没有可用血管特征" in document
    assert "`frame_00005310`" in document and "门静脉、脾静脉触边" in document
    assert "`frame_00016189`" in document and "腹主动脉触边" in document
    assert "`frame_00032757`" in document and "门静脉、下腔静脉触边" in document
    assert "`touches_image_edge`" in document
