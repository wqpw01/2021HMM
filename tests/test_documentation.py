from pathlib import Path
import re


PROJECT_GUIDE = Path(__file__).resolve().parents[1] / "HMM文档.md"

EXPECTED_SECTIONS = [
    "## 1. 执行摘要",
    "## 2. 项目目标、输入与输出",
    "## 3. 总体架构与详细流程图",
    "## 4. 项目结构",
    "## 5. 输入要求与参数配置",
    "## 6. 处理流程详解",
    "## 7. 数据流、特征与坐标系统",
    "## 8. 输出目录与数据协议",
    "## 9. 核心代码文件说明",
    "## 10. 安装、验证与正式运行",
    "## 11. 调试、测试与结果核验",
    "## 12. 性能、恢复与部署建议",
    "## 13. 注意事项与已知边界",
    "## 14. 故障排查表",
    "## 15. 复现与审计附录",
]


def _mermaid_blocks(document: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", document, flags=re.DOTALL)


def test_document_uses_reference_project_structure():
    document = PROJECT_GUIDE.read_text(encoding="utf-8")
    numbered_sections = re.findall(r"^## \d+\..+$", document, flags=re.MULTILINE)
    assert numbered_sections == EXPECTED_SECTIONS
    assert "## 结论" in document


def test_document_has_detailed_diagrams_and_fallbacks():
    document = PROJECT_GUIDE.read_text(encoding="utf-8")
    assert len(_mermaid_blocks(document)) >= 6
    assert document.count("不支持 Mermaid") >= 1
    assert document.count("```text") >= 8


def test_document_preserves_run_facts_and_claim_boundaries():
    document = PROJECT_GUIDE.read_text(encoding="utf-8")
    for fact in [
        "112,749",
        "EUS 总帧数=105",
        "retrieved=91",
        "unindexed=14",
        "diagnostic_only=88",
        "insufficient_contiguous_valid_frames=3",
        "HMM 窗口=43",
        "single_candidate_count",
        "equal_unit_intervals",
        "没有患者世界坐标",
        "不能据此计算定位准确率",
    ]:
        assert fact in document


def test_document_describes_implementation_boundaries_accurately():
    document = PROJECT_GUIDE.read_text(encoding="utf-8")
    for fact in [
        "不会自动归一化、正交化或翻转修复",
        "python -m pip install -e .",
        "额外标签只会增大类别数除数 `C`",
        "严格量纲不一致",
        'EO -- 否 --> ET["同帧 Label.tar"]',
        "验证标签已经排序且不重复",
        "前两帧 Top-1 CT 候选中心",
        "论文公式 5 使用所有类别数量差绝对值之和",
        "40 Hz",
        "`portal/hepatic`",
        "`artery/vein`",
        "EUS 只有 vein，而 Top-1 CT 只有 artery",
        "当前完全未使用",
    ]:
        assert fact in document


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
