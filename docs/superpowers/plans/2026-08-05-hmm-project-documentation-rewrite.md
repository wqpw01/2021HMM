# Ramalhinho 2021 HMM Project Documentation Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 CT 血管重采样项目说明为结构和排版基准，重写可维护的中文 HMM 项目说明，并在 Windows 桌面交付 Markdown 与 HTML 两种格式。

**Architecture:** 项目根目录的 `HMM文档.md` 作为唯一正文事实源，测试保护章节、流程图和关键运行事实；桌面 Markdown 是该源文件的发布副本，HTML 由同一 Markdown 机械转换生成。所有内容先与代码、测试和缓存的正式运行结果核对，最后才删除桌面旧的独立 `HMM文档.md`。

**Tech Stack:** Markdown、Mermaid 9.1.2/11、Mamba、Pandoc、Python/pytest、Node.js、xmllint、Git。

---

## 文件映射

**项目内维护文件：**

- Rename/Rewrite: `PROJECT_GUIDE_zh.md` → `HMM文档.md`
- Modify: `README.md`
- Modify: `tests/test_documentation.py`

**事实输入：**

- Read: `/tmp/hmm-doc-facts/run_metadata.json`
- Read: `/tmp/hmm-doc-facts/single_frame_summary.csv`
- Read: `/tmp/hmm-doc-facts/single_frame_results.jsonl`
- Read: `/tmp/hmm-doc-facts/hmm_diagnostic_windows.jsonl`
- Read: `/tmp/hmm-doc-facts/README.md`
- Read: `registration/2021.py`
- Read: `src/ramalhinho2021/*.py`
- Read: `tests/*.py`
- Reference: `/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明_20260804/CT血管重采样项目详细说明.md`
- Reference: `/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明_20260804/CT血管重采样项目详细说明.html`

**桌面交付文件：**

- Create: `/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.md`
- Create: `/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.html`
- Remove after verification: `/mnt/c/Users/zhangyutang/Desktop/HMM文档.md`

## Task 1: 固化文档契约并观察失败

**Files:**

- Modify: `tests/test_documentation.py`
- Test: `tests/test_documentation.py`

- [ ] **Step 1: 将文档测试指向用户已经移动后的维护文件名**

将：

```python
PROJECT_GUIDE = Path(__file__).resolve().parents[1] / "PROJECT_GUIDE_zh.md"
```

改为：

```python
PROJECT_GUIDE = Path(__file__).resolve().parents[1] / "HMM文档.md"
```

- [ ] **Step 2: 增加十五章、流程图和事实边界测试**

在 `tests/test_documentation.py` 增加：

```python
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


def test_document_uses_reference_project_structure():
    document = PROJECT_GUIDE.read_text(encoding="utf-8")
    positions = [document.index(section) for section in EXPECTED_SECTIONS]
    assert positions == sorted(positions)
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
        "105",
        "91",
        "14",
        "43",
        "88",
        "diagnostic_only",
        "equal_unit_intervals",
        "没有患者世界坐标",
        "不能据此计算定位准确率",
    ]:
        assert fact in document
```

- [ ] **Step 3: 运行测试并确认是预期失败**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction \
  python -m pytest tests/test_documentation.py -q
```

Expected: FAIL，原因应是当前文档缺少新十五章标题或不足六个流程图，而不是文件不存在或 Python 语法错误。

## Task 2: 建立可追溯的事实基线

**Files:**

- Read: `/tmp/hmm-doc-facts/*`
- Read: `registration/2021.py`
- Read: `src/ramalhinho2021/*.py`

- [ ] **Step 1: 验证事实缓存完整**

Run:

```bash
test -s /tmp/hmm-doc-facts/run_metadata.json
test -s /tmp/hmm-doc-facts/single_frame_summary.csv
test -s /tmp/hmm-doc-facts/single_frame_results.jsonl
test -s /tmp/hmm-doc-facts/hmm_diagnostic_windows.jsonl
test -s /tmp/hmm-doc-facts/README.md
sha256sum /tmp/hmm-doc-facts/*
```

Expected: 五个文件均非空并输出五条 SHA-256。

- [ ] **Step 2: 用结构化解析生成正式统计核对结果**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction python -c '
import csv, json
from collections import Counter
from pathlib import Path
root = Path("/tmp/hmm-doc-facts")
rows = list(csv.DictReader((root / "single_frame_summary.csv").open(encoding="utf-8")))
print("frames", len(rows))
print("retrieval", Counter(row["retrieval_status"] for row in rows))
print("hmm", Counter(row["hmm_status"] for row in rows))
print("candidate_counts", Counter(int(row["candidate_count"]) for row in rows))
print("windows", sum(1 for line in (root / "hmm_diagnostic_windows.jsonl").open(encoding="utf-8") if line.strip()))
metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
print("metadata_keys", sorted(metadata))
'
```

Expected: 105 帧；`retrieved=91`、`unindexed=14`；候选数主要为 200/0；HMM 窗口 43。

- [ ] **Step 3: 核对算法默认值和 CLI 默认值**

Run:

```bash
rg -n "sigma_[xyzt]|sigma_theta|search_range|hmm_window_size|organ_filter_mode|top_k|--k|--search-range" \
  registration/2021.py src/ramalhinho2021 tests README.md
```

Expected: 能定位 `sigma_x=0.6`、`sigma_y=0.6`、`sigma_z=3.0`、`sigma_theta=2.0`，以及 K、r、N 和器官过滤默认值。

- [ ] **Step 4: 建立写作时使用的口径表**

写作时只采用以下四栏，不把层级混写：

| 层级 | 内容 |
|---|---|
| 论文方法 | 多标签血管 CBIR、HMM/Viterbi 和运动转移模型 |
| `2021.py` | `VesselTriplet`、`FeatureVector`、`ProbePose`、CBIR、HMM 的实际实现 |
| 本地包装项目 | 器官预筛选、输入验证、窗口切分、输出和诊断状态 |
| 正式运行 | K=200、r=2、N=6、等单位时间、105/91/14/43/88 等统计 |

## Task 3: 重写第 1 至第 4 章

**Files:**

- Rewrite: `HMM文档.md`

- [ ] **Step 1: 写文档元信息与目录**

文件开头必须包含：

```markdown
# Ramalhinho 2021 器官预筛选、血管 CBIR 与 HMM 项目详细说明

> 文档基线：2026-08-05<br>
> 项目目录：`/home/zyt/ramalhinho_2021_local_reproduction`<br>
> Python 包：`ramalhinho2021`<br>
> 远程仓库：`https://github.com/wqpw01/2021HMM`<br>
> 适用对象：算法研发、医学影像工程、论文复现、结果审计与第一次接触项目的读者
```

随后列出十五章可点击目录，锚点文字与实际二级标题完全一致。

- [ ] **Step 2: 写第 1 章执行摘要**

必须包含：

- 一段完整任务定义；
- “EUS 准入 → CT 器官预筛选 → 血管 CBIR → 六帧 HMM”的一句话主链；
- 正式运行统计表：112,749 CT、105 EUS、91 retrieved、14 unindexed、43 窗口、88 HMM、3 帧仅缺 HMM；
- 状态表：`retrieved`、`unindexed`、`diagnostic_only`、`insufficient_contiguous_valid_frames`；
- 关键边界：无真值、等单位时间、器官不进入血管距离。

- [ ] **Step 3: 写第 2 章项目目标、输入与输出**

按“解决的问题、输入概览、输出概览、不属于本项目的能力”四节展开。输入同时列出 CT JSONL、EUS 每帧目录、可选时间戳和 `2021.py`；输出同时列出单帧、HMM、元数据和可视化。

- [ ] **Step 4: 写第 3 章六张详细流程图**

六张图分别为：

1. 端到端架构；
2. CT/EUS 双输入汇合；
3. EUS 查询准入与 CT 器官筛选；
4. 多标签单帧 CBIR；
5. 连续段、六帧窗口与 Viterbi；
6. 输出持久化。

所有节点使用 `A["文字"]`、`B{"判断"}` 形式；每张图后写纯文本步骤或解释。禁止在节点中使用未加引号的中英文混合标点。

- [ ] **Step 5: 写第 4 章项目结构**

目录树必须与当前仓库一致，说明 `run_reproduction.py`、`__main__.py`、`cli.py`、输入/器官/流水线/输出模块、`registration/2021.py`、测试和示例时间戳。另列 Git 基线和“当前用户已将说明文件重命名为 `HMM文档.md`”这一维护事实。

- [ ] **Step 6: 运行局部结构检查**

Run:

```bash
rg -n '^## [1-4]\. |^```mermaid|不支持 Mermaid' HMM文档.md
```

Expected: 第 1 至第 4 章按顺序存在，Mermaid 块总数至少 6。

## Task 4: 重写第 5 至第 7 章

**Files:**

- Modify: `HMM文档.md`

- [ ] **Step 1: 写第 5 章输入要求与参数配置**

必须覆盖：

- CT JSONL 字段表和真实结构示例；
- EUS 目录树和一帧记录示例；
- Label TAR 活动多边形解析与器官映射；
- EUS `status/features` 准入；
- 时间戳 CSV；
- K、r、N、器官模式、sigma 参数表；
- 论文/脚本/本项目/本次运行四种参数口径。

- [ ] **Step 2: 写第 6 章处理流程详解**

严格按调用顺序写：

```text
CLI → 加载 2021.py → 加载 CT 图库 → 生成数据库键
    → 加载 EUS → 器官解析 → EUS 准入
    → CT 器官预筛选 → 血管数量范围 r
    → 多标签距离 → Top-K → 连续段切分
    → 六帧窗口 → 时间间隔 → Viterbi → 输出
```

每节回答输入对象、输出对象、成功条件和失败状态。明确第 7.3 旧文档判断的是 EUS，不是 CT。

- [ ] **Step 3: 写第 7 章数据流、特征与坐标系统**

必须完整解释：

- JSONL 行到 `VesselTriplet`、`FeatureVector` 和 `ProbePose`；
- `x_mm/y_mm/area_mm2`；
- `center_world` 和 `u/v/normal`；
- ZYX 欧拉角只是同一朝向的另一种编码，不与 u/v/normal 数值相同；
- `surface_point`、`depth` 和 CT 候选位姿；
- `dx/dy/dz` 通过上一候选局部旋转矩阵转置投影；
- `theta` 由两帧法向夹角得到；
- sigma 对标准化转移代价的影响。

- [ ] **Step 4: 用代码定位检查术语没有脱离实现**

Run:

```bash
rg -n "class VesselTriplet|class FeatureVector|class ProbePose|class MultiLabelledCBIR|class HMMPoseEstimator|def _transition|def viterbi|def _rotation_matrix" registration/2021.py
```

Expected: 文档中的核心对象和运动计算都能定位到代码定义。

## Task 5: 重写第 8 至第 11 章

**Files:**

- Modify: `HMM文档.md`

- [ ] **Step 1: 写第 8 章输出目录与数据协议**

包括完整输出树、各产物矩阵、`single_frame_results.jsonl` 完整示例、CSV 字段、HMM 窗口示例、`run_metadata.json` 字段、可视化结构，以及四种无结果/回退状态判读。

- [ ] **Step 2: 写第 9 章核心代码文件说明**

按六组表格组织：入口与编排、输入和对象适配、器官解析、单帧/HMM、输出、算法核心与测试。每行包含“文件、关键对象/函数、核心职责、调试重点”。

- [ ] **Step 3: 写第 10 章安装、验证与正式运行**

命令必须使用 Mamba，并分别给出：

- 创建环境；
- `validate-eus`；
- `validate`；
- `run` 默认器官 overlap；
- `--timestamps-csv`；
- `--organ-filter-mode off` 基线；
- `python run_reproduction.py` 与 `python -m ramalhinho2021` 两种入口。

所有示例使用项目内相对路径或明确标记的 `/path/to/...` 用户替换路径，不写远端密码。

- [ ] **Step 4: 写第 11 章调试、测试与结果核验**

包括推荐调试顺序、pytest 命令、测试覆盖表、输入行数与状态检查、候选数检查、HMM 窗口检查、哈希检查和图片目视抽检。正式结果小节列出 18 个器官筛选有效帧、73 个空器官回退帧和四种器官候选池规模。

## Task 6: 重写第 12 至第 15 章和结论

**Files:**

- Modify: `HMM文档.md`

- [ ] **Step 1: 写第 12 章性能、恢复与部署建议**

只描述已有或可验证行为：图库一次加载、器官子库缓存、K/N 对计算量影响、输出目录必须为空、服务器/本地路径差异、Mamba 环境和输入哈希。明确当前项目不提供检索中途断点恢复。

- [ ] **Step 2: 写第 13 章注意事项与已知边界**

用编号列表覆盖至少 15 项，包括：无真值、HMM 诊断性、等单位时间、EUS 无世界位姿、器官过滤不是距离、Top-K 不是正确率、r 是逐标签数量容差、法向方向、欧拉角、depth、触边跳过、输出目录、患者数据安全和参数可比性。

- [ ] **Step 3: 写第 14 章故障排查表**

表格列为：

```text
现象/状态 | 常见原因 | 检查位置 | 处理方法
```

至少覆盖 12 类故障，并为 `unindexed` 区分“没有人工血管”和“人工血管触边被跳过”。

- [ ] **Step 4: 写第 15 章复现与审计附录**

包括：完整运行模板、正式运行前清单、运行后清单、哈希命令、本次运行标识、14 个无单帧结果帧、3 个仅缺 HMM 帧、术语表和文档依据。

- [ ] **Step 5: 写结论并检查禁用表述**

结论强调本项目输出的是 CT 候选及诊断性平滑路径。运行：

```bash
rg -n "定位准确率为|临床准确|真实轨迹已经|EUS 自带三维位姿" HMM文档.md
```

Expected: 无匹配；若为反例解释，必须改写为不含歧义的否定句。

## Task 7: 使文档契约变绿并更新仓库入口

**Files:**

- Modify: `README.md`
- Modify: `tests/test_documentation.py`
- Rename/Rewrite: `PROJECT_GUIDE_zh.md` → `HMM文档.md`

- [ ] **Step 1: 更新 README 文档链接**

将 README 中的：

```markdown
[`PROJECT_GUIDE_zh.md`](PROJECT_GUIDE_zh.md)
```

改为：

```markdown
[`HMM文档.md`](HMM文档.md)
```

- [ ] **Step 2: 运行文档测试**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction \
  python -m pytest tests/test_documentation.py -q
```

Expected: PASS。

- [ ] **Step 3: 运行完整项目测试**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction python -m pytest -q
```

Expected: 全部通过，无失败或错误。

- [ ] **Step 4: 运行 Markdown 静态检查**

Run:

```bash
git diff --check
rg -n "TBD|TODO|待补充" HMM文档.md
```

Expected: `git diff --check` 退出 0，规格占位词无匹配。

## Task 8: 发布桌面 Markdown 并生成 HTML

**Files:**

- Create: `/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.md`
- Create: `/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.html`
- Temporary: `/tmp/hmm-document-build/*`

- [ ] **Step 1: 创建干净的交付目录并复制唯一事实源**

Run:

```bash
mkdir -p '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805'
cp HMM文档.md \
  '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.md'
```

Expected: 目标 Markdown 与项目源文件 SHA-256 相同。

- [ ] **Step 2: 准备临时 Pandoc 工具**

Run:

```bash
mamba create -p /tmp/hmm-document-build/pandoc -y -c conda-forge pandoc
```

Expected: `/tmp/hmm-document-build/pandoc/bin/pandoc` 存在。

- [ ] **Step 3: 从参考 HTML 提取原始 CSS**

Run:

```bash
node -e '
const fs = require("fs");
const source = fs.readFileSync(
  "/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明_20260804/CT血管重采样项目详细说明.html",
  "utf8"
);
const match = source.match(/<style>[\s\S]*?<\/style>/);
if (!match) throw new Error("reference style block not found");
fs.writeFileSync("/tmp/hmm-document-build/style.html", match[0], "utf8");
'
```

Expected: `/tmp/hmm-document-build/style.html` 含 `<style>` 和响应式/打印规则。

- [ ] **Step 4: 创建 Mermaid Pandoc 过滤器**

Create `/tmp/hmm-document-build/mermaid.lua` with:

```lua
local function escape_html(text)
  return text:gsub("&", "&amp;")
             :gsub("<", "&lt;")
             :gsub(">", "&gt;")
end

function CodeBlock(block)
  if block.classes:includes("mermaid") then
    return pandoc.RawBlock(
      "html",
      '<pre class="mermaid">' .. escape_html(block.text) .. '</pre>'
    )
  end
end
```

- [ ] **Step 5: 创建正文包裹片段**

Create `/tmp/hmm-document-build/before.html` with:

```html
<main>
```

Create `/tmp/hmm-document-build/after.html` with:

```html
</main>
<div class="page-tools"><button type="button" title="返回顶部" aria-label="返回顶部" onclick="window.scrollTo({top:0,behavior:'smooth'})">↑</button></div>
<script type="module">
  try {
    const { default: mermaid } = await import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs');
    mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'strict' });
    await mermaid.run({ querySelector: '.mermaid' });
  } catch (error) {
    console.info('Mermaid 离线回退：保留流程图源码。', error);
  }
</script>
```

- [ ] **Step 6: 从 Markdown 生成独立 HTML**

Run:

```bash
/tmp/hmm-document-build/pandoc/bin/pandoc \
  '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.md' \
  --from=gfm \
  --to=html5 \
  --standalone \
  --metadata=lang:zh-CN \
  --metadata='title:Ramalhinho 2021 HMM 项目详细说明' \
  --include-in-header=/tmp/hmm-document-build/style.html \
  --include-before-body=/tmp/hmm-document-build/before.html \
  --include-after-body=/tmp/hmm-document-build/after.html \
  --lua-filter=/tmp/hmm-document-build/mermaid.lua \
  --output='/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.html'
```

Expected: HTML 为 UTF-8，含 `<main>`、十五章、`.mermaid`、页面按钮和参考 CSS。

## Task 9: 验证双格式交付并安全替换旧文件

**Files:**

- Test: desktop Markdown/HTML
- Remove: `/mnt/c/Users/zhangyutang/Desktop/HMM文档.md`

- [ ] **Step 1: 验证文件、篇幅和目录内容**

Run:

```bash
wc -l -w -c \
  '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.md' \
  '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.html'
find '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805' \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: 仅两份目标文件；Markdown 约 1,400 至 1,900 行。

- [ ] **Step 2: 验证 Markdown 与 HTML 内容对应**

Run:

```bash
rg -c '^## [0-9]+\. ' \
  '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.md'
rg -c '<h2 id="[0-9]+-' \
  '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.html'
xmllint --html --noout \
  '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.html'
```

Expected: Markdown 和 HTML 均为 15 个编号主章节；xmllint 退出码为 0。旧版
xmllint 可能提示 HTML5 的 `main` 标签未知，该兼容性提示不代表标签未闭合。

- [ ] **Step 3: 使用 Mermaid 9.1.2 实际解析全部图**

Run:

```bash
env LD_LIBRARY_PATH=/tmp/mermaid-browser-libs/lib \
  /tmp/mermaid-cli912/node_modules/.bin/mmdc \
  -i '/mnt/c/Users/zhangyutang/Desktop/Ramalhinho_2021_HMM项目说明_20260805/Ramalhinho_2021_HMM项目详细说明.md' \
  -o /tmp/hmm-project-document-rendered.md
```

Expected: 找到至少 6 张 Mermaid 图并全部生成 SVG，无 syntax error。

- [ ] **Step 4: 浏览器抽检 HTML**

使用本地浏览器打开：

```text
C:\Users\zhangyutang\Desktop\Ramalhinho_2021_HMM项目说明_20260805\Ramalhinho_2021_HMM项目详细说明.html
```

抽检首页、目录、至少一张流程图、一张宽表格、代码块、窄屏布局和返回顶部按钮。确认文字不重叠、表格可横向滚动、流程图不报错。

- [ ] **Step 5: 最后删除旧桌面独立文档**

仅当 Steps 1-4 全部通过后运行：

```bash
rm '/mnt/c/Users/zhangyutang/Desktop/HMM文档.md'
```

Expected: 旧文件不存在，新结果目录及两份文件仍存在。

## Task 10: 提交、推送与最终核验

**Files:**

- Commit: `HMM文档.md`
- Commit: deletion of `PROJECT_GUIDE_zh.md` as rename detection
- Commit: `README.md`
- Commit: `tests/test_documentation.py`

- [ ] **Step 1: 检查只提交预期项目文件**

Run:

```bash
git status --short
git diff --check
git diff -- README.md tests/test_documentation.py HMM文档.md PROJECT_GUIDE_zh.md
```

Expected: `.superpowers/` 视觉预览目录不进入提交；正文移动被 Git 识别为重命名或删除+新增；无算法代码变更。

- [ ] **Step 2: 再次运行完整测试**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction python -m pytest -q
```

Expected: 全部通过。

- [ ] **Step 3: 提交项目内维护源和回归测试**

Run:

```bash
git add README.md tests/test_documentation.py HMM文档.md PROJECT_GUIDE_zh.md
git commit -m "docs: rewrite comprehensive HMM project guide"
```

Expected: 提交只含文档、README 和文档测试。

- [ ] **Step 4: 推送并核对远端提交**

Run:

```bash
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: 本地 HEAD 与远端 `main` 哈希完全一致。

- [ ] **Step 5: 最终交付报告**

最终回复必须报告：

- 桌面结果目录和两份文件的可点击路径；
- Markdown/HTML 行数和大小；
- 主章节数、Mermaid 图数量；
- 项目测试数量和结果；
- Mermaid 9.1.2 与 HTML 结构验证结果；
- Git 提交哈希和远端同步状态；
- 旧 `HMM文档.md` 已在全部验证后删除。
