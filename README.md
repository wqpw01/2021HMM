# Ramalhinho 2021 本地检索复现

完整的数据流、算法、参数、输出字段、术语和无结果原因说明见
[`HMM文档.md`](HMM%E6%96%87%E6%A1%A3.md)。

本项目以 `registration/2021.py` 为核心，将已建 CT 特征库
`gallery.jsonl` 与 EUS 血管截面特征作为外部输入，执行：

1. EUS/CT 器官集合预筛选；
2. 多标签单帧血管 CBIR 检索；
3. 论文配置的六帧 HMM/Viterbi 候选路径筛选；
4. JSONL、CSV、运行元数据和对比图导出。

项目位于 WSL：

```text
/home/zyt/ramalhinho_2021_local_reproduction
```

Windows 资源管理器路径：

```text
\\wsl.localhost\Ubuntu\home\zyt\ramalhinho_2021_local_reproduction
```

数据不会复制进项目。CT `gallery.jsonl` 和 EUS 裁剪特征目录都是外部输入；本项目
不生成 `*_cropped_gallery.jsonl`，也不执行 CT 建库、分割或 TotalSegmentator。
唯一正式执行入口是下面的 `run` 子命令。

## 关键配置

`2021.py` 中 `LUSCTRegistrationFramework` 的默认 HMM 参数已修正为论文设置：

```text
sigma_x = 0.6 mm
sigma_y = 0.6 mm
sigma_z = 3.0 mm
sigma_theta = 2 degrees
```

默认检索与 HMM 参数：

```text
K = 200
r = 2
N = 6（目标帧 I1 + 5 个后续帧）
```

原始 `2021.py` 的 SHA-256 为：

```text
cd60f299d30d8cb9cfdf63820ed4b092e45df67ec1a7bcc69bdf960b43e1171b
```

运行元数据会同时记录原始哈希和当前修正版脚本哈希。

## 环境

按照本机长期规则，项目使用 Mamba：

```bash
cd /home/zyt/ramalhinho_2021_local_reproduction
mamba env create -f environment.yml
mamba activate ramalhinho-2021-reproduction
```

当前环境已经具备依赖时，也可以直接运行测试：

```bash
python -m pytest -q
```

## 输入要求

### CT 检索库

正式运行只读取：

```text
case_2/gallery/gallery.jsonl
```

默认器官过滤模式下，每条 `status="gallery"` 记录必须包含：

- `slice_id`、`organ`、`organ_labels`；
- `center_world`；
- `u_axis_world`、`v_axis_world`、`normal_world`；
- `features`，每项包含 `label、x_mm、y_mm、area_mm2`；
- `width_mm=100.0`、`length_mm=100.0` 和正的 `pixel_spacing_mm`；
- 每个特征的标签只能是 `artery` 或 `vein`，质心必须位于 100 mm × 100 mm 平面内；
- 可选的 `ct_overlay_png` 等相对图片路径。

三条方向轴必须是有限、正交、单位化且构成右手坐标系。图片路径相对于
`gallery.jsonl` 所在目录；图片缺失不影响数值检索，只会在可视化中显示占位图。

`organ` 表示该切面从哪个器官表面采样生成，不等于切面实际包含的器官，不能
用于过滤。`organ_labels` 才是切面实际相交的器官集合，必须是排序、去重后的
规范英文列表。空列表合法。

### EUS 特征

EUS 根目录按下列形式组织：

```text
<eus-root>/frame_*/<frame>_cropped_gallery.jsonl
```

项目按帧号末尾数字升序加载。`status="gallery"` 且 `features` 非空的帧参与
检索；`unindexed` 帧保留在报告中，并切断 HMM 连续序列。

正式 `run`/`validate` 会额外强制检查每个 EUS JSONL：文件必须与父目录和
`frame_id` 同名，`slice_id` 必须为 `<frame>_cropped`；必须声明
`pose_coordinate_system="synthetic_2d_10cm_crop"`、`patient_world_pose=false`、
100×100 mm 平面和有效 `pixel_spacing_mm`。EUS 的 `x_mm/y_mm` 采用左上角原点、
x 向右、y 向下的二维局部毫米坐标。

器官信息按以下顺序读取：

1. EUS JSONL 已有 `organ_labels` 时直接使用；
2. 缺少字段时，解析同目录 `<frame>_cropped_jpg_Label.tar`；
3. 两者都没有时，默认过滤模式报错。

TAR 只读取 `Polys[].Shapes[]` 中启用的直接器官轮廓，不读取帧级
`FrameLabelModel`，也不把胆管、胰管或胰腺分区推断为器官。支持的直接器官
标签为肝、胆囊、脾、胰腺、左右肾上腺、左右肾和十二指肠；轮廓必须至少有
三个有限坐标点且面积大于零。

默认匹配规则是“任意重合”：只要 EUS 与 CT 的 `organ_labels` 至少共享一个
器官，该 CT 切面即可进入后续血管 CBIR。EUS 器官为空或过滤后零候选时回退
全 CT 库，并在结果中记录不同的回退原因。

### 可选真实时间戳

时间戳 CSV 格式见 `examples/timestamps.example.csv`：

```csv
frame_id,timestamp_seconds
frame_00000073,0.000
frame_00000273,0.025
```

提供 CSV 时，参与窗口的时间戳必须完整、有限且严格递增。不提供时，每个六帧
窗口使用 `[0,1,2,3,4,5]`，仅表达等间隔，不代表 40 Hz 的真实采样时间。

## 使用方法

只检查 EUS 目录和特征清单时，可以先运行：

```bash
python run_reproduction.py validate-eus \
  --eus-root '/mnt/c/Users/zhangyutang/Desktop/交付宋老师文件2026.7.25/EUS标注与特征'
```

联合检查真实 `gallery.jsonl` 与 EUS 时，运行：

```bash
python run_reproduction.py validate \
  --gallery-jsonl '/path/to/case_2/gallery/gallery.jsonl' \
  --eus-root '/path/to/EUS标注与特征'
```

正式执行：

```bash
python run_reproduction.py run \
  --gallery-jsonl '/path/to/case_2/gallery/gallery.jsonl' \
  --eus-root '/path/to/EUS标注与特征' \
  --output-dir '/path/to/results'
```

默认值复现已完成实验（`K=200`、`r=2`、六帧 HMM、`sigma=0.6/0.6/3.0/2.0`）。
检索和 HMM 参数可在同一个正式入口覆盖，例如：

```bash
python run_reproduction.py run \
  --gallery-jsonl '/path/to/case_2/gallery/gallery.jsonl' \
  --eus-root '/path/to/EUS标注与特征' \
  --output-dir '/path/to/results-k100-r1' \
  --k 100 \
  --search-range 1 \
  --hmm-window-size 6 \
  --sigma-x 0.6 --sigma-y 0.6 --sigma-z 3.0 --sigma-theta 2.0
```

可覆盖的检索参数是 `--k`、`--search-range`；连续流程参数是
`--hmm-window-size`、四个 `--sigma-*`、`--timestamps-csv` 和
`--organ-filter-mode`。实际值会写入 `run_metadata.json` 的 `parameters`。

显式复现旧版、不使用器官预筛选的全库血管基线：

```bash
python run_reproduction.py run \
  --gallery-jsonl '/path/to/case_2/gallery/gallery.jsonl' \
  --eus-root '/path/to/EUS标注与特征' \
  --organ-filter-mode off \
  --output-dir '/path/to/results-baseline'
```

传入真实时间戳：

```bash
python run_reproduction.py run \
  --gallery-jsonl '/path/to/case_2/gallery/gallery.jsonl' \
  --eus-root '/path/to/EUS标注与特征' \
  --timestamps-csv examples/timestamps.example.csv \
  --output-dir '/path/to/results'
```

使用 `python run_reproduction.py run --help` 查看所有参数。所有覆盖值都会写入
`run_metadata.json`，避免结果与运行配置脱节。输出目录必须不存在或为空。

## 输出

```text
results/
├── single_frame_results.jsonl
├── single_frame_summary.csv
├── hmm_diagnostic_windows.jsonl
├── run_metadata.json
├── README.md
├── visualizations/
└── contact_sheets/
```

- `single_frame_results.jsonl`：每帧器官来源、过滤决策、查询特征、Top-K 和
  HMM 选中项；
- `single_frame_summary.csv`：便于表格查看的 Top-1 与 HMM 摘要；
- `hmm_diagnostic_windows.jsonl`：每个六帧窗口、时间戳、路径和转移代价；
- `run_metadata.json`：CT/EUS/可选时间戳输入哈希、`workflow_contract`、坐标输入
  契约、实际参数、统计、假设和姿态恢复误差；
- `visualizations`：EUS、单帧 Top-1、HMM 结果三联图。

## 事实边界

- 当前 `2021.py` 的 `r=2` 实现是每个输入标签分别允许数量差不超过 2，未改成
  论文公式中所有类别数量差之和不超过 2。
- HMM 继续使用原脚本逻辑：CBIR 先选 Top-K，进入 HMM 后节点代价为 0，路径
  主要由相邻候选的运动转移代价决定。
- 器官预筛选是本项目在论文血管 CBIR 之前新增的策略，不是论文原始距离公式。
  器官不参与血管距离或 HMM 转移代价。
- 多器官采用用户指定的“任意重合”，与服务器已有 DINO 流程采用的“包含全部”
  不是同一规则。
- `organ` 单值只写入结果；只有 `organ_labels` 集合参与预筛选。
- 当前 EUS 数据声明 `patient_world_pose=false`，因此 HMM 输出是 CT 候选路径，
  不是经过真值验证的患者三维轨迹。
- 没有患者真值时不计算 TRE，也不报告论文意义上的配准成功率。
