# Ramalhinho 2021 本地检索复现

本项目以 `registration/2021.py` 为核心，将已建 CT 特征库
`gallery.jsonl` 与 EUS 血管截面特征作为外部输入，执行：

1. 多标签单帧 CBIR 检索；
2. 论文配置的六帧 HMM/Viterbi 候选路径筛选；
3. JSONL、CSV、运行元数据和对比图导出。

项目位于 WSL：

```text
/home/zyt/ramalhinho_2021_local_reproduction
```

Windows 资源管理器路径：

```text
\\wsl.localhost\Ubuntu\home\zyt\ramalhinho_2021_local_reproduction
```

数据不会复制进项目。EUS 目录和未来取得的 `gallery.jsonl` 均通过命令行传入。

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

每条 `status="gallery"` 记录必须包含：

- `slice_id`、`organ`；
- `center_world`；
- `u_axis_world`、`v_axis_world`、`normal_world`；
- `features`，每项包含 `label、x_mm、y_mm、area_mm2`；
- 可选的 `ct_overlay_png` 等相对图片路径。

三条方向轴必须是有限、正交、单位化且构成右手坐标系。图片路径相对于
`gallery.jsonl` 所在目录；图片缺失不影响数值检索，只会在可视化中显示占位图。

### EUS 特征

EUS 根目录按下列形式组织：

```text
<eus-root>/frame_*/<frame>_cropped_gallery.jsonl
```

项目按帧号末尾数字升序加载。`status="gallery"` 且 `features` 非空的帧参与
检索；`unindexed` 帧保留在报告中，并切断 HMM 连续序列。

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

目前真实图库缺失时，可以先验证现有 EUS：

```bash
python run_reproduction.py validate-eus \
  --eus-root '/mnt/c/Users/zhangyutang/Desktop/交付文件2026.7.25/EUS标注与特征'
```

拿到真实 `gallery.jsonl` 后，先验证图库和 EUS：

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

- `single_frame_results.jsonl`：每帧查询特征、Top-K 和 HMM 选中项；
- `single_frame_summary.csv`：便于表格查看的 Top-1 与 HMM 摘要；
- `hmm_diagnostic_windows.jsonl`：每个六帧窗口、时间戳、路径和转移代价；
- `run_metadata.json`：输入哈希、参数、统计、假设和姿态恢复误差；
- `visualizations`：EUS、单帧 Top-1、HMM 结果三联图。

## 事实边界

- 当前 `2021.py` 的 `r=2` 实现是每个输入标签分别允许数量差不超过 2，未改成
  论文公式中所有类别数量差之和不超过 2。
- HMM 继续使用原脚本逻辑：CBIR 先选 Top-K，进入 HMM 后节点代价为 0，路径
  主要由相邻候选的运动转移代价决定。
- `organ` 只写入结果，不参与检索过滤；算法使用的是血管标签与截面特征。
- 当前 EUS 数据声明 `patient_world_pose=false`，因此 HMM 输出是 CT 候选路径，
  不是经过真值验证的患者三维轨迹。
- 没有患者真值时不计算 TRE，也不报告论文意义上的配准成功率。
