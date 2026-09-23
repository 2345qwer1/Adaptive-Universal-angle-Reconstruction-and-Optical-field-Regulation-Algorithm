# run_resolution_w_study_2.0.py 结构与功能说明

## 1. 脚本用途

`run_resolution_w_study_2.0.py` 用于扫描结构特征宽度 `m`，判断不同分析半径 `r/R` 处的结构是否能被 dose 场分辨出来。

核心判定逻辑是：

```text
如果某个 m 下 CTF_dose_mean >= CTF_DOSE_THRESHOLD，
则把该 r/R 位置第一次满足条件的 m 记录为分辨宽度 w。
```

因此，脚本最终回答的问题是：

```text
在不同半径位置 r/R 上，proto dose 和 optimized dose 分别需要多大的结构宽度 w 才能被分辨出来？
```

## 2. 支持的结构类型

脚本内部用 `a1/a2/a3` 表示三类结构：

| 名称 | 结构类型 | 含义 |
|---|---|---|
| `a1` | `radial_concentric_ring` | 径向同心环结构 |
| `a2` | `angular_blocks_single_layer` | 单层角向块结构，每个 `r/R` 单独生成一个 target |
| `a3` | `angular_blocks_multi_layer` | 多层角向块结构，多个 `r/R` 环带同时存在于一个 target |

当前运行哪一种结构由：

```python
CURRENT_STRUCTURE_TYPE = a3
```

决定。

## 3. 主要输入参数

### 扫描参数

| 参数 | 作用 |
|---|---|
| `M_VALUES` | 要扫描的结构宽度 `m`，单位为 px |
| `CTF_DOSE_THRESHOLD` | 判定分辨成功的 `CTF_dose_mean` 阈值 |
| `KEEP_SCANNING_AFTER_RESOLVED` | 命中 `w` 后是否继续扫描更大的 `m` |
| `RADIAL_RING_R_OVER_R_LIST` | `a1` 的分析半径列表 |
| `ANGULAR_SINGLE_LAYER_R_OVER_R_LIST` | `a2` 的分析半径列表 |
| `ANGULAR_MULTI_LAYER_R_OVER_R_LIST` | `a3` 的分析半径列表 |

### dose 类型

| 类型 | 含义 |
|---|---|
| `proto` | 初始投影经过补偿、量化和全局缩放后的 dose |
| `optimized` | 在 `proto` 基础上继续迭代优化后的 dose |

由下面参数控制：

```python
DOSE_TYPES = ["proto", "optimized"]
```

### VAM 仿真参数

| 参数 | 作用 |
|---|---|
| `CFG_N_SLICE` | 二维计算网格大小 |
| `CFG_ANGLE_NUM` | 投影角度数量 |
| `CFG_RESIN_ALPHA` | 树脂吸收系数 |
| `CFG_RESIN_EC` | 固化阈值 |
| `CFG_NUM_ITERATIONS` | optimized dose 的最大优化迭代次数 |
| `CFG_USE_NUMBA` | 是否使用 Numba 加速 |

## 4. 核心计算流程

主入口是：

```python
main()
```

整体流程如下：

1. `build_cfg()` 构建 VAM 仿真配置。
2. 根据 `CURRENT_STRUCTURE_TYPE` 选择对应的 `r/R` 列表。
3. 构建两个导出回调：
   - `build_resolved_profile_exporter()`
   - `build_m_iteration_exporter()`
4. 调用 `find_resolution_w_for_structure_fast()` 执行完整扫描。
5. 根据 `resolved_w` 生成 summary 表。
6. 导出 detail CSV、summary CSV、w-vs-r 图和 CTF_dose heatmap。

## 5. 分辨指标 CTF_dose_mean

脚本使用 `CTF_dose_mean` 判断结构是否被分辨。

基本形式是：

```text
CTF_dose = (D_peak - D_valley) / (D_peak + D_valley)
```

其中：

| 符号 | 含义 |
|---|---|
| `D_peak` | target 区域对应的 dose |
| `D_valley` | background 区域对应的 dose |
| `CTF_dose_mean` | 多个局部周期或采样位置上的 CTF_dose 平均值 |
| `CTF_dose_std` | 多个局部 CTF_dose 值的标准差 |

不同结构的计算方式不同：

| 结构 | CTF_dose 计算方式 |
|---|---|
| `a1` | 调用 `compute_radial_ring_ctf_dose_mean()`，沿径向 ring/gap 计算 |
| `a2` | 调用 `compute_angular_block_ctf_dose_mean()`，在指定 `r/R` 圆周上计算 |
| `a3` | 调用 `compute_angular_block_ctf_dose_mean()`，但使用多层角向块的对称轴参数 |

## 6. 主扫描函数结构

核心函数是：

```python
find_resolution_w_for_structure_fast()
```

它负责：

- 初始化 `resolved_w`
- 按 `m` 从小到大扫描
- 生成 target
- 运行 dose 仿真
- 对每个 `r/R` 和 `dose_type` 计算 `CTF_dose_mean/CTF_dose_std`
- 判断是否命中 `w`
- 记录 detail row
- 在命中时触发 `hit_w_structure` 导出
- 在每个 `m` 时触发 `m_iterations` 导出

重构后，重复逻辑被拆到以下辅助函数：

| 函数 | 作用 |
|---|---|
| `_init_resolved_w()` | 初始化每个 `(dose_type, r/R)` 的命中状态 |
| `_emit_iteration()` | 统一触发每个 `m` 的中间结果导出 |
| `_evaluate_radius_for_dose_types()` | 统一处理某个 `r/R` 下所有 dose type 的判定 |
| `_handle_resolution_hit()` | 只在第一次命中时记录 `w` 并触发导出 |
| `_record_detail_row()` | 统一写入 detail row |

## 7. 输出文件结构

输出根目录由：

```python
OUTPUT_DIR = ANALYSIS_OUTPUT_ROOT / OUTPUT_DIR_NAME
```

决定。

当前默认是：

```text
F:\USTC\项目\VAM\5.19-1\resolution_w_study_2_0
```

### 7.1 detail CSV

文件名由结构类型决定，例如：

```text
angular_blocks_multi_layer_resolution_detail.csv
```

每一行表示一次具体判定：

```text
structure_type + dose_type + m + r/R -> CTF_dose_mean / CTF_dose_std / resolved / w_px
```

字段包括：

| 字段 | 含义 |
|---|---|
| `structure_type` | 当前结构类型 |
| `dose_type` | `proto` 或 `optimized` |
| `m` | 当前扫描宽度 |
| `r_over_R` | 分析半径比例 |
| `r_px` | 实际分析半径，单位 px |
| `CTF_dose_mean` | 平均分辨指标 |
| `CTF_dose_std` | CTF_dose 值标准差 |
| `resolved` | 当前 `m` 是否达到阈值 |
| `w_px` | 该位置第一次命中的宽度，未命中时为空 |

### 7.2 summary CSV

文件名：

```text
resolution_w_summary.csv
```

每一行对应一个 `r/R`，记录最终命中的：

```text
w_proto_px
w_optimized_px
```

### 7.3 w-vs-r 图

文件名由结构类型决定，例如：

```text
fig_angular_blocks_multi_layer_w_vs_r.png
```

作用是显示不同 `r/R` 位置上的最终分辨宽度 `w`。

### 7.4 CTF_dose heatmap

文件名由结构类型决定，例如：

```text
fig_angular_blocks_multi_layer_CTF_dose_heatmap.png
```

作用是显示完整 `m-r/R` 平面上的 `CTF_dose_mean` 分布。

## 8. m_iterations 输出

目录：

```text
m_iterations/
```

该目录保存每个 `m` 下的中间结果。

### a1 输出

对 `a1`，每个 `m`、每个 `dose_type` 导出一套径向 dose distribution：

```text
proto_m_010_dose.svg
proto_m_010_dose_distribution.csv
proto_m_010_dose_distribution.svg
proto_m_010_target_cured_error.svg
proto_m_010_data.npz
proto_m_010_meta.json
```

### a2/a3 输出

对 `a2/a3`，每个 `m`、每个 `dose_type` 在 `m_iterations/` 根目录只保存整体结构结果：

```text
proto_m_010_dose.svg
proto_m_010_target_cured_error.svg
proto_m_010_data.npz
proto_m_010_meta.json
```

每个 `r/R` 的圆周 dose distribution 按 `m` 单独放在对应子目录：

```text
m_iterations/m_10_dose_distribution/
```

例如：

```text
m_iterations/m_10_dose_distribution/proto_m_010_rR_0.300_dose_distribution.csv
m_iterations/m_10_dose_distribution/proto_m_010_rR_0.300_dose_distribution.svg
```

## 9. hit_w_structure 输出

目录：

```text
hit_w_structure/
```

该目录保存“第一次命中 `w` 时”的证据数据。

它按 `r/R` 分子目录，例如：

```text
hit_w_structure/rR_0.100/
hit_w_structure/rR_0.200/
...
hit_w_structure/rR_0.900/
```

如果某个 `r/R` 在 `m=10` 时第一次满足 `CTF_dose_mean >= CTF_DOSE_THRESHOLD`，则该目录中保存对应 `m=w=10` 的结果。

对 `a2/a3`，典型文件包括：

```text
proto_m_010_dose.svg
proto_m_010_target_cured_error.svg
proto_m_010_data.npz
proto_m_010_meta.json
proto_m_010_rR_0.300_dose_distribution.csv
proto_m_010_rR_0.300_dose_distribution.svg
```

## 10. data.npz 和 meta.json

### data.npz

`data.npz` 是可用于后处理的压缩数组包。

通常包含：

| 数组 | 含义 |
|---|---|
| `target_mask` | 目标结构二值图 |
| `dose_field` | 二维 dose 场 |
| `cured_mask` | 按 `dose >= Ec` 得到的固化区域 |
| `error_map` | 欠固化/过固化误差图 |
| `profile_*` | 对应 profile 的采样数组 |

### meta.json

`meta.json` 是对应数据包的说明文件。

通常包含：

| 字段 | 含义 |
|---|---|
| `output_kind` | 输出类型，例如 `m_iteration` 或 `resolved_w` |
| `structure_type` | 结构类型 |
| `dose_type` | `proto` 或 `optimized` |
| `m` | 当前结构宽度 |
| `w_px` | 命中时记录的分辨宽度 |
| `r_over_R` | 分析半径比例 |
| `r_px` | 实际分析半径 |
| `container_r_px` | 容器半径 |
| `resin_Ec` | 固化阈值 |
| `dose_figure` | 对应 dose 图文件名 |
| `structure_figure` | 对应 target/cured/error 图文件名 |
| `profile_figure` | 对应 profile 图文件名 |
| `profile_csv` | 对应 profile CSV 文件名 |

## 11. 函数分组说明

### 配置与选择

| 函数 | 作用 |
|---|---|
| `build_cfg()` | 构建 VAM 仿真配置 |
| `get_active_r_over_r_list()` | 根据结构类型选择半径列表 |
| `should_export_angular_profile_r_over_r()` | 判断是否导出某个角向半径的 profile |

### dose 计算

| 函数 | 作用 |
|---|---|
| `_build_initial_sinogram_and_weight_projection()` | 复用 Radon 结果生成 sinogram 和权重投影 |
| `_compensate_and_print_initial_projection_from_weight_projection()` | 执行 proto 阶段补偿、量化、缩放和 dose 仿真 |
| `run_projection_pipeline()` | 统一生成 `proto` 和可选的 `optimized` dose |

### 导出逻辑

| 函数 | 作用 |
|---|---|
| `build_resolved_profile_exporter()` | 构造命中 `w` 时的导出回调 |
| `build_m_iteration_exporter()` | 构造每个 `m` 的中间结果导出回调 |
| `_export_resolved_hit()` | 导出 `hit_w_structure` 内容 |
| `_export_m_iteration_payload()` | 导出 `m_iterations` 内容 |
| `_export_output_set_structure_only()` | 导出整体 dose、target/cured/error 和数据包 |
| `_export_dose_distribution_only_with_angular_profile()` | 导出某个 `r/R` 圆周上的 dose distribution |
| `_export_output_set_with_radial_profile()` | 导出径向 dose distribution |
| `_save_figure_data_bundle()` | 写入 `data.npz` 和 `meta.json` |

### 扫描与判定

| 函数 | 作用 |
|---|---|
| `find_resolution_w_for_structure_fast()` | 主扫描函数 |
| `_evaluate_radius_for_dose_types()` | 对某个 `r/R` 和所有 dose type 执行判定 |
| `_handle_resolution_hit()` | 第一次命中时记录 `w` 并触发导出 |
| `_record_detail_row()` | 写入 detail CSV 的一行 |
| `_all_resolved()` | 判断是否所有位置都已经命中 |

## 12. 修改脚本时的注意事项

1. 如果只想切换结构类型，优先修改 `CURRENT_STRUCTURE_TYPE`。
2. 如果只想改变扫描范围，优先修改 `M_VALUES` 和对应的 `*_R_OVER_R_LIST`。
3. 如果只想关闭 optimized 计算，把 `DOSE_TYPES` 改为 `["proto"]`。
4. 如果要保持 heatmap 完整，应保留 `KEEP_SCANNING_AFTER_RESOLVED = True`。
5. 不建议直接修改 `find_resolution_w_for_structure_fast()` 内部判定逻辑，除非同步检查 detail CSV、summary CSV、heatmap 和 hit 输出。
6. 修改输出命名时，需要同时检查 `m_iterations`、`hit_w_structure`、`meta.json` 中记录的文件名是否一致。
