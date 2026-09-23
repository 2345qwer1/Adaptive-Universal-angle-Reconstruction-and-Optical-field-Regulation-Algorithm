# CAL 1.1 README

## 1. 项目概述

CAL 1.1 是一套基于单切片二维截面的体投影光固化（Volumetric Additive Manufacturing, VAM / CAL）仿真代码。该版本采用“函数层（`functions/`）+ 脚本层（`scripts/`）”的分层结构，覆盖如下完整流程：

1. 读取并预处理目标二值图像；
2. 由目标图生成初始正弦图（sinogram）；
3. 根据圆柱容器几何、Beer–Lambert 衰减与频域滤波生成初始投影矩阵；
4. 对初始投影矩阵执行全局补偿并进行前向剂量仿真；
5. 依据阈值能量 `Ec` 生成固化结果并计算 IoU；
6. 对投影矩阵执行连续优化，并在最终输出前离散化；
7. 对任意投影文件进行重打印验证；
8. 对点目标进行 PSF / 扩展误差分析。

该版本的核心定位是：

- **单切片二维物理仿真**；
- **圆形容器内剂量累积与固化判定**；
- **以 IoU 为核心几何相似性指标**；
- **支持投影矩阵离散级数量化**；
- **支持初始打印、优化打印、任意投影复现、点扩展误差分析**。

---

## 2. 当前版本的主要特性

### 2.1 分层目录结构

项目根目录下固定分为两个子文件夹：

```text
CAL 1.1/
├─ functions/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ io_utils.py
│  ├─ sinogram.py
│  ├─ initial_projection.py
│  ├─ physics_printing.py
│  ├─ optimization.py
│  └─ visualization.py
└─ scripts/
   ├─ __init__.py
   ├─ run_initial_projection_and_print.py
   ├─ run_optimized_projection_and_print.py
   ├─ run_any_projection_and_print.py
   └─ run_point_psf_error_analysis.py
```

其中：

- `functions/`：只放功能模块；
- `scripts/`：只放运行脚本；
- 所有脚本都假定从**项目根目录**运行。

### 2.2 投影矩阵量化机制

本版本支持将投影矩阵离散为有限级数，而不是保持连续值。

离散化方式不是由用户手动指定步长 `T`，而是：

- 只指定离散索引上限 `K = projection_quantization_max_index`；
- 根据当前投影矩阵的参考最大值自动计算有效步长：

\[
T = \frac{P_{\max}}{K}
\]

然后将投影矩阵量化为：

\[
0, T, 2T, \dots, KT
\]

这样做的目的，是让“离散程度”与“投影动态范围”分离：

- 用户控制的是**级数**；
- 步长是由当前投影矩阵自动推导；
- 不额外强行限制投影矩阵的物理范围。

### 2.3 IoU 仅在容器 mask 内计算

当前版本已统一修正 IoU 计算口径：

- IoU **只在圆形容器 mask 内统计**；
- 圆外区域不参与交集 / 并集计算；
- 避免整张正方形画布引入无意义的背景区域，导致 IoU 被错误稀释。

### 2.4 投影矩阵热图可视化

投影矩阵已支持整体热图可视化：

- 横坐标：角度索引；
- 纵坐标：探测器像素索引；
- 颜色：投影强度。

因此能够直接观察全部角度、全部探测器像素的投影分布，而不是只看单角度一维曲线。

---

## 3. 物理模型与基本假设

本代码的物理模型是**单切片、二维、标量剂量累积模型**，主要包含以下假设：

### 3.1 计算域与几何

- 仿真对象为 `N × N` 的二维正方形网格；
- 有效打印区域为其中的**圆形容器**；
- 容器半径由

\[
R = \frac{N \cdot voxel\_size\_xy}{2}
\]

自动给出；

- 圆外区域默认不参与固化。

### 3.2 投影与传播

每个角度对应一条一维投影；所有角度形成一个二维投影矩阵：

- 一个维度：探测器像素索引；
- 另一个维度：投影角度索引。

前向仿真中，对每个网格点，在所有角度上进行以下步骤：

1. 计算该点在当前角度下对应的投影坐标；
2. 由投影矩阵插值得到入射强度；
3. 根据 Beer–Lambert 模型计算沿程衰减；
4. 累积所有角度的贡献得到总剂量。

### 3.3 衰减模型

采用指数衰减形式：

\[
I = I_0 e^{-\alpha d}
\]

其中：

- `α = resin_alpha`：材料衰减系数；
- `d`：从入射侧到当前点的等效路径深度；
- `I0`：投影矩阵给出的入射强度。

### 3.4 固化判定

当前版本使用**硬阈值固化模型**：

\[
\text{cured}(x,y)=
\begin{cases}
1, & \text{dose}(x,y) \ge E_c \\
0, & \text{dose}(x,y) < E_c
\end{cases}
\]

其中 `Ec = resin_Ec` 为临界固化能量。

### 3.5 当前模型未显式包含的效应

当前 CAL 1.1 **未显式建模**以下复杂过程：

- 多重散射；
- 折射与界面像差；
- 波动光学传播；
- PSF 卷积成像核；
- 材料非线性光化学动力学；
- 真实三维体打印的层间耦合。

因此，本代码更适合作为：

- 初始物理验证模型；
- 投影优化方法测试平台；
- 参数敏感性分析平台；
- 与后续更高保真模型对接的基础版本。

---

## 4. 代码结构说明

## 4.1 `functions/config.py`

负责定义全局配置类 `VAMConfig`，以及输出目录构建。

主要内容：

- 输入输出路径；
- 网格尺寸与角度数量；
- 材料参数；
- 优化参数；
- 投影离散化参数；
- 图像显示与保存开关。

关键字段包括：

- `target_image_path`：目标图路径；
- `output_dir`：输出根目录；
- `n_slice`：单切片分辨率；
- `voxel_size_xy`：单像素物理尺寸；
- `angle_num`：投影角数量；
- `resin_alpha`：衰减系数；
- `resin_Ec`：固化阈值；
- `projection_quantization_enabled`：是否开启量化；
- `projection_quantization_max_index`：离散级数上限索引。

该文件还提供：

- `get_default_config()`：返回默认配置对象；
- `ensure_output_dirs(cfg)`：创建并返回输出目录字典。

## 4.2 `functions/sinogram.py`

负责目标图预处理与初始正弦图生成。

主要函数：

### `preprocess_target_image(cfg)`

功能：

- 读取目标图像；
- 若为 RGB，则转灰度；
- resize 到 `n_slice × n_slice`；
- 二值化；
- 乘以内接圆 mask；
- 根据显示约定可选 `flipud`。

输出：

- 圆容器内的二值目标图 `target_mask`。

### `generate_initial_sinogram(target_mask, cfg)`

功能：

- 调用 `radon()` 生成目标图的正弦图；
- 对每个角度的投影列做 min-max 归一化；
- 避免极小目标在 percentile 裁剪中被压扁。

### `create_radon_filter(proj_len, filter_type)`

功能：

- 构造用于频域滤波的 ramp / hanning 组合滤波核。

## 4.3 `functions/initial_projection.py`

负责由初始正弦图生成原始投影矩阵。

主函数：

### `generate_initial_projection_matrix(sinogram, cfg)`

核心流程：

1. 生成频域滤波核；
2. 根据圆柱几何估计各探测器位置的穿透深度；
3. 用 `exp(alpha * depth)` 做反向补偿；
4. 在频域滤波；
5. 对结果做非负裁剪，避免负振铃在前向剂量中引入“负剂量”抵消。

输出为初始原始投影矩阵。

## 4.4 `functions/physics_printing.py`

负责真实打印物理仿真、剂量累计、投影量化、初始打印补偿与 IoU 计算。

主要函数包括：

### `build_container_mask(shape, cfg)`

生成与当前容器半径一致的圆形 mask，用于：

- IoU 统计；
- 点目标分析中的容器约束；
- 统一几何口径。

### `quantize_projection_matrix(...)`

将投影矩阵离散到有限级数。

关键特性：

- 不再手动指定外部步长；
- 自动根据参考最大值确定有效步长；
- 支持返回量化元数据：
  - `reference_max_value`
  - `effective_step`
  - `max_index`
  - `enabled`

### `calculate_iou(target, cured, cfg=None, container_mask=None)`

当前版本的 IoU 统一口径函数。

逻辑：

- 若给定 `cfg` 或 `container_mask`，则只在容器 mask 内计算；
- 圆外区域不参与 IoU。

### `forward_dose_simulation(projection_matrix, cfg)`

真实物理打印前向剂量仿真入口。根据配置自动选择：

- Numba 加速版本；
- Numpy 纯 Python 版本。

### `compensate_and_print_initial_projection(raw_projection, target_mask, cfg)`

对初始投影执行：

- 全局补偿；
- 量化；
- 前向剂量仿真；
- 固化判定；
- 容器内 IoU 评估。

输出包含：

- 连续投影；
- 量化投影；
- 剂量场；
- 固化图；
- IoU；
- 全局补偿系数；
- 量化有效步长等。

### `print_any_projection(projection_matrix, target_mask, cfg, ...)`

对任意载入投影矩阵执行重打印，适用于：

- `proto_projection.npz`
- `projection.npz`
- `optimized_projection.npz`
- 其他外部投影矩阵文件

的统一验证。

## 4.5 `functions/optimization.py`

负责基于剂量损失的连续投影优化。

主要思想：

1. 对当前投影矩阵做前向剂量仿真；
2. 根据目标图与当前剂量场构造损失；
3. 计算损失对剂量场的梯度；
4. 将梯度从剂量场反传到投影矩阵；
5. 使用线搜索更新投影矩阵；
6. 记录连续空间最优解；
7. **优化结束后再量化输出**。

这意味着本版本采用的是：

- **优化内部连续**；
- **最终输出离散**。

主函数：

### `optimize_projection_matrix(initial_projection, target_mask, cfg)`

输出包含：

- 最终量化后的最优投影矩阵；
- 连续最优投影矩阵；
- 量化后的剂量场；
- 量化后的固化图；
- 量化后 IoU；
- 连续最优 IoU；
- 优化历史记录。

## 4.6 `functions/io_utils.py`

负责投影矩阵及相关元数据的存储与恢复。

主要函数：

### `save_projection_npz(...)`

保存内容包括：

- `projection_matrix`
- `angles`
- `N`
- `voxel_size_xy`
- `container_radius`
- `resin_alpha`
- `resin_Ec`
- `exposure_time_per_angle_ms`
- 量化相关参数
- 可选的 `target_mask`
- 额外元数据

### `load_projection_npz(npz_path)`

读取投影矩阵及附带元数据。

### `build_config_from_metadata(cfg, metadata)`

将 `npz` 中保存的关键元数据写回配置对象，保证：

- 打印；
- 优化；
- 任意投影复现

都能在与投影文件一致的参数口径下运行。

## 4.7 `functions/visualization.py`

负责统一图像输出。

主要函数：

### `build_compare_map(target, cured)`

生成四类对比标签图：

- 0：正确未固化；
- 1：正确固化；
- 2：欠固化；
- 3：过固化。

### `_plot_projection_heatmap(proj, cfg)`

生成投影矩阵热图：

- x 轴：角度索引；
- y 轴：探测器像素索引；
- colorbar：投影强度。

### `visualize_and_save_results(...)`

统一绘制并可选保存：

- Target
- Projection heatmap
- Dose
- Cured
- Target vs Cured

并自动在 compare 图中显示：

- `IoU(container)`

---

## 5. 脚本层说明

## 5.1 `scripts/run_initial_projection_and_print.py`

作用：

- 从目标图生成初始 sinogram；
- 生成原始初始投影矩阵；
- 对原始投影做初始打印补偿；
- 对补偿结果执行量化与前向打印；
- 保存原始初始投影与补偿投影；
- 输出初始打印 IoU 和图像。

典型输出文件：

- `proto_projection.npz`：量化后的原始初始投影；
- `projection.npz`：量化后的补偿投影；
- `initial_compare.png`：对比图；
- 一组 `initial_*.png` 可视化图像。

## 5.2 `scripts/run_optimized_projection_and_print.py`

作用：

- 读取 `projection.npz`；
- 在连续空间优化投影矩阵；
- 在最终输出前进行量化；
- 保存优化后的投影矩阵；
- 输出连续最优 IoU 与量化输出 IoU；
- 保存 compare 图和各类可视化图。

典型输出文件：

- `optimized_projection.npz`
- `optimized_compare.png`
- 一组 `optimized_*.png`

## 5.3 `scripts/run_any_projection_and_print.py`

作用：

- 读取任意 `npz` 投影文件；
- 用其中 metadata 重建配置口径；
- 对该投影再次执行量化、打印和 IoU 评估；
- 用于检查某个投影文件是否能在当前代码下稳定复现。

默认可测试：

- `projection.npz`
- `optimized_projection.npz`
- `proto_projection.npz`

## 5.4 `scripts/run_point_psf_error_analysis.py`

作用：

- 人工生成点目标；
- 对点目标执行初始投影打印或优化打印；
- 分析点目标在剂量场与固化结果中的扩展、中心偏移与误差；
- 评估点扩散 / 误差传播情况。

适合用于：

- 小目标成形能力分析；
- PSF 类几何扩展评估；
- 不同参数下点目标稳定性比较。

---

## 6. 运行环境与依赖

建议环境：

- Python 3.10 及以上；
- Windows / Linux 均可；
- VS Code + Python 扩展；
- 若安装 Numba，可自动使用加速分支。

主要依赖：

- `numpy`
- `matplotlib`
- `scikit-image`
- `scipy`
- `numba`（可选）

建议安装命令：

```bash
pip install numpy matplotlib scikit-image scipy numba
```

如果不安装 `numba`，代码仍可运行，但前向剂量仿真与梯度回传会更慢。

---

## 7. 如何使用

### 7.1 先修改配置

打开：

```text
functions/config.py
```

至少检查以下字段：

- `target_image_path`
- `output_dir`
- `n_slice`
- `angle_num`
- `resin_alpha`
- `resin_Ec`
- `exposure_time_per_angle_ms`
- `projection_quantization_enabled`
- `projection_quantization_max_index`

### 7.2 从根目录运行

必须在项目根目录执行脚本，例如：

```bash
python scripts/run_initial_projection_and_print.py
python scripts/run_optimized_projection_and_print.py
python scripts/run_any_projection_and_print.py
```

不要在 `scripts/` 文件夹内部切换工作目录后直接运行，否则相对导入与路径管理更容易混乱。

### 7.3 推荐标准流程

#### 第一步：初始投影生成与补偿打印

```bash
python scripts/run_initial_projection_and_print.py
```

输出：

- 初始原始投影；
- 补偿投影；
- 初始打印 IoU；
- Dose / Cured / Compare 图。

#### 第二步：优化投影矩阵

```bash
python scripts/run_optimized_projection_and_print.py
```

输出：

- 优化后投影；
- 优化历史；
- 连续最优 IoU；
- 最终量化输出 IoU。

#### 第三步：复现任意投影文件

```bash
python scripts/run_any_projection_and_print.py
```

用途：

- 检查 `projection.npz` 或 `optimized_projection.npz` 是否能稳定复现；
- 验证某次保存的投影文件在当前参数下的打印表现。

#### 第四步：点目标误差分析

```bash
python scripts/run_point_psf_error_analysis.py
```

适合研究：

- 小目标是否会消失；
- 过固化 / 欠固化范围；
- 点目标中心漂移；
- 半高全宽或连通宽度类指标。

---

## 8. 输出文件说明

在 `output_dir` 下，程序会自动创建：

```text
output/
├─ projection_result/
├─ compare_image/
├─ figures/
└─ dose_result/
```

### 8.1 `projection_result/`

主要保存投影矩阵：

- `proto_projection.npz`
- `projection.npz`
- `optimized_projection.npz`

其中 `npz` 内通常包含：

- `projection_matrix`
- `angles`
- `N`
- `voxel_size_xy`
- `container_radius`
- `resin_alpha`
- `resin_Ec`
- `exposure_time_per_angle_ms`
- 量化元数据
- 可选的 `target_mask`
- 可选的连续版投影矩阵
- 可选的优化历史

### 8.2 `compare_image/`

保存直接可查看的 compare 图，例如：

- `initial_compare.png`
- `optimized_compare.png`
- 其他任意投影打印结果的 compare 图。

### 8.3 `figures/`

保存完整的绘图组：

- `*_target.png`
- `*_projection_heatmap.png`
- `*_dose.png`
- `*_cured.png`
- `*_compare.png`

### 8.4 `dose_result/`

可用于后续自行扩展剂量场、误差分析或中间数据保存。

---

## 9. 关于投影矩阵形状的说明

当前代码内部的投影矩阵采用二维数组表示。根据现有实现：

- 一个维度对应探测器像素；
- 一个维度对应角度索引。

在可视化热图中：

- 横坐标定义为角度索引；
- 纵坐标定义为探测器像素索引；
- 颜色表示投影强度。

因此，如果你看到的热图是“角度沿 x、像素沿 y”，那是当前版本的标准显示约定。

---

## 10. 当前版本的重要口径说明

### 10.1 IoU 口径

当前 CAL 1.1 统一采用：

- **仅在容器 mask 内计算 IoU**。

因此你看到的：

- 打印日志中的 IoU；
- compare 图标题中的 IoU；
- 优化过程中显示的 IoU；

在当前版本下应当保持一致口径。

### 10.2 优化口径

当前优化采用：

- **内部连续优化**；
- **最终输出前统一量化**。

因此：

- “最佳连续 IoU” 与 “最终量化输出 IoU” 可能不同；
- 这是正常现象，不是程序错误。

### 10.3 离散化口径

当前配置控制的是：

- **离散级数**，不是外部手动指定步长。

因此：

- `projection_quantization_max_index = 255` 表示 256 级；
- 实际有效步长会根据当前投影矩阵最大值自动计算；
- 不建议再人为额外指定 `T`，否则会错误耦合投影范围。

---

## 11. 常见使用建议

### 11.1 先用较小参数验证流程

若只是检查代码链路是否正确，建议先用较小参数，例如：

- 降低 `n_slice`
- 降低 `angle_num`
- 降低 `num_iterations`

这样能更快判断：

- 导入是否正确；
- 输出目录是否正常；
- IoU 口径是否一致；
- 量化是否生效。

### 11.2 再回到正式分辨率

确认流程稳定后，再恢复：

- 1080 分辨率；
- 720 角度；
- 正式材料参数。

### 11.3 优先检查 compare 图与 heatmap

在调试时，建议优先看：

- `projection_heatmap`
- `dose`
- `compare`

因为这三张图最能快速定位问题是出在：

- 投影分布；
- 剂量累积；
- 欠固化 / 过固化。

---

## 12. 常见问题

### Q1. 为什么 Pylance 报 `无法解析导入 functions`？

通常是因为：

- 没有从项目根目录打开 VS Code；
- 或者运行时工作目录不在根目录；
- 或者 `functions/` 下缺少 `__init__.py`。

当前版本采用的是标准分层结构，只要在根目录下运行：

```bash
python scripts/run_initial_projection_and_print.py
```

一般就不会有问题。

### Q2. 为什么优化后的连续 IoU 和最终输出 IoU 不一样？

因为优化是在连续空间中进行，而最终保存的投影矩阵会被量化到有限级数。

所以：

- 连续最优解是理论最优；
- 量化后结果更接近真实离散投影硬件输出。

### Q3. 为什么 IoU 比旧版本低或高？

常见原因有：

1. IoU 统计口径已经改成“仅容器内”；
2. 当前版本的投影矩阵被离散化；
3. 目标图预处理方向或阈值不同；
4. `Ec`、`alpha`、`angle_num`、`n_slice` 等参数变化。

### Q4. 为什么小目标可能不固化？

可能原因包括：

- 目标太小，radon 列归一化后仍然总能量不足；
- `Ec` 过高；
- `alpha` 过大导致深部剂量衰减严重；
- 离散化后有效投影级数不够；
- 频域滤波后投影能量过分集中或过分稀疏。

这也是 `run_point_psf_error_analysis.py` 存在的意义之一。

---

## 13. 当前版本的适用范围

CAL 1.1 适合用于：

- 初始投影生成策略验证；
- 离散投影级数对成形结果影响分析；
- 容器内 IoU 口径统一比较；
- 目标图到固化结果的几何误差研究；
- 点目标扩展与小结构成形能力分析；
- 后续更高保真仿真版本的前置验证。

它不应被直接理解为：

- 完整真实三维 VAM 光场传播求解器；
- 含折射 / 散射 / PSF / 非线性材料反应的高保真模型；
- 可直接替代实验系统标定的终版模型。

---

## 14. 建议的版本管理方式

建议你在继续开发时，严格记录以下信息：

- `config.py` 中的关键参数；
- 每次投影文件对应的 `npz metadata`；
- 目标图名称；
- compare 图与 heatmap；
- 优化历史与最终量化输出 IoU。

这样后续对比不同版本时，才能明确区分：

- 是物理参数改变导致结果变化；
- 还是算法改动导致结果变化；
- 还是量化级数或 IoU 口径变化导致结果变化。

---

## 15. 后续开发建议

后续若继续升级，可以优先考虑以下方向：

1. 引入更明确的投影矩阵方向约定与断言；
2. 将 compare 图、dose 图、中间剂量结果统一保存为 `npz + png` 双格式；
3. 在优化中增加更多损失权重与正则项；
4. 引入更高保真的传播模型；
5. 将点目标分析扩展到线目标、双点目标和复杂微结构。

---

## 16. 一句话总结

CAL 1.1 是一套面向**单切片体投影光固化仿真**的分层代码框架，当前版本已经明确实现：

- `functions / scripts` 分层；
- 投影矩阵有限级数量化；
- IoU 仅在容器 mask 内统计；
- 投影矩阵整体热图可视化；
- 初始打印、优化打印、任意投影复现和点目标误差分析四条主链路。

在此基础上，你可以继续把它作为 `仿真1.0 / CAL 1.1` 的稳定中间版本，向更复杂的物理模型和实验系统对接推进。
