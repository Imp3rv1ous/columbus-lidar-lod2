# Columbus LiDAR → LOD2 建筑重建管线

从机载 LiDAR 点云重建 LOD2 建筑模型的研究代码。研究区为俄亥俄州哥伦布近东区
2.76 km²，数据为 USGS 3DEP 2019 B19 的 19 块 LAZ 瓦片，76,351,005 点，约 30 pts/m²。

> 本仓库只包含**源代码、方法文档和结果统计表**。原始 LiDAR 数据、模型权重和大体积
> 点云产物不在仓库内（见下方「仓库不含的内容」）。

## 管线状态

| 步骤 | 名称 | 状态 | 目录 |
|---|---|---|---|
| 0 | 原始数据 | ✅ | `roof_proj_code/00_LiDAR_data/` |
| 1 | 语义分割（RandLA-Net） | ✅ | `roof_proj_code/01_semantic_randlanet/` |
| 2 | 建筑单体分割 | ✅ | `roof_proj_code/02_03_instance_seg/` |
| 3 | 噪声清除（Gate A） | ✅ | 同上 |
| 4 | 屋顶平面提取 | ✅ | `roof_proj_code/04_05_roof_planes/` |
| 5 | 结构线索层（脊/谷/step） | ✅ | 同上 |
| 6 | 轮廓正则化 | ✅ | `roof_proj_code/06_outline_reg/` |
| 7 | 模型装配（LOD2 几何） | ⬜ | 计划中 |
| 8 | 语义化导出（CityJSON） | ⬜ | 计划中 |
| 9 | 精度评价 | ⬜ | 计划中 |

详细方法说明见 [`roof_proj_code/PIPELINE.md`](roof_proj_code/PIPELINE.md)，各步骤另有独立 README。

## 关键结果

- **语义分割**：mIoU 0.977（checkpoint 选择指标，非独立地理测试集）；建筑点 8.6M（11.3%）
- **单体分割**：2,296 个实例，分水岭拆开 168 个粘连组件
- **Gate A 清洗**：2,267 栋干净实例，13 个非建筑目标带原因码剔除
- **屋顶平面**：33,845 个平面（屋顶 26,643 + 墙 7,202），点分配率 91.2%
- **结构线索**：脊 8,830 / 谷 3,963 / step 14,111
- **轮廓正则化**：2,253/2,261 成功（99.65%），顶点中位数 7，偏移中位数 0.15 m

## 方法文档

- [`lecture_02_03_instance_seg/main.pdf`](lecture_02_03_instance_seg/main.pdf) — 步骤 2–3 详细方法讲义
- [`code_walkthrough_02_03/main.pdf`](code_walkthrough_02_03/main.pdf) — 步骤 2–3 源码逐行讲解（含玩具管线 `toy_pipeline.py`）
- [`progress_ppt/progress_report_instance_seg.pdf`](progress_ppt/progress_report_instance_seg.pdf) — 阶段汇报
- [`roof_proj_code/output/pdf/`](roof_proj_code/output/pdf/) — 源代码级方法教程

## 仓库不含的内容

以下内容因体积或版权原因未上传：

- **原始与中间点云**（`.laz` / `.las` / `.ply`，约 1.9 GB）：USGS 3DEP 数据可从
  [USGS 3D Elevation Program](https://www.usgs.gov/3d-elevation-program) 公开下载
- **模型权重**（`results/best.pth`，60 MB）
- **第三方参考代码与论文**（原 `ref/`）：请从各自的官方来源获取

## 环境

各步骤的依赖见 `roof_proj_code/01_semantic_randlanet/requirements.txt`。
点云处理主要依赖 `laspy`、`numpy`、`scipy`、`scikit-image`、`shapely`；
分割网络依赖 `torch`。

## 数据来源

USGS 3D Elevation Program (3DEP), Columbus 2019 B19。
CRS：NAD83(HARN) / Ohio South (ftUS) + NAVD88 (Geoid12B)。
