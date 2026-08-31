# Columbus LiDAR → LOD2 管线索引（汇报用）

研究区：哥伦布近东区 2.76 km²，19 块 USGS 2019 LAZ，76.35M 点，~30 pts/m²。
每步的文件夹以序号开头；"状态"列 ✅=已完成并验收，🔄=试点完成，⬜=未开始。

| 步骤 | 名称 | 状态 | 文件夹 / 关键文件 |
|---|---|---|---|
| 0 | 原始数据 | ✅ | `00_LiDAR_data/`（USGS 原始瓦片 + 训练标注） |
| 1 | 语义分割（RandLA-Net） | ✅ | `01_semantic_randlanet/results/preds/`（19 块，class 2 地面 / 5 植被 / 6 建筑） |
| 2 | 建筑单体分割 | ✅ | `02_03_instance_seg/instance_segmentation_v2.py` → `output_v2/buildings_instances_rgb.laz`（2,296 实例，防撞色 RGB） |
| 3 | 噪声清除（Gate A） | ✅ | 同文件夹：`gate_a_filter.py`、`gate_a2_shape.py` → `output_v2/buildings_clean.laz`（2,267 栋；含 22 个树冠实例追溯剔除与 6 个欠分割实例的证据拆分）、`instances_gateA.csv`（含 13 个剔除的原因码） |
| 4 | 屋顶平面提取 | ✅ | `04_05_roof_planes/plane_extraction.py`（v4 三遍法）+ `full_run.py` → `output_full/buildings_planes_rgb.laz`、`planes.csv`（33,849 面）、`building_quality.csv` |
| 5 | 结构线索层 | ✅ | 同文件夹：`cue_layer_full.py` → `output_full/cues.csv`（脊 8,830 / 谷 3,963 / step 14,111）、`edges.csv`（拓扑）、`outlines.csv`（WKT 轮廓）、`cues_lines.obj`（CloudCompare 叠加查看） |
| 6 | 轮廓正则化 | ✅ | `06_outline_reg/regularize_outlines.py` → `output/outlines_regularized.csv`（2,253/2,261 成功，顶点中位 7，偏移中位 0.15 m，正交占比 0.82）+ `outlines_regularized.obj` |
| 7 | 模型装配（LOD2 几何） | ⬜ | 计划：`07_assembly/`（轮廓剖分 + 落墙 + DTM 底面 → 水密 solid；City3D 与自研双轨） |
| 8 | 语义化导出 | ⬜ | 计划：`08_export/`（CityJSON：Roof/Wall/GroundSurface + 属性） |
| 9 | 精度评价 | ⬜ | 计划：`09_evaluation/`（30–50 栋分层人工标注 + 点到模型 RMSE） |

## 每步的方法一句话 + 汇报要点

1. **语义分割**：RandLA-Net 逐点四分类；建筑点 8.6M（11.3%）。
2. **单体分割**：0.5 m 栅格（无闭运算）→ 8 连通域 → 腐蚀取核 + marker 分水岭（以 −z_max 为地形，边界落在楼间高度谷）→ 非对称碎片吸收（防过分割）。要点：闭运算会桥接 1.5–2.5 m 的侧院间隙（v1 教训）；分水岭拆开 168 个粘连组件。
3. **Gate A**：DTM（class 2）+ 逐实例离地高度 + 形状规则（凸包充实度、EDT 内切圆宽度）捞候选 → 人工判读剔除 13 个非建筑（桥面、匝道折带、树列、电力线）。要点：桥/匝道平面性好，重建阶段不会失败，必须在此清除；全程原因码可审计。
4. **屋顶平面**：region growing 三遍法——主遍（K=20，55 点）+ 多尺度（K=10，20 点，救老虎窗）+ 无法向连通块拟合（12 点 ≈ 0.4 m²，救窄条外凸面）；两级 RMS 门控合并（10 cm step 保护内生）。要点：与 Jung 2017（高度聚类+RANSAC）15 栋消融——幻影平面 0 vs 41；分配率 90.8%；识别下限 0.4 m² 是 30 pts/m² 的物理极限。
5. **线索层**：脊/谷 = 相邻面解析交线（按边界点范围截取，平行脊束合并）；step = 2D 栅格邻接 + 平面方程高差 ≥0.25 m（3D kNN 会被错层间的墙点隔断，2D 检测是关键）；轮廓 = marching squares；拓扑 = 面邻接图。要点：LOD2 模型没有点、只有由这些线围成的面——本步产出的就是模型的"骨架"，也是论文定量评价的对象。

## 质量三道闸（贯穿设计）

Gate A（循环前，实例级）清假建筑 → Gate B（平面提取内生）树点/立面点因不共面被自动排除 → Gate C（`building_quality.csv`）逐栋分配率/RMS/形态标记，<70% 分配率的 31 栋已标记待复查。
