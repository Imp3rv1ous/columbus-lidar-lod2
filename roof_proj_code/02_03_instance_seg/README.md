# 步骤 2–3：建筑单体分割 + 语义噪声清除（Gate A）

输入：`01_semantic_randlanet/results/preds/`（19 块，class 2 地面 / 5 植被 / 6 建筑）
输出：`output_v2/buildings_clean.laz` —— 2,283 栋干净建筑实例，进入屋顶重建的正式输入。

## 步骤 2 · 单体分割（instance_segmentation_v2.py）

方法：0.5 m 占据栅格（每格 ≥2 点，**不做闭运算**）→ 8 连通域 → 对仍粘连的组件做
圆盘腐蚀取核（r=1.5 m）+ **marker 控制分水岭**（以 −z_max 高度面为地形，边界自动落在
两楼之间的高度谷底）→ 非对称碎片吸收（<20 m² 碎片并入 3 m 内高程相容的 ≥50 m² 实例；
"只许小并大"，防止制造新的欠分割）。结果 2,296 实例；168 个粘连组件被分水岭拆开。

### 论文依据
- 栅格化 + 形态学的建筑制图路线：**Song, H., Jung, J. (2023).** An Unsupervised, Open-Source
  Workflow for 2D and 3D Building Mapping From Airborne LiDAR Data. *IEEE JSTARS*.
- marker 分水岭分离弱连接目标（经典方法）：**Beucher, S., Meyer, F. (1993).** The morphological
  approach to segmentation: the watershed transformation. In *Mathematical Morphology in
  Image Processing*.
- 同问题的图聚类替代方案（对照/兜底）：**Li, J., Li, Y., Chen, Y., Fan, H., Wang, R. (2026).**
  City-scale building instance segmentation from LiDAR point clouds via structure-aware
  method (SI-DVDC). *Int. J. Applied Earth Observation and Geoinformation* 146:105086。
- 楼间距≈点间距时纯连通域失效的界定：**Yang, W. et al. (2022).** Object-based building instance
  segmentation from airborne LiDAR point clouds. *Int. J. Remote Sensing* 43(18)。
- 路线选型背景（footprint-guided 为生产标准、独栋场景聚类可行）：**Vosselman, G., Dijkman, S.
  (2001).** 3D building model reconstruction from point clouds and ground plans. *IAPRS*
  XXXIV-3/W4；**Peters, R. et al. (2022)**（见下）。本区为独栋住宅为主，选 footprint-free 路线，
  Franklin County 官方 footprint 保留作精度评价基准（未入管线）。

### 教训记录
v1 用 1 m 栅格 + 3×3 闭运算：闭运算桥接了 1.5–2.5 m 的侧院间隙，把整列独栋串成一条
（欠分割）。v2 移除闭运算（屋顶内部空洞改用逐组件 fill-holes，不会向外桥接）后解决。

## 步骤 3 · Gate A 噪声清除（gate_a_filter.py / gate_a2_shape.py / finalize_gate_a*.py)

方法：class 2 地面点建 1 m DTM（逐格最小值 + 中值滤波 + 最近邻补洞）→ 每实例
**离地高度** = 屋顶 z50 − 外扩 4 m 环内 DTM 的 P5 → 规则打标签（离地 <2 m 剔除；
细长高架体、低充实度大折带、骨架细长体等**只标记候选、人工判读后剔除**）。
共剔除 13 个非建筑（4 桥面/走廊构筑物、3 匝道折带、1 坡道、1 树列、2 电力线、2 其他），
全部带原因码记录在 `instances_gateA.csv`，被删点保存于 `noise_removed.laz` 可审计。

### 论文依据
- 每栋落地高取"轮廓外扩缓冲区内地面点 P5"的约定：**Peters, R., Dukai, B., Vitalis, S.,
  van Liempt, J., Stoter, J. (2022).** Automated 3D Reconstruction of LoD2 and LoD1 Models
  for All 10 Million Buildings of the Netherlands. *PE&RS* 88(3):165–170（3D BAG）。
- 设计原则"桥梁/道路平面性好、重建阶段不会自我暴露，必须在实例级用离地高度+形状清除"：
  由 **Huang et al. (2022, City3D)** 的重建机理与上述生产管线共同支撑（重建器无法区分
  几何同构的桥与楼）。

## 文件清单
| 文件 | 内容 |
|---|---|
| output_v2/buildings_instances_rgb.laz | 全部实例（防撞色 RGB + instance_id） |
| output_v2/buildings_clean.laz | 剔除噪声后的 2,283 栋（下游正式输入） |
| output_v2/instances_gateA.csv | 逐实例属性 + 离地高度 + 判定原因码 |
| output_v2/qa_*.png | 各环节审计图 |
| diagnose_coherence.py | 实例空间连贯性审计（验证无"远距离同 ID"） |

## 回波预过滤的评估记录（负结果，2026-08-29）

设想：在 instance 化前按 return_number < number_of_returns 直接删除中间回波点，
从源头去除树冠误分类。实测评估后**决定不采用**，两条证据（qa_echo_check.png）：
1. 代价：干净人字顶住宅损失 9.3% 的点，且**全部集中为屋檐一圈**（激光束跨屋檐
   分裂：屋檐回波是"第 1 次/共 2 次"，也是中间回波）——直接侵蚀外轮廓这一核心特征；
2. 收益：真实树团污染只能删掉 16%——存在**选择偏差**：RandLA-Net 恰好把回波上
   "像表面"的树冠点（单次/末次回波）误分为建筑，被误分的子集天然缺乏中间回波特征。
   （全体 class 5 植被点 94.8% 为中间回波，但那是未被误分的大多数。）
全域 class 6 构成：单次 84.5%、多次末回波 5.6%、中间回波(2次) 9.5%、中间回波(≥3次) 0.5%。
结论：回波保留为后续**面级**过滤的特征之一（剔除屋檐环后统计内部中间回波占比），
主兜底为装配阶段的拓扑孤立性。文献：Rutzinger et al. 2008；Höfle et al. 2012 (ISPRS JPRS)。
