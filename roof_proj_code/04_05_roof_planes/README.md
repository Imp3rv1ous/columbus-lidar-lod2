# 步骤 4–5：屋顶平面提取 + 结构线索层

输入：`02_03_instance_seg/output_v2/buildings_clean.laz`（2,283 栋）
输出：逐点 plane_id 的点云、平面参数表、特征线（脊/谷/step）、轮廓、拓扑图。

## 步骤 4 · 屋顶平面提取（plane_extraction.py + full_run.py）

方法：**三遍法 region growing**
1. 主遍：K=20 邻域 PCA 法向/曲率 → 低曲率种子生长（法向角 15°、点面距 0.15 m、
   最小支撑 55 点 ≈ 2 m²）；
2. 多尺度遍：对残余点用 K=10 小邻域重新生长（最小 20 点 ≈ 0.7 m²、RMS ≤ 4 cm）——
   召回老虎窗等小面（小面的 20-NN 邻域会跨面污染法向，这是它们漏检的根因）；
3. 无法向遍：残余点按空间连通分块（0.7 m 链接）后**直接做带修剪的 PCA 平面拟合**
   （12 点 ≈ 0.4 m² 下限、RMS ≤ 3.5 cm、inlier 取最大连通分量、直径 ≤ 8 m）——
   召回 <1 m 宽的窄条外凸面（任何 kNN 法向在窄条上都会被污染，故绕过法向）。
合并：两级 RMS 门控（大-大 8°/偏移 ≤10 cm——**10 cm 就是 step 保护阈**；小并大 20°，
联合拟合 RMS 不涨才真合并）。墙/屋面按法向 |nz|<0.34 分离。

全量结果：33,845 面（屋顶 26,643 + 墙 7,202），2,261 栋，平均点分配率 91.2%，
分配率 <70% 的 18 栋已在 `building_quality.csv` 标记（Gate C）。

### 论文依据
- 区域生长平面分割：**Rabbani, T., van den Heuvel, F., Vosselman, G. (2006).**
  Segmentation of point clouds using smoothness constraint. *IAPRS* 36(5):248–253。
  ——City3D 与 3D BAG 两条生产线采用的同款方法。
- 最小平面支撑按密度换算（8 pts/m² 下 15 点 → 本区 30 pts/m² 下 ~55 点）：
  **Peters, R. et al. (2022).** *PE&RS* 88(3)（3D BAG）。
- 消融对照组（高度聚类 δ_h=1 m + 顺序 RANSAC ζ=0.1 m）：**Jung, J., Jwa, Y., Sohn, G.
  (2017).** Implicit Regularization for Reconstructing 3D Building Rooftop Models Using
  Airborne LiDAR Data. *Sensors* 17(3):621；RANSAC：**Fischler, M., Bolles, R. (1981).**
  *CACM* 24(6)。15 栋同条件对比：空间不连贯"幻影平面" 0（本法）vs 41（RANSAC 路线），
  见 output_pilot/rg_vs_jung.png 与 rg_vs_jung_stats.csv。
- 墙点按法向剔除的约定：Peters et al. 2022；替代方案 DoPP 阈值见 Li et al. 2026（City-BIS）。

## 步骤 5 · 结构线索层（cue_layer_full.py）

方法：
- **脊/谷线**：3D kNN 相邻、法向角 >12° 的屋面对 → 两平面**解析交线**，按边界点投影
  范围截取线段；交线高于两面质心为脊、低于为谷；近平行共线段做束合并（方向 <10°、
  线距 <0.8 m、沿线有重叠）。
- **step 线**：屋面对的 0.5 m 栅格膨胀相交处，用两平面方程求高差，中位 |Δz| ≥ 0.25 m
  判 step，沿接触带 PCA 拟合线段。（3D kNN 会被错层间的墙点隔断——2D 检测是全量
  14,111 条 step 能被检出、而试点为 0 的原因。）
- **轮廓**：屋顶掩膜 marching squares 边界 + Douglas–Peucker 简化（正则化在步骤 6）。
- **拓扑**：面为节点、脊/谷/step 关系为边（edges.csv + planes.csv = roof topology graph）。

### 论文依据
- "建模线索"分类框架（交线 / step 线 / 外边界三类线索 + 拓扑恢复）：**Jung, Jwa & Sohn
  (2017)**（同上）——本层的概念蓝本；其"高度簇边界即 step 线"的洞见对应本实现的
  高差判据。
- 屋顶拓扑图（面邻接关系用于重建）：**Verma, V., Kumar, R., Hsu, S. (2006).** 3D Building
  Detection and Modeling from Aerial LIDAR Data. *CVPR 2006*；**Oude Elberink, S.,
  Vosselman, G. (2009).** Building Reconstruction by Target Based Graph Matching on
  Incomplete Laser Data. *Sensors* 9(8):6101。
- 轮廓简化：**Douglas, D., Peucker, T. (1973).** *Cartographica* 10(2)。轮廓正则化（步骤 6
  计划）：**Sampath, A., Shan, J. (2007).** Building Boundary Tracing and Regularization
  from Airborne Lidar Point Clouds. *PE&RS* 73(7)；**Huang, J., Stoter, J., Peters, R.,
  Nan, L. (2022).** City3D. *Remote Sensing* 14(9):2254 的 regularity enhancement。

## 文件清单
| 文件 | 内容 |
|---|---|
| output_full/buildings_planes_rgb.laz | 逐点 plane_id / surf_type，逐面防撞色 |
| output_full/planes.csv | 33,845 行平面参数（法向、质心、面积、RMS、坡度、方位角） |
| output_full/building_quality.csv | 逐栋质量表（Gate C：面数、分配率、RMS、屋顶形态、标记） |
| output_full/cues.csv / edges.csv / outlines.csv | 特征线 / 拓扑边 / WKT 轮廓（ftUS） |
| output_full/cues_ridge.obj, cues_valley.obj, cues_step.obj, outlines.obj | CloudCompare 叠加线框（按类型分文件便于分别上色/开关） |
| output_pilot/ | 方法选型消融材料（RG vs Jung） |

注意：本层的线是**未正则化的原始几何证据**——歪斜、长短不齐属预期状态，
步骤 6（正则化）与步骤 7（装配为闭合面片）才是它们的"成品形态"。

## 植被处理记录与后续过滤方案

**整实例剔除（已应用）**：22 个树冠误分类实例经 class 5 植被上下文 + 第 4 步平面性
反馈 + 逐个人工目验后剔除（Gate A 追溯判定），点云存档于
`02_03_instance_seg/output_v2/veg_reclassified.laz`；6 个被规则误标的真实结构
（树下棚屋、单坡车棚等）在目验中救回。

**平面级几何过滤（已回滚）**：曾按"植被上下文比 + 面下方植被证据"删除树冠假面，
但该规则在屋顶边缘失效（膨胀邻域伸出屋檐后误杀贴树房屋的真实边缘面），用户
复核确认后整体回滚。教训与判决记录保留在 veg_filter.py / plane_veg.csv。

**后续阶段过滤方案（计划，依据文献与实测）**：改用**回波（echo）特征**在面级判别
——本区数据实测：植被点 94.8% 为中间回波（return_number < number_of_returns），
建筑点仅 10.4%；逐面统计中间回波占比即可稳健分离树冠假面（阈值≈0.5），且不依赖
几何邻域、无屋檐边缘失效问题。文献依据：echo ratio 线的城市植被检测（Rutzinger,
Höfle, Hollaus, Pfeifer 2008，object-based point cloud analysis；Höfle, Hollaus,
Hagenauer 2012, *ISPRS JPRS* 67, 全波形分段特征分类，植被正确率 ~98%）。
互补判据：装配阶段的**拓扑孤立性**——真屋面经脊/step/轮廓关系连入屋顶拓扑图，
树冠假面是孤立节点，剖分认领时自然落选（对应"密封性"直觉）。
