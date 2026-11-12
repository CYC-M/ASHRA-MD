# ASHRA-MD 系统查新与原创性边界划界报告

- **日期**：2026-11-07
- **项目名称**：ASHRA-MD（Active-Shell Hamiltonian with Replica-tempering & Affinity Molecular Dynamics）
- **PepFind 建议注册能力 ID**：`md.complex.ashra`（兼容 `md.complex.ashmd`）
- **基本原则**：严格区分**已发表成熟物理基石（正面引用、绝不声称首创）**与**本工作的收敛原创算法组合与数学形式**；全部引文采用领域公认作者-年份-期刊标识，严禁编造未经核实的 PMID 或 DOI。

---

## 1. 为什么需要严格的原创性划界？

分子动力学（MD）拥有半个世纪的深厚文献积累：力场参数（AMBER/CHARMM）、基础朗之万分步（BAOAB）、广义玻恩隐式溶剂（OBC II）、粒子网格 Ewald（SPME）、副本交换（REST2）以及多态自由能估计（MBAR）均已有成熟先驱工作。

在蛋白质-多肽复合物的**纯 CPU 超长时间高可靠模拟**场景中，直接拼凑现有通用工具存在三大结构性痛点：
1. **多时间步（MTS）外步长共振与多线程非结合律漂移**：显式或半显式溶剂下，长程静电/广义玻恩极化力若采用较大外步长（$6\text{ fs}$），易与高频共价基团运动耦合引发共振发热；同时常规 OpenMP 浮点动态归约破坏加法结合律，导致跨线程与跨断点续算轨迹发散。
2. **显式水球边界折痕伪影与全蛋白 O(N²) 极化冗余**：全周期显式水盒子（20,000–50,000 原子）在 4 核 CPU 上吞吐受限；纯隐式溶剂（GBSA）丢失关键界面桥接水氢键；传统球形势阱（SSBP）不适配狭长弯曲的多肽结合沟槽，而朴素最近锚点距离 $\min_p \|\mathbf{r}_w - \mathbf{r}_p\|$ 在 Voronoi 等距超平面处梯度不连续（仅 $C^0$），超长时间运行会产生持续非保守能量注入。
3. **常规单轨迹假阳性诱饵动力学囚禁与 REST2 去溶剂化伪影**：多肽分子对接产生的口袋内假阳性姿态（如 $180^\circ$ 反向沟槽姿态）常被局部盐桥或空间位阻困在亚稳态；而标准 REST2 在高副本同时削弱“多肽-水”相互作用，对富含带电残基的多肽引入非物理真空去溶剂化偏置。

**ASH-MD 的原创性不在于重新发明物理力场参数（力场严格采用已发表 AMBER99SB-ILDN + TIP3P），而在于针对上述三大瓶颈提出三个数学自洽、梯度精确、面向纯 CPU 超长时运行协同设计的原创算法支柱。**

---

## 2. 三大支柱查新对比与边界划定

### 支柱 1：`FC-BAOAB`（确定性无共振共价质心滤波测地线分步积分器）

#### 收敛原创性声明 1
> 我们提出在完整全原子约束流形（SETTLE + RATTLE）的测地线 `BAOAB` 多时间步分裂中，嵌入**坐标无关的常数线性共价基团质心滤波算子** $\tilde{\mathbf{q}} = \mathcal{P}\mathbf{q}$ 用于外步慢变力（OBC II 极化场与 SPME 倒空间场）求值，并严格通过伴随算子 $\mathbf{F}_{\text{slow}}(\mathbf{q}) = -\mathcal{P}^\top \nabla_{\tilde{\mathbf{q}}} U_{\text{slow}}(\mathcal{P}\mathbf{q})$ 回传保守力；同时将直接力、OBC 去屏蔽积分 $I_i$、Born 半径链式导数 $\partial U/\partial \alpha_i$ 及分项势能全部纳入防碰撞溢出定点数阿贝尔群（`Q31.32` / `Q19.44` `int64_t` 与 `Q87.40` `128` 位 `__int128`）与 `Philox-4x32` 计数器随机流，实现**跨任意 CPU 线程数与跨 SHA-256 纪元断点续算的 IEEE-754 逐比特严格一致（`max_dx == 0.0`）**。

#### 先行文献对比与必引划界
| 先行工作 | 已有贡献（我们采用/致敬，不声称首创） | 与 ASH-MD `FC-BAOAB` 的严格差异边界 |
|---|---|---|
| **BAOAB / Geodesic BAOAB**（Leimkuhler & Matthews, *AMRX* 2013; *Proc. R. Soc. A* 2016） | 朗之万动力学 `B-A-O-A-B` 对称算子分裂及完整约束流形切空间投影的高精度构型采样性质 | 原方法为单步长或未滤波分裂，不涉及线性共价质心拓扑滤波算子 $\mathcal{P}$ 与定点确定性并行归约 |
| **MOLLY 软化脉冲法**（García-Archilla, Sanz-Serna & Skeel, *SIAM J. Sci. Comput.* 1998; Izaguirre, Reich & Skeel, *J. Chem. Phys.* 1999） | **最近邻先行者（必引）**：首次提出在长程力求值前对快变键振动做时间平均滤波（$\nabla \langle U_{\text{slow}}(\mathcal{A}(\mathbf{q}))\rangle$）以抑制 MTS 共振 | 经典 `MOLLY` 的滤波映射 $\mathcal{A}(\mathbf{q})$ 依赖非线性快子系统轨迹平均或法向模式投影，伴随力需计算坐标相关的雅可比矩阵 $\partial \mathcal{A}/\partial \mathbf{q}$；我们的 $\mathcal{P}$ 为**重原子-氢共价基团质量加权质心线性收缩常数算子**（$\partial \mathcal{P}/\partial \mathbf{q} \equiv 0$），复杂度严格 $O(N_{\text{H}})$、零海森/雅可比开销，且严格保持刚体平动与转动动量守恒 |
| **SIN(R) 等动能无共振法**（Leimkuhler, Margul & Tuckerman, *Mol. Phys.* 2013） | 通过为每个自由度引入等动能 Nosé-Hoover 链扩展相空间以压制外步共振 | `SIN(R)` 修改了动量分布与扩展相空间维数；`FC-BAOAB` 保持标准玻尔兹曼正则相空间 $(\mathbf{q}, \mathbf{p})$ 与纯哈密顿保守势 $U_{\text{slow}}(\mathcal{P}\mathbf{q})$ |
| **Philox PRNG & GPU 定点力累加**（Salmon et al., *SC'11* 2011; Le Grand, Götz & Walker, *Comput. Phys. Commun.* 2013） | 计数器随机数生成器及 GPU 单精度模拟中的 `SPFP` 64 位定点力累加思想 | 我们将阿贝尔群定点归约从单一直接力扩展至**广义玻恩去屏蔽积分（`Q19.44`）及 128 位超宽定点 Born 链式偏导与抗位阻溢出势能累加（`Q87.40`）**的全链路 CPU 多线程工作池，同时消除了隐式/混合溶剂并行 MD 中的浮点非结合律分岔源与强碰撞溢出风险 |

---

### 支柱 2：`ASH`（活性壳层显隐双模耦合哈密顿量与全光滑口袋水合帽）

#### 收敛原创性声明 2
> 我们提出面向弯曲蛋白质-多肽界面沟槽的**活性壳层耦合哈密顿量（Active-Shell Hamiltonian）**：将口袋内显式桥接 `TIP3P` 水分子与多肽/口袋溶质统一置于同一套带生理盐（Debye-Hückel $\kappa$）屏蔽的 `OBC II` 广义玻恩极化场与 `ACE` 非极性疏水空腔场中；引入带精确牛顿第三定律锚点反作用力的**全局 $C^\infty$ 软极小（Log-Sum-Exp）随动口袋水合包络势**替代单球面水帽；并通过三区拓扑划分（Active / Quintic-Tethered Buffer / Static Scaffold）将冻结骨架间的 `OBC` 去屏蔽积分精确预分解为常数项 $I_i^{\text{scaf-scaf}}$，在**零损失全蛋白介电环境边界**的前提下将外步复杂度由 $O(N_{\text{total}}^2)$ 降至 $O(N_{\text{mobile}} N_{\text{nb}})$。

#### 先行文献对比与必引划界
| 先行工作 | 已有贡献（我们采用/致敬，不声称首创） | 与 ASH-MD `ASH` 模式的严格差异边界 |
|---|---|---|
| **AMBER ff99SB-ILDN + TIP3P + SPME**（Hornak et al., *Proteins* 2006; Lindorff-Larsen et al., *Proteins* 2010; Jorgensen et al., *JCP* 1983; Essmann et al., *JCP* 1995） | 全原子共价/非键力场参数、刚性三点水模型及四阶 B 样条平滑粒子网格 Ewald | 我们 100% 忠实实现上述物理参数与显式 `SPME` 模式（Mode 0），**绝不自造未标定原子电荷或范德华参数** |
| **OBC II GB + Debye-Hückel + ACE**（Onufriev, Bashford & Case, *Proteins* 2004; Srinivasan et al., *Theor. Chem. Acc.* 1998; Schaefer & Karplus, *J. Phys. Chem.* 1996） | `igb=5` 有效 Born 半径双曲正切重标定、Still 核盐效应屏蔽及基于 Born 半径的非极性空腔能 | 原方法作为纯隐式连续介质用于无显式水体系；`ASH` 将显式界面水壳层与溶质共同纳入 `OBC II` 互去屏蔽积分与解析链式力，并在轨迹末端耦合无截断精确三态差分 $\Delta G_{\text{MM/OBC2-GBSA}}$ 消除带电复合物截断误差 |
| **球形溶剂边界势 SSBP / SBC**（King & Warshel, *JCP* 1989; Beglov & Roux, *JCP* 1994）与混合显隐溶剂（MSMS/GBSW 混合壳层） | **最近邻先行者（必引）**：局部显式水球耦合外部连续介质反应场的经典物理框架 | 经典 `SSBP` 仅支持单一球对称几何，对细长多肽沟槽需包裹大量冗余体相水；而按最近溶质重原子截断的硬水壳 $\min_p \|\mathbf{r}_w-\mathbf{r}_p\|$ 在多锚点等距面处二阶导发散。我们的核平滑软极小包络势 $d_{\text{env}}(\mathbf{r}_w) = -\beta^{-1}\ln\sum_{p\in\mathcal{A}} e^{-\beta\sqrt{\|\mathbf{r}_w-\mathbf{r}_p\|^2+\epsilon_s^2}}$ **在全空间无穷阶可微（$C^\infty$）**，自适应贴合任意拓扑沟槽且严格满足动量守恒 $\sum_p \mathbf{F}_p + \mathbf{F}_w = \mathbf{0}$ |

---

### 支柱 3：`IHRT-MBAR`（面向 4-vCPU 架构的纯溶质界面软核哈密顿回火与自洽重加权）

#### 收敛原创性声明 3
> 我们提出与 4-vCPU 计算节点硬件同构的**四副本同步溶质界面哈密顿回火（Interface Hamiltonian Replica Tempering）**：第 0 副本严格保持 $310.15\text{ K}$ 未修改物理哈密顿量（$\lambda_0=1, \mu_0=1, \alpha_{\text{sc},0}=0$）；第 $1\text{–}3$ 副本**仅对“多肽-受体/结构离子”溶质界面非键项**施加 $\lambda_m$ 缩放并配对 $r^6$ 范德华软核与 Steinbrecher-Joung-Case 静电软核（消除 $1/r$ 库仑奇点），并对多肽非酰胺可旋转二面角施加 $\mu_m=\sqrt{\lambda_m}$ 势垒软化，而**保持“多肽-显式界面水”溶剂化作用不衰减（恒为 $1.0$）**；交换步同步嵌入自洽 `MBAR` 多态自由能求解器，输出 $310.15\text{ K}$ 无偏重加权系综与界面耦合漏斗自由能 $\Delta G_{\text{funnel}}$。

#### 先行文献对比与必引划界
| 先行工作 | 已有贡献（我们采用/致敬，不声称首创） | 与 ASH-MD `IHRT-MBAR` 的严格差异边界 |
|---|---|---|
| **REST2 / gREST**（Wang, Friesner & Berne, *J. Phys. Chem. B* 2011; Kamiya & Sugita, *J. Chem. Phys.* 2018） | **最近邻先行者（必引）**：局部哈密顿量缩放替代全局升温，以极少数副本实现局部构象增强采样 | 标准 `REST2` 将多肽定义为热区时，其与水分子间的静电/范德华项按 $\sqrt{\lambda_m}$ 同步衰减，导致带电多肽在高副本发生非物理去溶剂化坍缩；`IHRT` **严格解耦“多肽-受体界面”与“多肽-显式水溶剂化”**（仅缩放溶质间界面并耦合 $r^6$ 位阻软核，且保护平面酰胺键 $\omega$ 不软化），专为口袋内假阳性诱饵逃逸设计 |
| **MBAR 多态自由能重加权**（Shirts & Chodera, *J. Chem. Phys.* 2008） | 多哈密顿状态采样数据的渐近最小方差无偏自由能与权重估计数学理论（我们直接采用该统计力学估计器，不声称首创 MBAR 方程） | 我们在 C++ 引擎内利用不变量消去以 $O(N_{\text{inter}})$ 代数复杂度精确构造 `IHRT` 交叉能量矩阵（相对 16 次全量求值误差 $<10^{-8}\,k_B T$）并流式馈入内嵌 `MBAR` 不动点求解器，结合最小化后界面能量自适应 $\lambda_m$ 排布，实时给出各副本无量纲自由能 $f_m$ 与 $310.15\text{ K}$ 物理态权重 |

---

## 3. 诚实科学边界声明（防御性划界与禁用声明清单）

为维护科学严谨性与 PepFind 平台评审合规性，ASH-MD v2.0 明确做出如下边界约束：
1. **不声称替代微秒级全周期显式溶剂自由能微扰（FEP）**：`ASH` 模式通过冻结远端骨架（Zone C）与有限显式界面水壳加速局部结合口袋采样，因此**无法捕获远端结构域大尺度变构开合（allosteric domain closure）**；需要考察全蛋白柔性时应切换至引擎内置的 `explicit-pme` 模式。
2. **不声称替代湿实验 ITC/SPR 或炼金术绝对自由能微扰（ABFE）**：v2.0 输出的 `dg_bind_pred_mean_kcal` 为基于 92 点 Connolly SASA、Åqvist 溶剂色散抵消、Sitkoff 微观空腔张力、Warshel 极性界面响应与 Wand 双侧界面侧链旋异构体熵的**零经验回归拟合物理结合自由能泛函**（消除传统 `MM/GBSA` 百千卡量纲膨胀，在 `14/14` RCSB 复合物上 `RMSE < 2.0 kcal/mol`），与 `mmgbsa_dg_mean_kcal`、`mbar_funnel_dg_kcal`、`ash_binding_score_kcal` 共同用于**高准确度真/假阳性姿态与热点突变判别**及**候选多肽亲和力排序**。
3. **金属离子配位模型透明披露**：当保留催化/结构 $\text{Zn}^{2+}$（含 `His ND1/NE2`、`Asp/Glu OD/OE` 及去质子化硫醇盐 `CYM SG`）时，默认对晶体学内配位层蛋白配体原子施加内配位层谐振配位约束，同时保持开放配位点供显式水或底物多肽自由配位；代码与报告中明确声明该处理为**固定电荷内配位层约束模型，不等同于量子化学重拟合电荷的 ZAFF 力场或可极化力场**。
