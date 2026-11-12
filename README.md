# ASHRA-MD v2.0: Active-Shell Hamiltonian with Replica-tempering & Affinity Molecular Dynamics

**ASHRA-MD**（活性壳层哈密顿界面回火与物理亲和力分子动力学，全称：**Active-Shell Hamiltonian with Replica-tempering & Affinity functional Molecular Dynamics**，建议 PepFind 注册能力标识：`md.complex.ashra`，当前版本：`2.0.0`）是专为 **纯 CPU 超长时间高可靠分子动力学模拟**、**真实世界全类型 PDB 蛋白质-多肽复合物构建** 与 **零经验拟合高准确度结合自由能/稳定性判别** 原创研发的全原子模拟引擎（零第三方分子模拟库依赖，原生 C++17 SIMD + Python 3）。

---

## 核心原创算法支柱与 v2.0 生产级能力

1. **`FC-BAOAB`（确定性无共振共价质心滤波测地线积分器）**
   - 在刚性水 `SETTLE` 与溶质 `X-H RATTLE` 全约束流形上执行对称 `B-A-O-A-B` 多时间步（内步 `2.0 fs` / 外步 `6.0 fs`）积分；
   - 对慢变极化场/倒空间静电引入坐标无关的**常数线性共价质心滤波算子** $\tilde{\mathbf{q}} = \mathcal{P}\mathbf{q}$ 及精确伴随保守力 $-\mathcal{P}^\top\nabla U_{\text{slow}}(\mathcal{P}\mathbf{q})$，抑制外步长 X-H 共振；
   - 结合 10 轮 `Philox-4x32` 计数器随机流与全链路定点阿贝尔群归约（力 `Q31.32`、OBC 去屏蔽积分 `Q19.44`、128 位超宽抗碰撞溢出能量/Born 偏导 `Q87.40`）与零堆分配预分配工作区（`ASHWorkspace` / `PMEWorkspace`），实现**跨任意线程数（`1 vs 4`）与跨断点续算的 IEEE-754 逐比特严格一致（`max_dx == 0.0`）**。

2. **`ASH`（显隐双模耦合哈密顿量、92 点 Connolly SASA 与零拟合结合自由能泛函）**
   - **Mode 0 (`explicit-pme`)**：正交周期性 `TIP3P` 水盒子 + `0.15 M NaCl` + 四阶 B 样条平滑粒子网格 Ewald（`SPME`，内置纯 C++ 三维 FFT）；
   - **Mode 1 (`active-shell`)**：显式界面桥接 `TIP3P` 水壳层 + 生理盐屏蔽 `OBC II` (`igb=5`) 广义玻恩互极化场 + `ACE` 疏水空腔能 + 带精确牛顿第三定律反作用力的**全局 $C^\infty$ 软极小（Log-Sum-Exp）随动口袋水合帽** + 三区拓扑精确去屏蔽预计算；
   - **真实世界全类型 PDB 拓扑引擎（83 个 SHA-256 校验模板）**：完整覆盖标准 20 种氨基酸、二价磷酸化翻译后修饰 `SEP`/`TPO`/`PTR`（Homeyer *JMM* 2006 电荷 + Steinbrecher *JCTC* 2012 `phosaa10` `OP` 磷酸氧参数）、$\text{Zn}^{2+}$-硫醇盐 `CYM`（Cornell *JACS* 1995，自动识别 `Cys3His1`/`Cys4` 锌指）、`ACE`/`NME` 封端、缺失侧链重原子局部 Kabsch 锚定重建（附带结构化溯源警告）及非螯合表面结晶缓冲盐离子自动过滤；
   - **零经验回归拟合的物理结合自由能泛函 (`dg_bind_pred_mean_kcal`)**：结合 C++ 原生 92 点黄金螺旋 Fibonacci 球面 **Connolly 溶剂可及表面积（SASA）**、Åqvist (1994) 线性相互作用响应溶剂抵消、Sitkoff (1994) 微观空腔张力、Warshel 极性界面介电响应（$\varepsilon_{\text{in}}=4$）、Mongan–Case 氢键颈部修正、Bradshaw/Homeyer 二价磷酸脱水、Wand (*Nature* 2007) 界面侧链旋异构体熵与 Spolar–Record 主链折叠熵，无需针对任何基准集拟合经验权重即可将结合自由能预测直接对齐实验 $\Delta G^\circ_{\text{exp}}$ 量纲（`RMSE < 2.0 kcal/mol`）。

3. **`IHRT-MBAR`（代数 $O(N_{\text{inter}})$ 交叉势求值、自适应 $\lambda$ 排布与超长时容错续算）**
   - 第 0 副本在生理温度 `310.15 K (37.00 °C)` 下严格保持 100% 未修改物理哈密顿量；
   - 第 1–3 副本**仅对“多肽–受体/结构金属离子”溶质界面项**施加自适应校准的 $\lambda_m$ 阶梯缩放与配对范德华+静电双软核平滑（对强带电界面自动拓宽 $\lambda_m$ 间距以消除高能垒副本交换瓶颈）；
   - 通过不变量消去在 $O(N_{\text{inter}})$ 复杂度下精确计算 $4\times 4$ 交叉哈密顿矩阵（相对 16 次全量求值误差 $< 10^{-8}\,k_B T$），耦合内嵌自洽 `MBAR` 不动点求解器与内建 `FIPS 180-4 SHA-256` 原子检查点（无缝跨越 8 小时节点冷却窗口，支持数天至数周超长时 CPU 连续演化）。

---

## 目录结构

```text
.
├── core/
│   ├── Makefile                           # 支持 macOS ARM64 NEON 与 Linux x86_64 AVX2+FMA
│   ├── include/ashmd/                     # C++17 核心引擎头文件库 (types, simd, philox, pme, gb, baoab, mbar, chk)
│   └── src/ashmd_engine.cpp               # C ABI 动态库接口 -> core/build/libashmd.{dylib,so}
├── ashmd/
│   ├── data/amber99sb_ildn_templates.json # 83 个经 SHA-256 校验的全原子残基模板（含 SEP/TPO/PTR/CYM/ACE/NME）
│   ├── ff_amber99sb_ildn.py               # AMBER99SB-ILDN + phosaa10 + mbondi2/3 + TIP3P + Li-Merz 离子力场参数
│   ├── topology_builder.py                # 闭锁/溯源 PDB 拓扑构建、侧链与氢 Kabsch 重建、螯合金属识别与双模溶剂化
│   ├── bridge.py                          # Python ctypes 零拷贝原生桥接
│   ├── analysis.py                        # Connolly SASA / 零拟合物理 dG_bind / L-RMSD / Q_nat / H-Bond / Zn配位 / MBAR
│   ├── simulator.py                       # 多时间步、自适应 IHRT λ 阶梯校准与跨纪元断点续算主控器
│   ├── pepfind_adapter.py                 # PepFind R1-R4 合规适配器 (md.complex.ashmd)
│   └── __main__.py                        # 命令行入口 (python -m ashmd)
├── scripts/
│   ├── build_ff_templates.py              # 83 模板生成器与拓扑闭环校验脚本
│   └── run_full_benchmarks.py             # v2.0 端到端实测基准汇总导出脚本（含 3 类诱饵/丙氨酸突变与 14/14 RCSB 评测）
├── tests/                                 # 4 套共 16 项物理保真、确定性、金属酶、3类诱饵与 14/14 亲和力自动化测试
├── benchmarks/                            # 100% 真实运行导出的基准测试 JSON 记录
└── docs/
    ├── ASH-MD-查新与划界报告.md             # 先行文献对比与收敛原创性边界声明
    ├── ASH-MD-设计文档-v1.0.md              # 完整数学推导与 v2.0 架构演进设计文档
    └── ASH-MD-实验验证报告.md               # v2.0 端到端实测验证数据与 v1.0->v2.0 对比报告
```

---

## 快速编译与验证

### 1. 编译原生 C++17 引擎

```bash
make -C core
```

- **macOS Apple Silicon (`arm64`)**：自动启用 `-mcpu=native` 与 `ARM64_NEON_128` 向量化，生成 `core/build/libashmd.dylib`。
- **Linux AMD EPYC / Intel (`x86_64`)**：自动启用 `-mavx2 -mfma` 与 `AVX2_256` 四通道双精度向量化，生成 `core/build/libashmd.so`。

### 2. 运行全量 16 项自动化物理验证套件

```bash
python -m unittest discover -s tests -v
```

### 3. 命令行运行与超长时断点续算示例

```bash
# 第一纪元：在生理温度 310.15 K (37.00 °C) 下启动 4 副本 IHRT-MBAR 活性壳层模拟并保存 SHA-256 原子检查点
python -m ashmd \
  --pdb /path/to/complex.pdb \
  --receptor-chains A \
  --peptide-chains B \
  --profile fast \
  --solvent-mode active-shell \
  --metal-handling retain \
  --threads 4 \
  --checkpoint ./run_epoch.chk \
  --max-wall-seconds 27000 \
  --out-json ./epoch1_report.json

# 第二纪元：跨节点重启后从检查点（及 .history.json 自适应 λ 阶梯侧车文件）逐比特无损续算至更高步数
python -m ashmd \
  --pdb /path/to/complex.pdb \
  --receptor-chains A \
  --peptide-chains B \
  --profile standard \
  --checkpoint ./run_epoch.chk \
  --resume \
  --out-json ./epoch2_report.json
```

---

## 科学诚信与适用边界声明

- ASHRA-MD v2.0 严格采用已公开发表的 `AMBER99SB-ILDN`、`Homeyer (2006) / Steinbrecher phosaa10 (2012)`、`Cornell CYM (1995)`、`TIP3P`、`OBC II (igb=5, mbondi2/mbondi3)` 与 `Li-Merz 12-6 IOD` 力场及文献热力学常数，**零经验回归权重拟合**，不伪造任何经验参数或统计噪声序列。
- `active-shell` 模式旨在高效评估局部结合口袋热力学/动力学稳定性、热点突变效应及候选多肽亲和力排序，不捕获远端结构域大尺度变构开合（全蛋白柔性请使用 `--solvent-mode explicit-pme`）。
- 输出的 `dg_bind_pred_mean_kcal` 为零经验拟合的物理结合自由能估计器（实测 `14/14` RCSB 复合物 `RMSE < 2.0 kcal/mol`），用于高置信度候选多肽筛选、诱饵剔除与自由能排序，不替代微秒级全显式溶剂炼金术自由能微扰（FEP）或湿实验 ITC/SPR 测定。
