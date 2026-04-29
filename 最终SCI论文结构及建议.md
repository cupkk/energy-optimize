# 论文总体定位：从“离线 ADMM 调度”升级为“在线随机分布式调度”

我建议你直接按第二版写，而且不要再把论文定位成普通的 **Distributed EV Charging via ADMM**。你原来的方案已经有一个很好的基础：它考虑了变压器容量、单车充电功率、充电截止时间、时变电价，并计划比较 centralized optimization、dual decomposition 和 ADMM，指标包括 cost、peak load、convergence speed 和 fairness。 但如果目标是 **SCI 或 CCF-B 级别的计算机交叉论文**，仅仅“建立凸优化模型 + 用 ADMM 分布式求解”创新性偏弱。你之前确定的研究问题已经强调了成本、公平性、削峰和可扩展性之间的 trade-off，这条主线是对的，但现在需要进一步加入第 13–15 节课带来的**随机优化、Lyapunov 在线控制、MDP/队列理论**思想。

我建议最终论文题目定为：

> **Online Risk-Aware Distributed EV Charging Scheduling via Lyapunov-ADMM under Time-Varying Electricity Prices**

中文可以写成：

> **考虑时变电价与不确定到达的风险感知型在线分布式电动汽车充电调度：一种 Lyapunov-ADMM 方法**

这个题目比原来更强，因为它不再是假设所有车辆信息提前已知的离线调度，而是面向真实充电站中的**在线决策场景**：车辆随机到达，离站时间不同，剩余需求动态变化，电价和基础负荷时变，调度策略必须边运行边更新。这一点会显著提升论文的研究价值。

---

# 论文核心研究问题

整篇论文建议围绕一个主问题展开：

> **在车辆随机到达、电价时变、容量受限和用户截止时间异质的智能充电网络中，如何设计一种既可在线运行、又可分布式实现、同时兼顾成本、削峰、公平性与准时完成率的充电调度方法？**

这个主问题可以拆成三个研究问题，但论文里不要写得太散。你可以在 Introduction 里自然引出：

**RQ1：如何将 EV 充电调度建模为一个在线随机优化问题？**
原来的离线模型假设 (a_i,d_i,E_i,\pi_t) 都提前已知，但现实中车辆到达、需求、电价和基础负荷都具有不确定性。因此，本文需要建立一个动态模型，用状态变量描述“当前还剩多少未完成充电需求”，而不是一次性求出全天调度计划。

**RQ2：如何利用 Lyapunov drift-plus-penalty 将长期 deadline satisfaction 与实时成本优化转化为每时隙可解的凸优化问题？**
这是论文最关键的理论升级。你不再只是优化一个静态目标，而是把每辆车的未完成需求看成一个虚拟队列，通过 Lyapunov drift 控制队列不积压，再用 penalty 项控制电价成本和峰值负荷。

**RQ3：如何用 ADMM 将每时隙优化问题分解为车辆侧局部更新和充电站侧全局协调？**
这保持了你原论文的 ADMM 主线，但现在 ADMM 不是孤立方法，而是服务于 online Lyapunov control 的分布式求解器。这样“Lyapunov 负责在线控制，ADMM 负责分布式实现”，逻辑非常清楚。

---

# 论文核心贡献设计

你的贡献不要写成“本文使用 ADMM”。ADMM 是已有工具，不是贡献本身。真正的贡献应该写成下面三条。

**第一，提出 deadline-aware virtual queue 建模方法。**
将每辆 EV 的剩余充电需求表示为动态虚拟队列 (Q_i(t))，用队列积压程度刻画用户的 deadline urgency。这样可以把“离站前必须充够电”这种长期约束，转化为在线可控的队列稳定问题。

**第二，提出 risk-aware Lyapunov-ADMM 在线分布式调度框架。**
利用 Lyapunov drift-plus-penalty 将长期平均性能优化转化为每个时隙的凸优化子问题；进一步通过 ADMM 将该子问题分解为 EV-level local update 和 aggregator-level coordination，从而降低集中计算和通信开销。

**第三，系统揭示 cost-delay-fairness-scalability trade-off。**
通过仿真实验比较 offline centralized optimum、greedy charging、dual decomposition、offline ADMM 和 proposed online Lyapunov-ADMM，分析总成本、峰值负荷、deadline violation、队列积压、公平性和运行时间之间的权衡关系。

这三条贡献能形成一个完整论文故事：**建模创新 → 方法创新 → 实验洞察**。

---

# 建议论文结构

## Title

**Online Risk-Aware Distributed EV Charging Scheduling via Lyapunov-ADMM under Time-Varying Electricity Prices**

这个题目里每个词都有作用：

**Online** 表示不是离线全局规划，而是实时决策；
**Risk-Aware** 表示考虑容量不确定性或基础负荷扰动；
**Distributed** 表示不是单中心求解；
**Lyapunov-ADMM** 表示方法主线明确；
**Time-Varying Electricity Prices** 表示应用场景具体。

---

## Abstract

摘要建议按“背景—问题—方法—结果—意义”写。不要一上来写“本文研究了……”，而要直接指出研究痛点。

可以这样组织：

> Large-scale electric vehicle charging networks require real-time coordination under uncertain vehicle arrivals, heterogeneous charging deadlines, time-varying electricity prices, and transformer capacity limits. Existing offline centralized optimization methods usually rely on complete future information and may suffer from poor scalability and limited deployability in practical smart charging systems. To address these challenges, this paper proposes an online risk-aware distributed EV charging framework based on Lyapunov drift-plus-penalty and ADMM. A deadline-aware virtual queue is introduced to characterize the remaining charging demand and urgency of each EV, and the long-term charging satisfaction problem is transformed into a sequence of per-slot convex optimization problems. An ADMM-based distributed solver is then developed to decompose each per-slot problem into EV-level local updates and aggregator-level coordination. Numerical experiments show that the proposed method achieves near-centralized performance in charging cost and peak-load reduction while significantly improving scalability, deadline satisfaction, and online adaptability.

中文意思是：
大规模 EV 充电调度面临随机到达、异质 deadline、时变电价和容量约束；传统离线集中式方法依赖完整未来信息，不适合真实部署；本文提出 Lyapunov-ADMM 框架，用虚拟队列表达未完成需求，用 drift-plus-penalty 得到在线凸子问题，再用 ADMM 分布式求解；实验验证它接近集中式性能，同时更可扩展。

---

## 1. Introduction

Introduction 的逻辑必须严密。建议分成四段，不要写太散。

第一段讲背景：EV 渗透率提升导致充电网络面临峰值负荷、配电容量、电价波动和用户充电时限压力。无序充电会造成 transformer overload、peak demand increase 和 electricity cost increase。

第二段讲现有研究不足：许多 centralized optimization 方法可以得到较优调度，但依赖完整未来信息和中心化计算；一些 distributed ADMM 方法提升了可扩展性，但多数仍是 offline deterministic scheduling；RL 方法可以处理在线随机性，但需要训练样本、可解释性弱，并且在大规模车辆场景下容易出现状态空间爆炸。

第三段引出本文核心思想：将未完成充电需求建模为 deadline-aware virtual queues，用 Lyapunov drift-plus-penalty 处理在线随机性，再用 ADMM 分布式求解每个时隙的凸优化问题。这里要明确说：**本文不是用 ADMM 替代 centralized solver，而是把 ADMM 嵌入 online Lyapunov control framework。**

第四段列贡献。贡献建议写三条，控制在半页以内：

> This paper makes the following contributions. First, we formulate an online stochastic EV charging scheduling problem with deadline-aware virtual queues under time-varying electricity prices and capacity constraints. Second, we derive a Lyapunov drift-plus-penalty formulation that converts long-term charging urgency and cost optimization into tractable per-slot convex programs. Third, we develop a distributed ADMM-based solver and provide extensive simulations to evaluate the trade-offs among charging cost, peak load, deadline satisfaction, fairness, and scalability.

---

## 2. Related Work

Related Work 不要按论文逐篇罗列，要按研究线索组织。

第一类是 **offline EV charging optimization**。这一类主要做成本最小化、valley filling、peak shaving、deadline-aware scheduling。你要指出：它们通常假设未来信息已知，因此适合计划调度，但不完全适合随机在线场景。

第二类是 **distributed EV charging and ADMM-based coordination**。这一类是你原方案的基础，强调 dual decomposition、price-based coordination、ADMM distributed energy management。你要指出：它们解决了可扩展性问题，但许多方法仍是在离线或准静态条件下求解。

第三类是 **online stochastic optimization and Lyapunov control**。这一类是你新增的理论支柱。你要强调 Lyapunov optimization 的优势是 prediction-free、online、low-complexity，并且能把长期队列稳定转化为每时隙优化问题。

第四类是 **reinforcement learning for EV charging**。这一类不用深入展开，重点是把它作为对比背景：RL 可以学习复杂策略，但通常需要大量训练和调参；本文选择更具可解释性和低计算成本的 Lyapunov-ADMM 路线。

最后一段必须总结 gap：

> Existing studies have not sufficiently addressed online EV charging scheduling under simultaneous uncertainty, deadline urgency, and distributed deployment requirements. In particular, few works integrate Lyapunov-based online control with ADMM-based distributed coordination in a unified convex optimization framework.

这句话就是你论文的缺口。

---

## 3. System Model

这一节是地基，要写得干净、专业、可复现。

设时间被离散为 (t=1,\ldots,T)，每个时隙长度为 (\Delta t)。在时刻 (t)，当前活跃车辆集合为 (\mathcal{N}(t))。每辆车 (i) 有到达时间 (a_i)、离站时间 (d_i)、总能量需求 (E_i)、最大充电功率 (p_i^{\max})、充电效率 (\eta_i)。

定义决策变量：

[
p_i(t)
]

表示车辆 (i) 在时刻 (t) 的充电功率。

基础约束为：

[
0 \le p_i(t) \le p_i^{\max}, \quad i\in \mathcal{N}(t)
]

[
p_i(t)=0,\quad t<a_i \ \text{or}\ t>d_i
]

[
\sum_{i\in\mathcal{N}(t)}p_i(t)+B_t \le C_t
]

其中 (B_t) 是基础负荷，(C_t) 是变压器容量。

如果你想体现 risk-aware，可以加入一个简单、可控的风险容量约束。比如假设基础负荷预测误差为 (\tilde B_t)，则约束写成：

[
\Pr\left(\sum_i p_i(t)+\tilde B_t \le C_t\right)\ge 1-\epsilon
]

为了保持模型可解，可以把它转为安全裕度形式：

[
\sum_i p_i(t)+\hat B_t+\Gamma_t(\epsilon)\le C_t
]

其中 (\Gamma_t(\epsilon)) 是风险缓冲项。第一稿可以把它设为：

[
\Gamma_t(\epsilon)=\kappa\sigma_t
]

这里 (\sigma_t) 表示基础负荷预测误差标准差，(\kappa) 表示风险保守程度。这样你既体现了 risk-aware，又不会把论文变成复杂的 chance-constrained programming。

---

## 4. Deadline-Aware Virtual Queue Formulation

这是论文最重要的建模创新。

定义每辆车的剩余未满足需求：

[
Q_i(t)
]

可以理解为“这辆车还欠多少电没有充完”。当车辆 (i) 到达时：

[
Q_i(a_i)=E_i
]

之后队列更新为：

[
Q_i(t+1)=\left[Q_i(t)-\eta_i p_i(t)\Delta t\right]^+
]

其中 ([x]^+=\max(x,0))。

这个公式非常直观：
本时刻充得越多，剩余需求越少；如果已经充够，就不再为负。

为了体现 deadline urgency，可以引入时间紧迫权重：

[
w_i(t)=\frac{1}{d_i-t+\delta}
]

其中 (\delta>0) 防止分母为零。越接近离站时间，(w_i(t)) 越大。定义加权 Lyapunov 函数：

[
L(t)=\frac{1}{2}\sum_{i\in\mathcal{N}(t)} w_i(t)Q_i^2(t)
]

这个设计有明确物理意义：

**剩余需求越多、离站时间越近，系统越应该优先给它充电。**

这比普通 fairness penalty 更有论文价值，因为它把 deadline 直接放进动态控制结构，而不只是静态目标函数里加一项。

---

## 5. Lyapunov Drift-Plus-Penalty Optimization

定义 Lyapunov drift：

[
\Delta(t)=\mathbb{E}\left[L(t+1)-L(t)\mid \mathbf Q(t)\right]
]

它表示当前决策会让系统“未满足需求压力”增加还是减少。

同时定义当前时刻运行成本：

[
C_{\text{op}}(t)=\pi_t\sum_i p_i(t)+\beta\left(\sum_i p_i(t)-P_{\text{ref}}(t)\right)^2
]

其中第一项是电价成本，第二项是负荷平滑或削峰项。

本文每个时刻最小化：

[
\Delta(t)+V C_{\text{op}}(t)
]

其中 (V>0) 是权衡参数。
(V) 大，算法更看重电费和削峰；
(V) 小，算法更看重尽快满足未完成需求。

经过 drift upper bound 推导，可以得到每个时隙的凸优化问题：

[
\min_{\mathbf p(t)}
V\pi_t\sum_i p_i(t)
-\sum_i w_i(t)Q_i(t)\eta_i p_i(t)\Delta t
+
V\beta\left(\sum_i p_i(t)-P_{\text{ref}}(t)\right)^2
]

subject to:

[
0\le p_i(t)\le p_i^{\max}
]

[
\sum_i p_i(t)+\hat B_t+\Gamma_t(\epsilon)\le C_t
]

这个式子是整篇论文的核心。它很好解释：

[
V\pi_t\sum_i p_i(t)
]

表示电价高时少充；

[
-\sum_i w_i(t)Q_i(t)\eta_i p_i(t)\Delta t
]

表示欠电多、快离站的车优先充；

[
V\beta\left(\sum_i p_i(t)-P_{\text{ref}}(t)\right)^2
]

表示不要制造新的负荷尖峰。

这一节最后要证明或说明：该 per-slot problem 是凸优化问题，因为目标由线性项和二次凸项组成，约束均为线性约束。

---

## 6. Distributed ADMM Solver

这一节把你的原始 ADMM 主线接回来。

每个时隙的优化问题虽然是凸的，但集中式求解需要 aggregator 收集所有车辆的完整信息。为了分布式实现，引入变量分裂：

[
p_i(t)=z_i(t)
]

其中 (p_i(t)) 是车辆本地变量，(z_i(t)) 是 aggregator 侧一致性变量。

写成：

[
\min_{\mathbf p,\mathbf z}
\sum_i f_i(p_i;t)+g(\mathbf z;t)
]

subject to:

[
\mathbf p-\mathbf z=0
]

其中：

[
f_i(p_i;t)=V\pi_t p_i-w_i(t)Q_i(t)\eta_i p_i\Delta t
]

本地约束是：

[
0\le p_i\le p_i^{\max}
]

而 (g(\mathbf z;t)) 包含负荷平滑项和容量约束：

[
g(\mathbf z;t)=V\beta\left(\sum_i z_i-P_{\text{ref}}(t)\right)^2
+I_{\mathcal Z}(\mathbf z)
]

其中 (I_{\mathcal Z}) 是容量可行集的指示函数。

增广拉格朗日函数为：

[
\mathcal{L}_{\rho}(\mathbf p,\mathbf z,\mathbf u)
=================================================

\sum_i f_i(p_i;t)+g(\mathbf z;t)
+
\frac{\rho}{2}|\mathbf p-\mathbf z+\mathbf u|_2^2
]

ADMM 更新为：

[
p_i^{k+1}
=========

\arg\min_{0\le p_i\le p_i^{\max}}
f_i(p_i;t)
+
\frac{\rho}{2}(p_i-z_i^k+u_i^k)^2
]

[
\mathbf z^{k+1}
===============

\arg\min_{\mathbf z\in\mathcal Z}
g(\mathbf z;t)
+
\frac{\rho}{2}|\mathbf p^{k+1}-\mathbf z+\mathbf u^k|_2^2
]

[
\mathbf u^{k+1}
===============

\mathbf u^k+\mathbf p^{k+1}-\mathbf z^{k+1}
]

这一节要强调分工：

**车辆侧**只需要知道自己的 (Q_i(t),d_i,p_i^{\max})；
**aggregator 侧**只负责容量、价格、负荷平滑和一致性协调；
这样可以减少隐私暴露和集中计算压力。

---

## 7. Theoretical Analysis

这一节不需要做得非常深，但必须有，否则论文像工程仿真报告。

建议放三个性质。

**Proposition 1：Convexity of the per-slot problem.**
说明每个时隙的 Lyapunov-derived optimization problem 是凸的，因此可由 ADMM 收敛求解。

**Proposition 2：Distributed convergence of ADMM.**
在标准凸性和可行性假设下，每个时隙内 ADMM 的 primal residual 和 dual residual 收敛到零。这里不要夸大为全局长期最优，只说 per-slot convex subproblem 的分布式收敛。

**Proposition 3：Cost-backlog trade-off.**
在到达过程和容量满足可稳定条件时，Lyapunov 参数 (V) 控制成本和队列积压的折中。可以写成定性或半理论表述：

[
\text{Cost gap}=O(1/V),\quad \text{Average backlog}=O(V)
]

这类结论是 Lyapunov optimization 的典型形式。你可以在论文中作为“expected theoretical property”或在严格假设下证明简化版本。它非常有用，因为实验里可以画 (V) 对 cost 和 backlog 的影响，理论与实验能对应起来。

---

## 8. Numerical Experiments

实验设计要围绕研究问题，不要只画收敛曲线。

仿真场景建议设为：一天 24 或 48 个时隙，EV 数量为 (50,100,200,500)，车辆到达服从泊松过程或经验分布，离站时间由 parking duration 随机生成，需求 (E_i) 从区间分布采样，电价采用峰谷电价并叠加扰动，基础负荷采用日负荷曲线并加入噪声。

对比方法建议包括：

**Uncontrolled charging**：到达后以最大功率充电。
**Greedy price-based charging**：低电价时优先充。
**Offline centralized optimum**：假设完整未来信息已知，作为性能下界。
**Offline ADMM**：你原来的方法，作为分布式离线基线。
**Dual decomposition**：课程相关 baseline。
**Proposed Online Lyapunov-ADMM**：你的主方法。

评价指标建议固定为六个：

[
\text{Total charging cost}
]

[
\text{Peak load / peak-to-average ratio}
]

[
\text{Deadline violation rate}
]

[
\text{Average remaining demand / backlog}
]

[
\text{Fairness index}
]

[
\text{Runtime and ADMM iterations}
]

结果组织建议按照问题展开：

第一组回答：在线方法相比 greedy 和 uncontrolled 是否降低成本和峰值？
第二组回答：相比 offline optimum，性能损失有多大？
第三组回答：(V) 如何控制 cost-delay trade-off？
第四组回答：车辆数增大时，ADMM 是否比 centralized 更可扩展？
第五组回答：risk buffer (\Gamma_t(\epsilon)) 增大时，容量 violation 是否下降、成本是否上升？

这样实验就不是“为了画图而画图”，而是在回答论文研究问题。

---

## 9. Discussion

这一节建议保留。SCI/CCF 论文里，Discussion 可以显著提升成熟度。

第一段讨论为什么 Lyapunov 比普通 greedy 更稳。Greedy 只看当前价格，而 Lyapunov 同时看当前价格和未来 deadline pressure，因此能避免“为了便宜一直拖延，最后来不及充”的问题。

第二段讨论为什么 ADMM 适合这个问题。EV charging 的目标天然具有本地项，容量约束和负荷平滑项是全局耦合项，因此变量分裂后非常适合 distributed consensus optimization。

第三段讨论为什么不采用 RL 作为主方法。完整 MDP/Q-learning 能表达长期最优策略，但车辆数量增大时状态空间爆炸，而且需要训练样本；本文方法不需要训练，计算成本低，可解释性更强。

第四段诚实写限制。比如当前实验主要基于合成数据，未加入 battery degradation，没有考虑 V2G，没有做真实充电站数据验证，risk model 只采用简化安全裕度。这样反而更像成熟论文。

---

## 10. Conclusion

Conclusion 不要重复摘要，要回到研究问题。

可以这样写：

> This paper presented an online risk-aware distributed EV charging framework that integrates deadline-aware virtual queues, Lyapunov drift-plus-penalty optimization, and ADMM-based distributed coordination. The proposed method transforms long-term charging urgency and cost minimization into tractable per-slot convex programs, enabling real-time scalable implementation under time-varying electricity prices and uncertain charging demand. Numerical results demonstrate that the proposed Lyapunov-ADMM approach achieves a favorable balance among charging cost, peak-load reduction, deadline satisfaction, fairness, and scalability. Future work will incorporate real-world charging datasets, battery degradation models, renewable generation uncertainty, and asynchronous communication protocols.

---

# 最终论文主线

你这篇论文最重要的是保持一条线：

> **真实 EV 充电是随机在线问题，不是静态离线问题；deadline 可以用虚拟队列表达；Lyapunov drift-plus-penalty 可以把长期在线控制转成每时隙凸优化；ADMM 可以把每时隙问题分布式求解；实验验证 cost-delay-fairness-scalability trade-off。**

这条线非常清楚，也比原来的“fairness-aware ADMM”更有研究价值。

我建议你后续就按这个版本推进。原来的离线 ADMM 方案不要丢掉，而是放到实验 baseline 里。这样你既保留了课程前半部分的凸优化和 ADMM 主线，又融合了第 13–15 节的随机优化、Lyapunov、MDP 思想，整体会更像一篇可以往 **计算机交叉 SCI / CCF-B 方向**推进的完整论文。
