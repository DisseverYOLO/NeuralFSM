# NeuralFSM: 基于时序图网络的自适应有限状态机多智能体系统
# NeuralFSM: Adaptive Finite State Machine Multi-Agent System via Temporal Graph Networks

**完整论文文档索引**

---

## 📚 文档结构

本文档分为六个部分，完整涵盖了NeuralFSM系统的所有技术内容：

### [第一部分：引言与系统概述](PAPER_PART1_INTRODUCTION_AND_OVERVIEW.md)

**内容**：
- 摘要
- 引言（研究背景、核心创新、主要贡献）
- 相关工作
- 系统概述（整体架构、核心组件、工作流程、系统特点）

**关键内容**：
- 四个核心创新的概述
- 系统整体架构图
- 三个核心组件的介绍
- 典型工作流程

---

### [第二部分：有限状态机定义与内部机制](PAPER_PART2_FSM_DEFINITION_AND_MECHANISMS.md)

**内容**：
- 有限状态机的形式化定义
- 状态转移机制
- 状态完成机制
- 监听关系机制
- FSM执行流程
- FSM生成机制

**关键公式**：
- FSM六元组定义：`FSM = (S, A, δ, L, C, s₀)`
- 状态转移概率（任务自适应）：`P(sⱼ | sᵢ, c, q_emb) = TGN_transition(sᵢ, c, q_emb)`
- 监听权重（任务自适应）：`w(sᵢ, aⱼ, q_emb) = TGN_listener(sᵢ, aⱼ, q_emb)`
- 完成条件检查：`is_complete(sᵢ, output) = check_completion_condition(C(sᵢ), output)`
- 概率采样：`s_{t+1} ~ P(· | sᵢ, c, q_emb)`，`listeners ~ P(· | sᵢ, q_emb)`

---

### [第三部分：时序图网络运行机制与数学公式](PAPER_PART3_TGN_MECHANISMS_AND_FORMULAS.md)

**内容**：
- TGN在NeuralFSM中的核心作用
- TGN的四大核心组件（记忆模块、消息函数、消息聚合器、嵌入更新器）
- TGN完整前向传播流程
- FSM状态转移预测
- 监听路径预测
- TGN参数总结

**关键公式**：
- 记忆更新：`m_v(t) = GRU(m_v(t⁻), msg_v(t))`
- 消息计算：`msg(u → v, t) = MLP([h_u(t), h_v(t), e_{uv}, ϕ(Δt_{uv})])`
- 消息聚合：`msg_v(t) = AGG({msg(u → v, t) | u ∈ N(v)})`
- 时间编码：`ϕ(Δt) = [cos(ω₁·Δt), sin(ω₁·Δt), ...]`
- 嵌入更新：`z_v(t) = σ(W_emb · [h_v || m_v(t) || ϕ(Δt)] + b_emb)`
- 问题嵌入编码：`q_encoded = QuestionEncoder(q_emb)`，其中 `q_emb ∈ ℝ³⁸⁴`
- 状态转移概率（任务自适应）：`P(sⱼ | sᵢ, c, q_emb) = softmax(TGN_transition(sᵢ, c, q_emb))ⱼ`
- 监听权重（任务自适应）：`w(sᵢ, aⱼ, q_emb) = TGN_listener(sᵢ, aⱼ, q_emb)`
- 概率采样：`s_{t+1} ~ P(· | sᵢ, c, q_emb)`，`listeners ~ Multinomial(w(sᵢ, ·, q_emb))`

**参数数量**：约543,977个参数

---

### [第四部分：保护机制设计](PAPER_PART4_PROTECTION_MECHANISMS.md)

**内容**：
- 保护机制概述（双重防御策略）
- 中心性分析（介数中心性、PageRank中心性）
- 异常检测（频率异常、语义异常、异常分数融合、攻击类型与检测方法对应关系）
- 信任分数计算
- 消息权重计算
- 保护损失（训练时防御）
- 保护机制集成（训练时防御 + 运行时防御）
- 总损失函数与梯度

**关键公式**：
- 保护优先级：`π(i) = w_BC · BC(i) + w_PR · PR(i)`
- 介数中心性：`BC(i) = Σ_{s≠i≠t} [σ_st(i) / σ_st]`
- PageRank：`PR(i) = (1-α)/n + α · Σ_{j→i} [PR(j) / deg_out(j)]`
- 频率异常：`α_freq(i, t) = sigmoid(Z_score(count_i))`
- 语义异常：`α_semantic(i, t) = 1 - cos_similarity(z_i(t), μ_emb_i)`
- 综合异常：`α(i, t) = λ_freq · α_freq(i, t) + λ_semantic · α_semantic(i, t)`
- 信任分数：`trust(i, t) = (1 - α(i,t)) · (1 + π(i))`
- 消息权重：`weight(u, v) = MLP(trust(u), π(v), edge_features)`
- 保护损失：`L_protect = Σ_{(u,v) ∈ E} [risk(u,v) × message_strength(u,v)]`
- 总损失函数：`L_total = α · L_policy + β · L_trans + γ · L_listener + δ · L_cost + λ · L_protect`
- 总损失梯度：`∇_θ L_total = α · ∇_θ L_policy + β · ∇_θ L_trans + γ · ∇_θ L_listener + δ · ∇_θ L_cost + λ · ∇_θ L_protect`

**攻击类型与检测方法**：
- 频率攻击 → 频率异常检测（核心）
- 语义攻击 → 语义异常检测（核心）
- 拜占庭攻击 → 拜占庭攻击检测（扩展）
- Sybil攻击 → Sybil攻击检测（扩展）
- 自私攻击 → 自私攻击检测（扩展）

**参数数量**：约2,373个可学习参数

---

### [第五部分：损失函数设计](PAPER_PART5_LOSS_FUNCTIONS.md)

**内容**：
- 损失函数概述
- 策略梯度损失（REINFORCE算法）
- 状态转移损失（交叉熵损失）
- 监听路径损失（二元交叉熵损失）
- LLM成本损失（平方损失）
- 保护损失（加权损失）
- 总损失函数组合
- 梯度反向传播

**关键公式**：
- 总损失：`L_total = α · L_policy + β · L_trans + γ · L_listener + δ · L_cost + λ · L_protect`
- 策略损失：`L_policy = -E[log P(τ | θ) · R(τ)]`
- 转移损失：`L_trans = CrossEntropy(P(· | s_t, c_t, θ), s_{t+1}^true)`
- 监听损失：`L_listener = BinaryCrossEntropy(w(s, a), y_true(s, a))`
- 成本损失：`L_cost = (C_episode - C_baseline)²`
- 保护损失：`L_protect = Σ_{(u,v) ∈ E} [risk(u,v) × message_strength(u,v)]`

**默认权重**：
- `α = 1.0`（策略损失）
- `β = 0.3`（转移损失）
- `γ = 0.2`（监听损失）
- `δ = 0.1`（成本损失）
- `λ = 0.1`（保护损失）

---

### [第六部分：实验设计与结果](PAPER_PART6_EXPERIMENTS_AND_RESULTS.md)

**内容**：
- 数据集介绍（MMLU、GSM8K、HumanEval、HotpotQA、ALFWorld、MATH）
- 实验设置（训练配置、TGN配置、FSM配置、保护机制配置、提示词优化配置）
- 基线系统（Baseline-Fixed、Baseline-Dialogue、Baseline-Single）
- 实验结果（主要结果、状态转移准确率、监听路径准确率、LLM成本分析）
- 消融实验（核心组件消融、保护机制消融、损失函数消融）
- 提示词优化效果
- 保护机制效果分析
- 计算效率分析
- 讨论与分析
- 结论

**关键结果**：
- 平均准确率提升：+16.4%
- 保护机制在攻击场景下准确率提升：+20-40%
- 训练时间增加：+70%
- LLM成本增加：+12%

---

## 📊 公式汇总

### FSM相关公式

1. **FSM定义**：`FSM = (S, A, δ, L, C, s₀)`
2. **状态转移概率（任务自适应）**：`P(sⱼ | sᵢ, c, q_emb) = softmax(TGN_transition(sᵢ, c, q_emb))ⱼ`
3. **监听权重（任务自适应）**：`w(sᵢ, aⱼ, q_emb) = TGN_listener(sᵢ, aⱼ, q_emb)`
4. **概率采样**：`s_{t+1} ~ P(· | sᵢ, c, q_emb)`，`listeners ~ Multinomial(w(sᵢ, ·, q_emb))`
5. **问题嵌入**：`q_emb = SentenceTransformer(question_text) ∈ ℝ³⁸⁴`

### TGN相关公式

1. **记忆更新**：`m_v(t) = GRU(m_v(t⁻), msg_v(t))`
2. **消息计算**：`msg(u → v, t) = MLP([h_u(t), h_v(t), e_{uv}, ϕ(Δt_{uv})])`
3. **消息聚合**：`msg_v(t) = AGG({msg(u → v, t) | u ∈ N(v)})`
4. **时间编码**：`ϕ(Δt) = [cos(ω₁·Δt), sin(ω₁·Δt), ...]`
5. **嵌入更新**：`z_v(t) = σ(W_emb · [h_v || m_v(t) || ϕ(Δt)] + b_emb)`

### 保护机制相关公式

1. **保护优先级**：`π(i) = w_BC · BC(i) + w_PR · PR(i)`
2. **介数中心性**：`BC(i) = Σ_{s≠i≠t} [σ_st(i) / σ_st]`
3. **PageRank**：`PR(i) = (1-α)/n + α · Σ_{j→i} [PR(j) / deg_out(j)]`
4. **频率异常**：`α_freq(i, t) = sigmoid(Z_score(count_i))`
5. **语义异常**：`α_semantic(i, t) = 1 - cos_similarity(z_i(t), μ_emb_i)`
6. **综合异常**：`α(i, t) = λ_freq · α_freq(i, t) + λ_semantic · α_semantic(i, t)`
7. **信任分数**：`trust(i, t) = (1 - α(i,t)) · (1 + π(i))`
8. **消息权重**：`weight(u, v) = MLP(trust(u), π(v), edge_features)`
9. **保护损失（训练时防御）**：`L_protect = Σ_{(u,v) ∈ E} [risk(u,v) × message_strength(u,v)]`
10. **消息衰减（运行时防御）**：`protected_output = trust × TGN_output + (1 - trust) × original_features`
11. **连接过滤（运行时防御）**：`if weight(u, v) < threshold: filter_edge(u → v)`
12. **总损失函数**：`L_total = α · L_policy + β · L_trans + γ · L_listener + δ · L_cost + λ · L_protect`
13. **总损失梯度**：`∇_θ L_total = α · ∇_θ L_policy + β · ∇_θ L_trans + γ · ∇_θ L_listener + δ · ∇_θ L_cost + λ · ∇_θ L_protect`

### 损失函数相关公式

1. **总损失**：`L_total = α · L_policy + β · L_trans + γ · L_listener + δ · L_cost + λ · L_protect`
2. **策略损失**：`L_policy = -E[log P(τ | θ) · R(τ)]`
3. **转移损失**：`L_trans = CrossEntropy(P(· | s_t, c_t, θ), s_{t+1}^true)`
4. **监听损失**：`L_listener = BinaryCrossEntropy(w(s, a), y_true(s, a))`
5. **成本损失**：`L_cost = (C_episode - C_baseline)²`
6. **保护损失**：`L_protect = Σ_{(u,v) ∈ E} [risk(u,v) × message_strength(u,v)]`

---

## 🔍 快速查找指南

### 查找特定内容

- **FSM定义和机制** → 第二部分
- **TGN数学公式** → 第三部分
- **保护机制设计** → 第四部分
- **损失函数公式** → 第五部分
- **实验结果** → 第六部分

### 查找特定公式

- **状态转移** → 第二部分第2.3节、第三部分第3.6节
- **监听关系** → 第二部分第2.4节、第三部分第3.7节
- **异常检测** → 第四部分第4.3节
- **攻击类型与检测方法** → 第四部分第4.3.4节
- **信任计算** → 第四部分第4.4节
- **总损失函数** → 第四部分第4.8节、第五部分第5.1节
- **损失函数** → 第五部分

### 查找实验数据

- **主要结果** → 第六部分第8.4节
- **消融实验** → 第六部分第8.5节
- **保护机制效果** → 第六部分第8.7节
- **计算效率** → 第六部分第8.8节

---

## 📝 使用建议

1. **阅读顺序**：建议按照第一部分到第六部分的顺序阅读，了解系统的完整故事
2. **公式理解**：每个公式都有详细的符号说明和计算示例，建议仔细阅读
3. **实验验证**：第六部分提供了详细的实验结果，可以验证理论设计的有效性
4. **代码对照**：建议结合代码实现理解公式，代码位置见各部分的实现说明

---

## 🎯 文档特点

1. **完整性**：涵盖了NeuralFSM系统的所有技术内容
2. **详细性**：每个公式都有详细的推导和示例
3. **系统性**：从理论到实验，完整讲述了系统的设计和实现
4. **可读性**：结构清晰，符号统一，易于理解

---

**文档创建日期**：2024-11-13  
**文档版本**：V1.0  
**文档状态**：完整版


