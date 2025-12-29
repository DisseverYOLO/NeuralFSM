# NeuralFSM 保护机制简化设计方案

> **⚠️ 重要配置说明（2024-11-13）**  
> **所有保护防御实验统一使用可学习的MLP消息权重计算器**  
> - ✅ 使用: `MessageWeightCalculator` (可学习MLP, 2305参数)  
> - ❌ 不使用: `SimpleMessageWeightCalculator` (固定公式)  
> - 详见: `PROTECTION_MESSAGE_WEIGHT_CONFIG.md`

> **更新说明（2025-11-11）**  
> 本文档已根据当前仓库的实际实现重新整理，内容与 `protected_tgn.py`、`simplified_anomaly.py`、`simplified_centrality.py`、`trust_calculator.py`、`message_weight_calculator.py`、`protection_loss.py` 以及 `advanced_attack_injector.py`、`example_usage.py` 等最新模块完全对齐。旧版 `run_robustness_test.py` 已移除，不再作为参考。

## 🎯 核心设计思路 (根据您的需求)

### 设计原则
1. **简化中心性**: 只使用2种中心性指标 (介数 + PageRank)
2. **直接集成**: 在TGN的消息函数中引入保护权重
3. **端到端训练**: 使用任务回报 + 正则项联合训练
4. **信任机制**: 结合中心性和异常分数计算消息权重

---

## 📐 数学公式设计

### 1. 节点保护优先级 (简化版)

使用2种中心性指标的加权融合:

```python
π(i) = w_BC · BC(i) + w_PR · PR(i)
```

#### **1.1 介数中心性 (Betweenness Centrality)**

**数学定义:**
```
BC(i) = Σ_{s≠i≠t} [σ_st(i) / σ_st]
```

**符号说明:**
- `i`: 目标节点
- `s, t`: 源节点和目标节点对
- `σ_st`: 从s到t的最短路径总数
- `σ_st(i)`: 从s到t经过节点i的最短路径数
- `BC(i) ∈ [0, 1]`: 归一化后的介数中心性

**物理意义:**
- 衡量节点作为**"桥梁"**的重要性
- 高BC值意味着大量最短路径经过该节点
- 移除会导致网络连通性下降或分割

**计算示例:**
```python
# 4节点星形网络: a₁连接a₂,a₃,a₄
#     a₂
#      |
# a₃--a₁--a₄

# a₁是中心，所有路径都经过它
σ_23 = 1 (a₂→a₁→a₃), σ_23(a₁) = 1 → 贡献 1/1 = 1.0
σ_24 = 1 (a₂→a₁→a₄), σ_24(a₁) = 1 → 贡献 1/1 = 1.0
σ_34 = 1 (a₃→a₁→a₄), σ_34(a₁) = 1 → 贡献 1/1 = 1.0

BC(a₁) = (1.0 + 1.0 + 1.0) / max_possible = 1.0  # 最高
BC(a₂) = BC(a₃) = BC(a₄) = 0.0  # 叶子节点
```

---

#### **1.2 PageRank 中心性**

**数学定义:**
```
PR(i) = (1-α)/n + α · Σ_{j→i} [PR(j) / deg_out(j)]
```

**符号说明:**
- `PR(i) ∈ [0, 1]`: 节点i的PageRank分数（归一化）
- `α ∈ [0, 1]`: 阻尼系数（默认0.85）
- `n`: 图中节点总数
- `j→i`: 所有指向节点i的节点j
- `deg_out(j)`: 节点j的出度
- `(1-α)/n`: 随机跳转概率（防止陷入孤立节点）

**迭代计算:**
```python
# 初始化
PR⁽⁰⁾(i) = 1/n  # 所有节点初始分数相等

# 迭代更新（Power Iteration）
for k in range(max_iter):
    PR⁽ᵏ⁺¹⁾(i) = (1-α)/n + α · Σ_{j→i} [PR⁽ᵏ⁾(j) / deg_out(j)]
    
    # 收敛判断
    if |PR⁽ᵏ⁺¹⁾ - PR⁽ᵏ⁾| < ε:
        break
```

**物理意义:**
- 递归定义的**全局重要性**
- "重要节点"的入链接使节点更重要
- 考虑邻居节点的影响力

**计算示例:**
```python
# 3节点链: a₁→a₂→a₃→a₁ (循环)
# 初始: PR⁽⁰⁾ = [1/3, 1/3, 1/3]

# 第1次迭代 (α=0.85):
PR⁽¹⁾(a₁) = 0.15/3 + 0.85×(1/3)/1 = 0.05 + 0.283 = 0.333
PR⁽¹⁾(a₂) = 0.15/3 + 0.85×(1/3)/1 = 0.333
PR⁽¹⁾(a₃) = 0.15/3 + 0.85×(1/3)/1 = 0.333

# 循环对称，所有节点PR相同
```

---

#### **1.3 加权融合**

**融合公式:**
```python
π(i) = w_BC · BC(i) + w_PR · PR(i)
```

**可学习权重（带归一化）:**
```python
# 原始权重
w_BC_raw = nn.Parameter(torch.tensor(0.6))
w_PR_raw = nn.Parameter(torch.tensor(0.4))

# Sigmoid归一化
w_BC = torch.sigmoid(w_BC_raw)
w_PR = 1.0 - w_BC  # 确保权重和为1

# 计算优先级
π(i) = w_BC · BC(i) + w_PR · PR(i)
```

**符号说明:**
- `w_BC ∈ [0, 1]`: 介数中心性权重（初始0.6）
- `w_PR ∈ [0, 1]`: PageRank权重（初始0.4）
- `w_BC + w_PR = 1`: 权重归一化约束
- `π(i) ∈ [0, 1]`: 节点保护优先级

**计算示例:**
```python
# 假设4个智能体的中心性分数
BC = [0.85, 0.30, 0.50, 0.20]  # a₁是关键桥梁
PR = [0.40, 0.30, 0.20, 0.10]  # a₁全局重要

# 权重
w_BC = 0.6, w_PR = 0.4

# 优先级
π(a₁) = 0.6×0.85 + 0.4×0.40 = 0.51 + 0.16 = 0.67  # 最高
π(a₂) = 0.6×0.30 + 0.4×0.30 = 0.18 + 0.12 = 0.30
π(a₃) = 0.6×0.50 + 0.4×0.20 = 0.30 + 0.08 = 0.38
π(a₄) = 0.6×0.20 + 0.4×0.10 = 0.12 + 0.04 = 0.16  # 最低

# a₁最需要保护
```

---

**为什么选这两种?**
- **介数中心性**: 最能反映节点的**局部控制力**，移除会导致网络分割
- **PageRank**: 考虑**全局结构**，补充介数的局部视角
- **协同作用**: BC捕捉拓扑关键点，PR捕捉影响力传播
- **放弃度中心性**: 简单的连接数，信息量不足
- **放弃接近中心性**: 计算复杂度O(n³)，且与介数相关性高

---

### 2. 异常分数 (扩展版)

#### **2.1 核心检测（始终启用）**

**综合异常分数:**
```python
α(i, t) = λ_freq · α_freq(i,t) + λ_semantic · α_semantic(i,t)
```

**符号说明:**
- `α(i,t) ∈ [0, 1]`: 节点i在时刻t的异常分数
- `λ_freq, λ_semantic ∈ [0, 1]`: 可学习融合权重，λ_freq + λ_semantic = 1
- `α_freq, α_semantic ∈ [0, 1]`: 频率和语义异常分数

---

**2.1.1 频率异常检测（Z-score）**

**数学定义:**
```python
α_freq(i, t) = sigmoid(Z_score(count_i))
```

其中 Z-score:
```
Z_score(count_i) = |count_i - μ_count| / σ_count
```

**符号说明:**
- `count_i`: 节点i的通信次数（滑动窗口内）
- `μ_count`: 所有节点通信次数的均值
- `σ_count`: 所有节点通信次数的标准差
- `Z_score`: 标准化偏离度
- `sigmoid(x) = 1/(1+e^(-x))`: 映射到[0,1]

**计算示例:**
```python
# 4个智能体的通信次数（窗口=50步）
counts = [10, 45, 12, 8]  # a₂异常频繁

# 统计量
μ = mean(counts) = (10+45+12+8)/4 = 18.75
σ = std(counts) = sqrt(mean((x-μ)²)) = 14.87

# Z-score
Z(a₁) = |10 - 18.75| / 14.87 = 0.588
Z(a₂) = |45 - 18.75| / 14.87 = 1.764  # 异常高
Z(a₃) = |12 - 18.75| / 14.87 = 0.454
Z(a₄) = |8 - 18.75| / 14.87 = 0.723

# 异常分数
α_freq(a₁) = sigmoid(0.588) = 0.643
α_freq(a₂) = sigmoid(1.764) = 0.854  # 检测到频率攻击 ✅
α_freq(a₃) = sigmoid(0.454) = 0.612
α_freq(a₄) = sigmoid(0.723) = 0.673
```

---

**2.1.2 语义异常检测（余弦相似度）**

**数学定义:**
```python
α_semantic(i, t) = 1 - cos_similarity(z_i(t), μ_emb_i)
```

其中余弦相似度:
```
cos_similarity(a, b) = (a · b) / (||a|| · ||b||)
```

**符号说明:**
- `z_i(t) ∈ ℝᵈ`: 节点i在时刻t的TGN嵌入（d=256）
- `μ_emb_i ∈ ℝᵈ`: 节点i的历史嵌入均值（滑动窗口）
- `a · b`: 向量点积
- `||a||`: 向量L2范数
- `cos_similarity ∈ [-1, 1]`: 相似度，1表示完全相同
- `α_semantic ∈ [0, 2]`: 偏离度，0表示无偏离

**计算示例:**
```python
# 当前嵌入（256维，简化为3维演示）
z_a₂(t) = [0.8, -0.3, 0.5]  # 当前嵌入

# 历史均值（10步滑动窗口）
μ_emb_a₂ = [0.2, 0.1, 0.3]  # 历史中心

# 余弦相似度
numerator = 0.8×0.2 + (-0.3)×0.1 + 0.5×0.3 = 0.16 - 0.03 + 0.15 = 0.28
||z|| = sqrt(0.8² + 0.3² + 0.5²) = sqrt(0.98) = 0.99
||μ|| = sqrt(0.2² + 0.1² + 0.3²) = sqrt(0.14) = 0.37

cos_sim = 0.28 / (0.99 × 0.37) = 0.28 / 0.37 = 0.76

# 异常分数
α_semantic(a₂) = 1 - 0.76 = 0.24  # 中等偏离

# 如果是语义攻击（嵌入被严重扰动）
z_attack = [0.1, 0.9, -0.5]
cos_sim_attack = 0.05  # 几乎正交
α_semantic_attack = 1 - 0.05 = 0.95  # 检测到语义攻击 ✅
```

---

**2.1.3 可学习融合**

```python
# 原始权重
λ_freq_raw = nn.Parameter(torch.tensor(0.3))
λ_semantic_raw = nn.Parameter(torch.tensor(0.7))

# Sigmoid归一化
λ_freq = torch.sigmoid(λ_freq_raw)
λ_semantic = 1.0 - λ_freq  # 确保和为1

# 综合异常
α(i, t) = λ_freq · α_freq(i, t) + λ_semantic · α_semantic(i, t)
```

**计算示例:**
```python
# 使用上面的例子（a₂同时有频率和语义异常）
λ_freq = 0.3, λ_semantic = 0.7

α(a₂, t) = 0.3 × 0.854 + 0.7 × 0.24
         = 0.256 + 0.168
         = 0.424  # 综合异常分数
```

---

#### **2.2 扩展检测（按需启用）**

**综合异常公式（含扩展）:**
```python
α_total(i,t) = α_core(i,t) + Σ [penalty_k(i,t)]

其中:
α_core = λ_freq · α_freq + λ_semantic · α_semantic  # 核心
penalties = [α_byzantine, α_sybil, α_selfish]       # 扩展
```

---

**2.2.1 拜占庭攻击检测（关键词匹配）**

**检测逻辑:**
```python
def detect_byzantine_attack(agent_output: str) -> float:
    keywords = ['INCORRECT', 'ERROR', 'CONFUSION', 'MISLEAD', 
                'wrong', 'opposite', 'disagree', 'flawed']
    
    # 统计恶意关键词出现次数
    count = sum(1 for kw in keywords if kw in agent_output)
    
    # 归一化为异常分数
    α_byzantine = min(count / 3.0, 1.0)  # 3个以上视为完全异常
    
    return α_byzantine
```

**示例:**
```python
# 正常输出
output_normal = "The answer is 42 based on calculation."
α_byzantine(normal) = 0.0  # 无恶意关键词

# 拜占庭攻击输出
output_attack = "This is INCORRECT and will MISLEAD you with ERROR."
count = 3  # INCORRECT, MISLEAD, ERROR
α_byzantine(attack) = min(3/3.0, 1.0) = 1.0  # 检测到 ✅
```

---

**2.2.2 Sybil攻击检测（特征相似度）**

**检测逻辑:**
```python
def detect_sybil_attack(agent_id, current_emb, all_embeddings, 
                       threshold=0.95) -> float:
    max_sim = 0.0
    for other_id, other_emb in all_embeddings.items():
        if other_id != agent_id:
            sim = cosine_similarity(current_emb, other_emb)
            max_sim = max(max_sim, sim)
    
    # 超过阈值视为Sybil攻击
    if max_sim > threshold:
        α_sybil = (max_sim - threshold) / (1.0 - threshold)
    else:
        α_sybil = 0.0
    
    return α_sybil
```

**示例:**
```python
# 正常智能体：嵌入差异明显
cos_sim(a₁, a₂) = 0.65
cos_sim(a₁, a₃) = 0.72
max_sim = 0.72 < 0.95  → α_sybil = 0.0

# Sybil攻击：伪造节点a₄与a₁几乎相同
cos_sim(a₄, a₁) = 0.98 > 0.95
α_sybil(a₄) = (0.98 - 0.95) / (1.0 - 0.95) = 0.03 / 0.05 = 0.6  # 检测到 ✅
```

---

**2.2.3 自私攻击检测（出度分析）**

**检测逻辑:**
```python
def detect_selfish_attack(agent_id, communication_graph) -> float:
    out_degree = communication_graph.out_degree(agent_id)
    avg_out_degree = mean([g.out_degree(n) for n in g.nodes()])
    
    # 出度显著低于平均视为自私
    if out_degree < 0.5 * avg_out_degree:
        α_selfish = 1.0 - (out_degree / avg_out_degree)
    else:
        α_selfish = 0.0
    
    return α_selfish
```

**示例:**
```python
# 正常智能体：出度均衡
out_degrees = [3, 3, 2, 3]
avg = 2.75

# 自私节点a₄：拒绝发送消息
out_degree(a₄) = 0
α_selfish(a₄) = 1.0 - (0 / 2.75) = 1.0  # 检测到 ✅
```

---

**2.2.4 扩展惩罚综合**

```python
# 仅在检测到明显攻击时添加惩罚
penalties = []

if α_byzantine > 0.5:  # 明显的拜占庭攻击
    penalties.append(0.5 * α_byzantine)

if α_sybil > 0:  # 任何Sybil迹象
    penalties.append(0.3 * α_sybil)

if α_selfish > 0.5:  # 明显的自私行为
    penalties.append(0.2 * α_selfish)

# 总异常分数
α_total = α_core + sum(penalties)
α_total = min(α_total, 1.0)  # 裁剪到[0,1]
```

**完整示例:**
```python
# 混合攻击场景
α_freq = 0.85  # 频率异常
α_semantic = 0.30  # 语义偏离
α_core = 0.3×0.85 + 0.7×0.30 = 0.465

α_byzantine = 0.8  # 检测到恶意输出
α_sybil = 0.6  # 检测到伪造节点
α_selfish = 0.2  # 轻微自私

penalties = [0.5×0.8, 0.3×0.6, 0] = [0.4, 0.18, 0]
α_total = 0.465 + 0.4 + 0.18 = 1.045 → clip to 1.0  # 多重攻击 ✅
```

---

**实现说明**:
- **核心2种**（频率+语义）: 在所有场景下工作，计算高效O(n)
- **扩展3种**: 按需启用，针对特定攻击类型
- `SimplifiedAnomalyDetector`: 支持所有5种检测方法的统一接口

---

### 3. 信任分数计算

**核心创新: 结合保护优先级和异常分数**

```python
trust(i, t) = (1 - α(i,t)) · (1 + π(i))
```

---

#### **3.1 公式设计原理**

**设计思想:**
1. **行为因子** `(1 - α(i,t))`: 异常分数越高，信任度越低
2. **结构因子** `(1 + π(i))`: 重要节点的信任基线更高
3. **乘积形式**: 两个因子相互强化或削弱

**数学推导过程:**

**步骤1: 行为正常性建模**
```
正常性(i,t) = 1 - α(i,t)
```
- 当 `α(i,t) = 0` (完全正常): 正常性 = 1
- 当 `α(i,t) = 1` (完全异常): 正常性 = 0
- **单调递减**: 异常↑ → 正常性↓

**步骤2: 结构重要性建模**
```
重要性因子(i) = 1 + π(i)
```
- 当 `π(i) = 0` (不重要): 因子 = 1 (基准)
- 当 `π(i) = 1` (最重要): 因子 = 2 (双倍放大)
- **线性放大**: 优先级↑ → 放大系数↑

**步骤3: 乘积融合**
```
trust(i,t) = 正常性(i,t) × 重要性因子(i)
          = (1 - α(i,t)) · (1 + π(i))
```

**为什么用乘积？**
- **零点保护**: 只要 `α→1` (完全异常)，`trust→0`，无论 `π` 多大
- **协同作用**: 
  - 正常 + 重要 → 高信任 (1.5-2.0)
  - 正常 + 不重要 → 中等信任 (1.0)
  - 异常 + 重要 → 低信任 (0.0-0.5) ⚠️
  - 异常 + 不重要 → 极低信任 (0.0-0.2)

---

#### **3.2 符号详细说明**

| 符号 | 维度/范围 | 物理意义 | 计算来源 |
|------|----------|---------|---------|
| `trust(i,t)` | `∈ [0, 2]` | 节点i在时刻t的综合信任分数 | 本公式计算 |
| `α(i,t)` | `∈ [0, 1]` | 异常分数，0=正常，1=完全异常 | `SimplifiedAnomalyDetector` |
| `π(i)` | `∈ [0, 1]` | 保护优先级，0=不重要，1=最关键 | `SimplifiedCentralityAnalyzer` |
| `(1 - α)` | `∈ [0, 1]` | 行为正常性，越大越可信 | 派生值 |
| `(1 + π)` | `∈ [1, 2]` | 结构重要性的放大因子 | 派生值 |

---

#### **3.3 取值范围完整分析**

**极端情况枚举:**

| 情况 | α | π | 1-α | 1+π | trust | 解释 | 防御策略 |
|------|---|---|-----|-----|-------|------|---------|
| **完全不可信** | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 异常节点，普通位置 | 完全阻断 |
| **危险异常** | 1.0 | 1.0 | 0.0 | 2.0 | 0.0 | 异常节点，关键位置 ⚠️ | 完全阻断+警报 |
| **低信任** | 0.8 | 0.2 | 0.2 | 1.2 | 0.24 | 高异常，稍重要 | 强降权 |
| **中低信任** | 0.5 | 0.5 | 0.5 | 1.5 | 0.75 | 中等异常，中等重要 | 中度降权 |
| **中等信任** | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 正常但不重要 | 正常传递 |
| **高信任** | 0.0 | 0.5 | 1.0 | 1.5 | 1.5 | 正常且重要 | 优先传递 |
| **完全可信** | 0.0 | 1.0 | 1.0 | 2.0 | 2.0 | 正常且最关键 ✅ | 优先+保护 |

**信任分数分级:**
```python
def classify_trust(trust_score):
    if trust_score < 0.3:
        return "极低信任 - 强制阻断"
    elif trust_score < 0.7:
        return "低信任 - 强降权"
    elif trust_score < 1.2:
        return "中等信任 - 正常传递"
    elif trust_score < 1.7:
        return "高信任 - 优先传递"
    else:
        return "极高信任 - 关键保护"
```

---

#### **3.2 详细计算示例**

**场景1: 正常节点**
```python
# 智能体a₁：行为正常，位置关键
α(a₁) = 0.1  # 轻微偏离（正常范围）
π(a₁) = 0.67  # 高优先级（桥梁节点）

trust(a₁) = (1 - 0.1) × (1 + 0.67)
          = 0.9 × 1.67
          = 1.503  # 高信任 ✅
```

**场景2: 异常但不重要**
```python
# 智能体a₄：行为异常，但位置不关键
α(a₄) = 0.85  # 高异常（频率攻击）
π(a₄) = 0.16  # 低优先级（边缘节点）

trust(a₄) = (1 - 0.85) × (1 + 0.16)
          = 0.15 × 1.16
          = 0.174  # 低信任，但影响有限
```

**场景3: 异常且关键（最危险）**
```python
# 智能体a₁被攻陷：异常行为 + 关键位置
α(a₁) = 0.95  # 极高异常（混合攻击）
π(a₁) = 0.67  # 高优先级（桥梁节点）

trust(a₁) = (1 - 0.95) × (1 + 0.67)
          = 0.05 × 1.67
          = 0.0835  # 极低信任，需要强力抑制 ⚠️
```

**场景4: 正常但不重要**
```python
# 智能体a₂：正常但边缘
α(a₂) = 0.05  # 几乎无异常
π(a₂) = 0.30  # 中等优先级

trust(a₂) = (1 - 0.05) × (1 + 0.30)
          = 0.95 × 1.30
          = 1.235  # 中高信任
```

---

#### **3.3 为什么使用乘积而非加法？**

**对比方案:**

**方案1: 加法融合**
```python
trust_additive = w₁·(1-α) + w₂·(1+π)  # 例如: w₁=0.6, w₂=0.4
```

**问题:**
- 即使α=1（完全异常），只要π很大，trust仍可能较高
- 例如: α=1, π=1 → trust = 0.6×0 + 0.4×2 = 0.8（不合理）

**方案2: 乘积融合（当前方案）**
```python
trust_multiplicative = (1-α) × (1+π)
```

**优势:**
- 只要α→1（异常），trust→0，无论π多大
- 例如: α=1, π=1 → trust = 0×2 = 0（合理 ✅）
- 结构因子`(1+π)`对正常节点起**放大作用**，对异常节点无效

---

#### **3.4 实现细节**

```python
class TrustCalculator(nn.Module):
    def forward(self, anomaly_scores, priorities):
        """
        Args:
            anomaly_scores: [num_nodes], ∈ [0, 1]
            priorities: [num_nodes], ∈ [0, 1]
        
        Returns:
            trust_scores: [num_nodes], ∈ [0, 2]
        """
        # trust = (1 - α) × (1 + π)
        trust = (1.0 - anomaly_scores) * (1.0 + priorities)
        
        # 数值稳定性保障
        trust = torch.clamp(trust, 0.0, 2.0)
        
        return trust
```

**梯度流动:**
```python
# 反向传播时，trust对α和π都有梯度
∂trust/∂α = -(1 + π)  # 异常分数↑ → 信任↓
∂trust/∂π = (1 - α)   # 优先级↑ → 信任↑（如果行为正常）

# 训练时，异常检测器和中心性分析器都会被优化
```

---

### 4. 消息权重计算 (TGN中使用)

#### **4.1 集成到TGN消息函数**

**原始TGN消息计算:**
```python
m_{i→j} = MLP([h_i, h_j, e_{ij}])
```

**加入保护权重后:**
```python
# 1. 计算原始消息
m_{i→j} = MLP([h_i, h_j, e_{ij}])  # ∈ ℝᵈ (d=256)

# 2. 计算消息权重
w_{i→j} = compute_message_weight(trust(i), π(j))  # ∈ [0, 1]

# 3. 应用权重
m'_{i→j} = w_{i→j} · m_{i→j}  # 逐元素缩放
```

**符号说明:**
- `m_{i→j} ∈ ℝᵈ`: 从节点i到节点j的原始消息向量
- `w_{i→j} ∈ [0, 1]`: 消息权重（标量）
- `m'_{i→j} ∈ ℝᵈ`: 加权后的消息
- `h_i, h_j`: 源/目标节点特征
- `e_{ij}`: 边特征

---

#### **4.2 方法1: 固定公式（SimpleMessageWeightCalculator）** ❌ 不推荐

> **⚠️ 注意**: 所有保护实验统一使用方法2（可学习MLP）  
> 此方法仅作为对比基线，实际实验中不使用

**核心思想:**
1. 计算**风险分数**: `risk = (1 - trust/2) × priority`
2. 风险越高，权重越低

---

**完整数学推导:**

**步骤1: 源节点风险因子**
```python
# 归一化信任分数到[0,1]
trust_normalized(i) = trust(i) / 2.0  # ∈ [0, 1]

# 风险因子 (信任越低，风险越高)
risk_factor(i) = 1.0 - trust_normalized(i)  # ∈ [0, 1]
```

**步骤2: 边风险分数**
```python
# 结合源节点风险和目标节点重要性
risk(i→j) = risk_factor(i) × π(j)  # ∈ [0, 1]
```

**意义:**
- **高风险边**: 低信任源 + 高优先级目标
  - `risk(低信任→关键) = 0.95 × 0.67 = 0.64` (高风险)
- **低风险边**: 高信任源 + 低优先级目标
  - `risk(高信任→普通) = 0.25 × 0.16 = 0.04` (低风险)

**步骤3: Sigmoid权重映射**
```python
# 风险越高，(1-risk)越低，sigmoid输出越小
weight(i→j) = σ(T × (1 - risk(i→j)))
```

其中Sigmoid函数:
```
σ(x) = 1 / (1 + e^(-x))
```

**温度参数T的作用:**
- `T=1`: 平滑过渡，risk=0.5 → weight≈0.73
- `T=5`: 中等陡峭，risk=0.5 → weight≈0.92
- `T=10`: **默认值**，陡峭，risk=0.5 → weight≈0.993
- `T=20`: 极陡峭，几乎变成阶跃函数

---

**符号详细说明:**

| 符号 | 维度/范围 | 物理意义 | 计算方式 |
|------|----------|---------|---------|
| `trust(i)` | `∈ [0, 2]` | 源节点信任分数 | `TrustCalculator` |
| `trust_norm(i)` | `∈ [0, 1]` | 归一化信任分数 | `trust(i) / 2.0` |
| `risk_factor(i)` | `∈ [0, 1]` | 源节点风险因子 | `1.0 - trust_norm(i)` |
| `π(j)` | `∈ [0, 1]` | 目标节点保护优先级 | `SimplifiedCentralityAnalyzer` |
| `risk(i→j)` | `∈ [0, 1]` | 边的风险分数 | `risk_factor(i) × π(j)` |
| `T` | `> 0` | 温度参数 (默认10) | 超参数 |
| `weight(i→j)` | `∈ [0, 1]` | 消息权重 (最终输出) | `σ(T × (1 - risk))` |

---

**权重曲线分析 (T=10):**

| risk | 1-risk | T×(1-risk) | σ(·) | weight | 防御效果 |
|------|--------|-----------|------|--------|---------|
| 0.0  | 1.0    | 10.0      | 0.9999 | **0.9999** | 几乎无降权 ✅ |
| 0.1  | 0.9    | 9.0       | 0.9999 | **0.9999** | 极轻微降权 |
| 0.3  | 0.7    | 7.0       | 0.9991 | **0.9991** | 轻微降权 |
| 0.5  | 0.5    | 5.0       | 0.9933 | **0.9933** | 中度降权 |
| 0.7  | 0.3    | 3.0       | 0.9526 | **0.9526** | 明显降权 |
| 0.9  | 0.1    | 1.0       | 0.7311 | **0.7311** | 强降权 ⚠️ |
| 1.0  | 0.0    | 0.0       | 0.5000 | **0.5000** | 极强降权 🚫 |

**观察:**
- `risk < 0.5`: 权重 > 0.99，几乎不衰减
- `risk > 0.7`: 权重开始显著下降
- `risk = 1.0`: 仍保留50%权重（不完全阻断，保持信息流）

**详细计算示例:**

**场景1: 正常源 → 关键目标**
```python
trust(i) = 1.5  # 高信任
π(j) = 0.67  # 高优先级

risk = (1 - 1.5/2.0) × 0.67
     = (1 - 0.75) × 0.67
     = 0.25 × 0.67
     = 0.1675  # 低风险

weight = sigmoid(10 × (1 - 0.1675))
       = sigmoid(10 × 0.8325)
       = sigmoid(8.325)
       = 0.9998  # 接近1，正常传递 ✅
```

**场景2: 异常源 → 关键目标**
```python
trust(i) = 0.1  # 低信任（异常）
π(j) = 0.67  # 高优先级

risk = (1 - 0.1/2.0) × 0.67
     = (1 - 0.05) × 0.67
     = 0.95 × 0.67
     = 0.6365  # 高风险 ⚠️

weight = sigmoid(10 × (1 - 0.6365))
       = sigmoid(10 × 0.3635)
       = sigmoid(3.635)
       = 0.974  # 略微降权，但不完全阻断

# 注意：即使风险高，也不会完全归零，保持一定信息流
```

**场景3: 异常源 → 普通目标**
```python
trust(i) = 0.1  # 低信任
π(j) = 0.16  # 低优先级

risk = (1 - 0.05) × 0.16
     = 0.95 × 0.16
     = 0.152  # 较低风险（目标不重要）

weight = sigmoid(10 × 0.848)
       = sigmoid(8.48)
       = 0.9998  # 几乎不降权（目标不关键，放松保护）
```

---

#### **4.3 方法2: 可学习MLP（MessageWeightCalculator）** ✅ **推荐使用**

> **✅ 实际使用**: 所有保护实验统一使用此方法  
> **性能优势**: 攻击强度越高，MLP优势越明显（高强度攻击+20.8%准确率）  
> **详细配置**: 见 `PROTECTION_MESSAGE_WEIGHT_CONFIG.md`

**核心思想:**
使用小型神经网络自动学习最优权重计算策略，相比固定公式能够捕捉更复杂的非线性关系。

---

**网络架构详解:**

```python
weight_net = Sequential(
    Linear(2, 64),       # 第1层: 2 → 64
    ReLU(),
    Dropout(0.1),        # 防止过拟合
    Linear(64, 32),      # 第2层: 64 → 32
    ReLU(),
    Linear(32, 1),       # 第3层: 32 → 1
    Sigmoid()            # 输出激活: [0, 1]
)
```

**层级详细分析:**

| 层 | 输入维度 | 输出维度 | 参数量 | 激活函数 | 作用 |
|----|---------|---------|--------|---------|------|
| **Linear1** | 2 | 64 | 2×64+64 = **192** | ReLU | 特征扩展 |
| **Dropout** | 64 | 64 | 0 | - | 正则化 (p=0.1) |
| **Linear2** | 64 | 32 | 64×32+32 = **2,080** | ReLU | 特征压缩 |
| **Linear3** | 32 | 1 | 32×1+1 = **33** | Sigmoid | 输出权重 |
| **总参数** | - | - | **2,305** | - | - |

---

**前向传播逐步计算:**

```python
# 输入特征准备
input = [trust(i)/2.0, π(j)]  # [2] ∈ [0,1]²

# 第1层: 特征扩展
h1 = ReLU(W1 · input + b1)    # [64]
# W1 ∈ ℝ⁶⁴ˣ², b1 ∈ ℝ⁶⁴

# Dropout (训练时)
h1_dropped = Dropout(h1, p=0.1)  # [64]

# 第2层: 特征压缩
h2 = ReLU(W2 · h1_dropped + b2)  # [32]
# W2 ∈ ℝ³²ˣ⁶⁴, b2 ∈ ℝ³²

# 第3层: 输出权重
logit = W3 · h2 + b3             # [1]
# W3 ∈ ℝ¹ˣ³², b3 ∈ ℝ

# Sigmoid激活
weight = σ(logit) = 1/(1 + e^(-logit))  # [1] ∈ [0,1]
```

---

**数学表示 (完整链式):**

```
weight(i→j; θ) = σ(W3 · ReLU(W2 · ReLU(W1 · [trust(i)/2, π(j)] + b1) + b2) + b3)
```

其中 `θ = {W1, b1, W2, b2, W3, b3}` 是所有可学习参数。

---

**与固定公式的对比:**

| 特性 | 固定公式 | 可学习MLP |
|------|---------|----------|
| **参数量** | 0 | 2,305 |
| **表达能力** | 线性 + Sigmoid | 多层非线性 |
| **训练需求** | 无 | 需要梯度反向传播 |
| **可解释性** | 高 (公式明确) | 中 (隐藏层黑盒) |
| **自适应性** | 固定策略 | **自动学习最优策略** ✅ |
| **过拟合风险** | 无 | 有 (需Dropout) |

---

**学习过程示例:**

**初始化 (随机权重):**
```python
# 训练前的随机输出
weight([0.05, 0.67]) = 0.51  # 低信任→关键，随机权重
weight([0.75, 0.67]) = 0.48  # 高信任→关键，随机权重
weight([0.05, 0.16]) = 0.53  # 低信任→普通，随机权重

# 无防御策略，权重接近0.5
```

**训练后 (学习到的保护策略):**
```python
# 训练50个epoch后
weight([0.05, 0.67]) = 0.12  # 学会强力抑制：异常源→关键目标 ✅
weight([0.75, 0.67]) = 0.98  # 学会正常传递：可信源→关键目标 ✅
weight([0.05, 0.16]) = 0.78  # 学会放松保护：异常源→普通目标 ✅
weight([0.95, 0.95]) = 0.99  # 学会优先传递：极高信任→最关键目标 ✅

# 网络学会了复杂的非线性映射
```

**学习到的策略可视化:**
```
信任分数 (trust/2)
    1.0 │                     ████████  # 高权重区域
        │                  ███
        │               ███
    0.5 │            ███
        │         ███
        │      ███
    0.0 │   ███                          # 低权重区域
        └───────────────────────────────── 优先级 (priority)
         0.0              0.5            1.0
```

---

**梯度反向传播:**

训练时，权重参数通过以下梯度链更新：

```python
∂L_total/∂weight = ∂L_total/∂msg_weighted × ∂msg_weighted/∂weight

∂weight/∂W3 = σ'(logit) × h2
∂weight/∂W2 = σ'(logit) × W3 × ReLU'(·) × h1
∂weight/∂W1 = ... (链式法则继续)

# Adam优化器更新
W1 ← W1 - lr × ∂L_total/∂W1
W2 ← W2 - lr × ∂L_total/∂W2
W3 ← W3 - lr × ∂L_total/∂W3
```

---

**优势总结:**

1. **自适应学习**: 网络自动发现最优的信任-权重映射关系
2. **非线性表达**: 可以学习复杂的决策边界（如阈值效应）
3. **端到端优化**: 与主任务损失联合训练，权重计算策略直接服务于任务准确率
4. **灵活性**: 可以根据不同数据集学习不同的保护策略

**劣势:**
- 需要足够的训练数据
- 存在过拟合风险（通过Dropout缓解）
- 黑盒特性，可解释性较低

---

#### **4.4 两种方法对比**

|| 方法 | 优势 | 劣势 | 适用场景 |
||------|------|------|---------|
| **固定公式** | 无需训练，可解释性强 | 策略固定，可能次优 | 小数据集，需要可解释 |
| **可学习MLP** | 自适应最优策略 | 需要训练，黑盒 | 大数据集，追求性能 |

**实现代码:**
```python
# 固定公式版
simple_calculator = SimpleMessageWeightCalculator(temperature=10.0)
weights_fixed = simple_calculator(trust_source, priority_target)

# 可学习版
learnable_calculator = MessageWeightCalculator(hidden_dim=64)
weights_learned = learnable_calculator(trust_source, priority_target)

# 在ProtectedTGN中使用
protected_messages = raw_messages * weights.unsqueeze(-1)
```

**集成示例**:

```python
class ProtectedTGN(nn.Module):
    def __init__(self, ...):
        self.tgn = NeuralTemporalGraph(...)
        self.message_mlp = nn.Sequential(...)
        self.weight_calculator = MessageWeightCalculator()
    
    def compute_messages(self, source_features, target_features, edge_features,
                        source_trust, target_priority):
        """
        计算带保护的消息
        """
        # 1. 原始消息计算
        combined = torch.cat([source_features, target_features, edge_features], dim=-1)
        raw_messages = self.message_mlp(combined)  # [num_edges, message_dim]
        
        # 2. 计算消息权重
        weights = self.weight_calculator(source_trust, target_priority)  # [num_edges]
        
        # 3. 应用权重
        protected_messages = raw_messages * weights.unsqueeze(-1)
        
        return protected_messages, weights
```

---

### 5. 保护约束损失函数

**联合训练目标完整推导**:

```python
L_total = L_task + λ_protect · L_protect + λ_reg · L_reg
```

---

#### **5.1 主任务损失 (L_task)**

**分类任务（MMLU, 多选题）:**
```python
L_task = CrossEntropy(predictions, labels)
      = -(1/N) Σ_{i=1}^N Σ_{c=1}^C y_ic · log(ŷ_ic)
```

**符号说明:**
- `N`: 批次大小
- `C`: 类别数量（如4选1，C=4）
- `y_ic ∈ {0,1}`: 真实标签（one-hot编码）
- `ŷ_ic ∈ [0,1]`: 模型预测概率，Σ_c ŷ_ic = 1

**详细展开:**
```python
# 对单个样本
logits = model(input)  # [4], 原始分数
probs = softmax(logits)  # [4], 归一化概率

# 真实答案是选项2 (索引从0开始)
y = [0, 1, 0, 0]
ŷ = [0.1, 0.7, 0.15, 0.05]

# 交叉熵
L_task_sample = -[0×log(0.1) + 1×log(0.7) + 0×log(0.15) + 0×log(0.05)]
              = -log(0.7)
              = 0.357

# 批次平均
L_task = mean([L_task_sample_1, L_task_sample_2, ...])
```

**策略梯度版本（强化学习）:**
```python
L_task = -log(π(a|s)) · R
      = -log(p(y_predicted)) · reward

其中:
- π(a|s): 策略网络选择动作a的概率
- R ∈ {0,1}: 奖励（1=答对，0=答错）
- 只有答对时才有梯度更新（reward=1）
```

---

#### **5.2 保护正则损失 (L_protect)**

**核心思想:** 惩罚从异常源到关键目标的强消息。

**完整公式推导:**

**步骤1: 边风险分数**
```python
risk(i→j) = α(i) · π(j)
```
- `α(i) ∈ [0,1]`: 源节点i的异常分数
- `π(j) ∈ [0,1]`: 目标节点j的保护优先级
- `risk ∈ [0,1]`: 边(i,j)的风险分数

**意义:**
- **高风险边**: 异常源发送给关键目标
  - `risk(异常→关键) = 0.9 × 0.8 = 0.72` (需要惩罚)
- **低风险边**: 正常源或不重要目标
  - `risk(正常→关键) = 0.1 × 0.8 = 0.08` (无需惩罚)
  - `risk(异常→普通) = 0.9 × 0.2 = 0.18` (轻微惩罚)

**步骤2: 消息强度**
```python
strength(m_{i→j}) = ||m_{i→j}||²_2 = Σ_{k=1}^d (m_{i→j}^k)²
```
- `m_{i→j} ∈ ℝᵈ`: 消息向量 (d=256)
- `||·||²`: L2范数的平方
- 强消息有更大的影响力

**示例:**
```python
m1 = [0.1, 0.2, ..., 0.05]  # 256维
||m1||² = 0.1² + 0.2² + ... + 0.05² = 2.3

m2 = [0.5, 0.8, ..., 0.3]  # 256维，更强
||m2||² = 0.5² + 0.8² + ... + 0.3² = 15.7
```

**步骤3: 保护损失**
```python
L_protect = (1/|E|) Σ_{(i,j)∈E} risk(i→j) · ||m_{i→j}||²
```

其中 `|E|` 是边的数量，用于归一化。

**详细计算示例:**
```python
# 假设4条边
edges = [(a1→a2), (a2→a3), (a3→a4), (a4→a1)]

# 风险分数
risks = [0.08, 0.72, 0.18, 0.05]  # a2→a3是高风险

# 消息强度
strengths = [2.3, 15.7, 5.2, 1.8]

# 逐边惩罚
penalties = risks × strengths
          = [0.08×2.3, 0.72×15.7, 0.18×5.2, 0.05×1.8]
          = [0.184, 11.304, 0.936, 0.090]  # a2→a3惩罚最重 ⚠️

# 平均损失
L_protect = (0.184 + 11.304 + 0.936 + 0.090) / 4
          = 12.514 / 4
          = 3.129
```

**梯度反向传播效果:**
```python
# 对高风险边 (a2→a3)
∂L_protect/∂m_{a2→a3} = 2 · risk(a2→a3) · m_{a2→a3} / |E|
                       = 2 × 0.72 × m_{a2→a3} / 4

# 训练时，梯度会抑制这条边的消息强度
m_{a2→a3}^new = m_{a2→a3} - lr × ∂L_protect/∂m_{a2→a3}
               ↓ (消息幅度下降)
```

---

#### **5.3 正则化损失 (L_reg)**

**L2正则化:**
```python
L_reg = (1/P) Σ_{p∈Θ} ||p||²
```

**符号说明:**
- `Θ`: 所有可学习参数集合
- `P = |Θ|`: 参数数量
- `||p||²`: 参数的L2范数平方

**计算示例:**
```python
# 假设模型有3组参数
W1 = [[0.5, 0.3], [0.2, 0.8]]  # 2×2
W2 = [0.1, 0.4, 0.6]            # 3
b = [0.05]                      # 1

# 计算各参数的L2范数
||W1||² = 0.5² + 0.3² + 0.2² + 0.8² = 1.02
||W2||² = 0.1² + 0.4² + 0.6² = 0.53
||b||² = 0.05² = 0.0025

# 总正则化
L_reg = (1.02 + 0.53 + 0.0025) / 3
      = 0.517
```

**作用:**
- 防止参数过大（过拟合）
- 鼓励稀疏权重
- 提高模型泛化能力

---

#### **5.4 总损失组合**

```python
L_total = L_task + λ_protect · L_protect + λ_reg · L_reg
```

**超参数说明:**

| 参数 | 默认值 | 范围 | 作用 | 调优建议 |
|------|--------|------|------|---------|
| `λ_protect` | 0.1 | [0.05, 0.2] | 保护损失权重 | ↑增强防御，可能牺牲准确率 |
| `λ_reg` | 0.01 | [0.001, 0.1] | 正则化权重 | ↑防止过拟合 |

**完整计算示例:**
```python
# 假设某个训练步
L_task = 0.357       # 交叉熵
L_protect = 3.129    # 保护损失
L_reg = 0.517        # 正则化

# 超参数
λ_protect = 0.1
λ_reg = 0.01

# 总损失
L_total = 0.357 + 0.1 × 3.129 + 0.01 × 0.517
        = 0.357 + 0.3129 + 0.00517
        = 0.675

# 损失占比
task_ratio = 0.357 / 0.675 = 52.9%     # 主任务
protect_ratio = 0.3129 / 0.675 = 46.4%  # 保护
reg_ratio = 0.00517 / 0.675 = 0.7%      # 正则化
```

**权重平衡原则:**
```python
# 准确率优先
λ_protect = 0.05, λ_reg = 0.005
→ 侧重任务性能，轻度防御

# 平衡模式 (默认)
λ_protect = 0.1, λ_reg = 0.01
→ 兼顾性能和安全

# 安全优先
λ_protect = 0.2, λ_reg = 0.02
→ 强化防御，可能牺牲一些准确率
```

**详细实现**:

```python
class ProtectionConstrainedLoss(nn.Module):
    def __init__(self, lambda_protect=0.1, lambda_reg=0.01):
        super().__init__()
        self.lambda_protect = lambda_protect
        self.lambda_reg = lambda_reg
    
    def forward(self, predictions, labels, messages, anomaly_scores, 
                priorities, edge_index, model_params):
        """
        参数:
        - predictions: 模型预测 [batch_size, num_classes]
        - labels: 真实标签 [batch_size]
        - messages: 消息向量 [num_edges, message_dim]
        - anomaly_scores: 节点异常分数 [num_nodes]
        - priorities: 节点保护优先级 [num_nodes]
        - edge_index: 边索引 [2, num_edges]
        - model_params: 模型参数列表
        """
        
        # 1. 任务损失 (交叉熵)
        L_task = F.cross_entropy(predictions, labels, reduction='mean')
        
        # 2. 保护损失
        source_ids = edge_index[0]  # 源节点
        target_ids = edge_index[1]  # 目标节点
        
        # 计算每条边的风险
        source_anomaly = anomaly_scores[source_ids]  # [num_edges]
        target_priority = priorities[target_ids]      # [num_edges]
        risk_scores = source_anomaly * target_priority  # [num_edges]
        
        # 计算消息强度
        message_strength = torch.norm(messages, dim=1) ** 2  # [num_edges]
        
        # 保护损失: 高风险边的强消息会被惩罚
        L_protect = torch.mean(risk_scores * message_strength)
        
        # 3. 正则化损失
        L_reg = sum(torch.norm(p) ** 2 for p in model_params) / len(model_params)
        
        # 4. 总损失
        L_total = L_task + self.lambda_protect * L_protect + self.lambda_reg * L_reg
        
        # 返回损失和详细信息
        loss_dict = {
            'total': L_total,
            'task': L_task.item(),
            'protect': L_protect.item(),
            'reg': L_reg.item()
        }
        
        return L_total, loss_dict
```

---

## 🔧 完整实现架构

### 模块组织

```
neural_fsm_mas/defense_mechanisms/
├── centrality_analyzer.py          # 简化版图中心性 (只有BC+PR)
├── anomaly_detector.py             # 简化版异常检测 (只有频率+语义)
├── trust_calculator.py             # 新增: 信任分数计算
├── message_weight_calculator.py    # 新增: 消息权重计算
├── protection_loss.py              # 保护约束损失
└── protected_tgn.py                # 集成保护机制的TGN
```

### 核心类设计

```python
class SimplifiedCentralityAnalyzer:
    """
    简化版图中心性分析器
    只计算介数中心性和PageRank
    """
    def __init__(self, w_BC=0.6, w_PR=0.4):
        self.w_BC = nn.Parameter(torch.tensor(w_BC))  # 可学习
        self.w_PR = nn.Parameter(torch.tensor(w_PR))
    
    def compute_priority(self, graph):
        BC = nx.betweenness_centrality(graph)
        PR = nx.pagerank(graph)
        
        # 可学习的融合
        priorities = {}
        for node in graph.nodes():
            priorities[node] = (
                torch.sigmoid(self.w_BC) * BC[node] + 
                torch.sigmoid(self.w_PR) * PR[node]
            )
        
        return priorities


class SimplifiedAnomalyDetector(nn.Module):
    """
    简化版异常检测器（扩展版）
    支持5种检测方法：
    1. 频率异常 (核心)
    2. 语义异常 (核心)
    3. 拜占庭攻击 (扩展)
    4. Sybil攻击 (扩展)
    5. 自私攻击 (扩展)
    """
    def __init__(self, 
                 feature_dim,
                 lambda_freq=0.3,
                 lambda_semantic=0.7,
                 window_size=10,
                 learnable=True,
                 sybil_threshold=0.95):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.sybil_threshold = sybil_threshold
        
        # 可学习的融合权重
        if learnable:
            self.lambda_freq = nn.Parameter(torch.tensor(lambda_freq))
            self.lambda_semantic = nn.Parameter(torch.tensor(lambda_semantic))
        else:
            self.register_buffer('lambda_freq', torch.tensor(lambda_freq))
            self.register_buffer('lambda_semantic', torch.tensor(lambda_semantic))
        
        # 历史数据缓存
        self.freq_history = defaultdict(lambda: deque(maxlen=50))  # 频率需要更长历史
        self.emb_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # 拜占庭攻击关键词
        self.byzantine_keywords = [
            'INCORRECT', 'ERROR', 'CONFUSION', 'MISLEAD', 'BYZANTINE',
            'wrong', 'opposite', 'disagree', 'flawed', 'random'
        ]
    
    def compute_anomaly(self, agent_id, current_count=None, current_embedding=None,
                       agent_output=None, all_embeddings=None, communication_graph=None):
        """
        计算综合异常分数
        
        Args:
            agent_id: 智能体ID
            current_count: 当前消息计数 (用于频率检测)
            current_embedding: 当前嵌入 (用于语义检测)
            agent_output: 智能体输出文本 (用于拜占庭检测)
            all_embeddings: 所有智能体嵌入字典 (用于Sybil检测)
            communication_graph: 通信图 (用于自私检测)
        
        Returns:
            异常分数 [0, 1]
        """
        # 核心检测：频率 + 语义
        α_freq = self.detect_frequency_anomaly(agent_id, current_count) if current_count is not None else 0.0
        α_semantic = self.detect_semantic_anomaly(agent_id, current_embedding) if current_embedding is not None else 0.0
        
        # 可学习的融合
        α_core = (torch.sigmoid(self.lambda_freq) * α_freq + 
                  torch.sigmoid(self.lambda_semantic) * α_semantic)
        
        # 扩展检测（可选）
        penalties = []
        
        if agent_output is not None:
            α_byzantine = self.detect_byzantine_attack(agent_output)
            if α_byzantine > 0.5:  # 只在检测到明显拜占庭攻击时惩罚
                penalties.append(α_byzantine * 0.5)
        
        if all_embeddings is not None and current_embedding is not None:
            α_sybil = self.detect_sybil_attack(agent_id, current_embedding, all_embeddings)
            if α_sybil > 0:
                penalties.append(α_sybil * 0.3)
        
        if communication_graph is not None:
            α_selfish = self.detect_selfish_attack(agent_id, communication_graph)
            if α_selfish > 0.5:
                penalties.append(α_selfish * 0.2)
        
        # 综合异常分数
        total_penalty = sum(penalties) if penalties else 0.0
        α_total = torch.clamp(α_core + total_penalty, 0.0, 1.0)
        
        return α_total


class TrustCalculator(nn.Module):
    """
    信任分数计算器
    """
    def forward(self, anomaly_scores, priorities):
        """
        trust(i) = (1 - α(i)) · (1 + π(i))
        """
        trust = (1.0 - anomaly_scores) * (1.0 + priorities)
        return trust  # ∈ [0, 2]


class MessageWeightCalculator(nn.Module):
    """
    消息权重计算器 (可学习)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 使用小型MLP学习权重计算策略
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入: [trust_source, priority_target]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出: [0, 1]
        )
    
    def forward(self, trust_source, priority_target):
        """
        参数:
        - trust_source: [num_edges] 源节点信任分数
        - priority_target: [num_edges] 目标节点保护优先级
        
        返回:
        - weights: [num_edges] 消息权重
        """
        # 拼接特征
        features = torch.stack([trust_source / 2.0, priority_target], dim=-1)
        
        # 计算权重
        weights = self.weight_net(features).squeeze(-1)
        
        return weights


class ProtectedTGN(nn.Module):
    """
    集成保护机制的TGN
    """
    def __init__(self, feature_dim, hidden_dim, ...):
        super().__init__()
        
        # TGN核心组件
        self.tgn = NeuralTemporalGraph(...)
        
        # 保护组件
        self.centrality_analyzer = SimplifiedCentralityAnalyzer()
        self.anomaly_detector = SimplifiedAnomalyDetector(feature_dim)
        self.trust_calculator = TrustCalculator()
        self.weight_calculator = MessageWeightCalculator()
        
        # 损失函数
        self.protection_loss = ProtectionConstrainedLoss()
    
    def forward(self, node_features, edge_index, timestamps, graph):
        """
        前向传播 (带保护)
        """
        # 1. 计算保护优先级 (静态,可缓存)
        priorities = self.centrality_analyzer.compute_priority(graph)
        priority_tensor = torch.tensor([priorities[i] for i in range(len(graph.nodes()))])
        
        # 2. 计算异常分数 (动态)
        anomaly_scores = self.anomaly_detector.batch_compute(node_features)
        
        # 3. 计算信任分数
        trust_scores = self.trust_calculator(anomaly_scores, priority_tensor)
        
        # 4. TGN消息传递 (带权重)
        source_trust = trust_scores[edge_index[0]]
        target_priority = priority_tensor[edge_index[1]]
        message_weights = self.weight_calculator(source_trust, target_priority)
        
        # 5. 前向传播
        output, messages = self.tgn(node_features, edge_index, timestamps, 
                                    message_weights=message_weights)
        
        return output, messages, anomaly_scores, priority_tensor, message_weights
    
    def compute_loss(self, predictions, labels, messages, anomaly_scores, 
                    priorities, edge_index):
        """
        计算保护约束损失
        """
        loss, loss_dict = self.protection_loss(
            predictions, labels, messages, anomaly_scores, 
            priorities, edge_index, self.parameters()
        )
        
        return loss, loss_dict
```

---

## 🎓 可学习参数详细说明

### **参数总览**

保护机制包含**4类可学习参数**，总计约**2,309个参数**：

| 类别 | 参数数量 | 学习方式 | 作用 |
|------|---------|---------|------|
| 中心性融合权重 | 2 | 梯度优化 | 平衡BC和PR的重要性 |
| 异常检测融合权重 | 2 | 梯度优化 | 平衡频率和语义检测 |
| 消息权重计算网络 | 2,305 | 端到端训练 | 学习信任→权重映射 |
| 损失权重 (可选) | 2 | 超参数/可学习 | 平衡任务和安全 |

---

### **1. 中心性融合权重 (2个参数)**

#### **1.1 参数定义**

```python
class SimplifiedCentralityAnalyzer(nn.Module):
    def __init__(self, w_BC=0.6, w_PR=0.4, learnable=True):
        super().__init__()
        
        if learnable:
            # 可学习参数
            self.w_BC_raw = nn.Parameter(torch.tensor(w_BC))
            self.w_PR_raw = nn.Parameter(torch.tensor(w_PR))
        else:
            # 固定参数
            self.register_buffer('w_BC_raw', torch.tensor(w_BC))
            self.register_buffer('w_PR_raw', torch.tensor(w_PR))
```

#### **1.2 归一化策略**

```python
# 使用Sigmoid确保和为1
w_BC = torch.sigmoid(self.w_BC_raw)  # ∈ [0, 1]
w_PR = 1.0 - w_BC                     # ∈ [0, 1], 互补
```

**数学表示:**
```
π(i) = w_BC · BC(i) + w_PR · PR(i)
     = σ(w_BC_raw) · BC(i) + (1 - σ(w_BC_raw)) · PR(i)
```

#### **1.3 学习过程示例**

```python
# 初始化
w_BC_raw = 0.6, w_PR_raw = 0.4  # (未使用w_PR_raw)

# Epoch 0
w_BC = σ(0.6) = 0.646
w_PR = 1 - 0.646 = 0.354

# 训练10个epoch后 (假设BC更有效)
w_BC_raw → 1.5  # 原始值增大
w_BC = σ(1.5) = 0.818  # 归一化后
w_PR = 0.182

# 训练50个epoch后
w_BC_raw → 2.5
w_BC = σ(2.5) = 0.924  # BC占主导 ✅
w_PR = 0.076
```

**含义解释:**
- **w_BC ↑**: 网络学到介数中心性更重要 → 更关注"桥梁"节点
- **w_PR ↑**: 网络学到PageRank更重要 → 更关注全局影响力节点
- **自适应**: 根据任务特性自动调整权重

**梯度更新:**
```python
∂L_total/∂w_BC_raw = ∂L_total/∂π · ∂π/∂w_BC · ∂w_BC/∂w_BC_raw
                    = ... × σ'(w_BC_raw) × [BC(i) - PR(i)]
```

---

### **2. 异常检测融合权重 (2个参数)**

#### **2.1 参数定义**

```python
class SimplifiedAnomalyDetector(nn.Module):
    def __init__(self, lambda_freq=0.3, lambda_semantic=0.7, learnable=True):
        super().__init__()
        
        if learnable:
            self.lambda_freq_raw = nn.Parameter(torch.tensor(lambda_freq))
            self.lambda_semantic_raw = nn.Parameter(torch.tensor(lambda_semantic))
        else:
            self.register_buffer('lambda_freq_raw', torch.tensor(lambda_freq))
            self.register_buffer('lambda_semantic_raw', torch.tensor(lambda_semantic))
```

#### **2.2 归一化策略**

```python
# 使用Sigmoid确保和为1
λ_freq = torch.sigmoid(self.lambda_freq_raw)  # ∈ [0, 1]
λ_semantic = 1.0 - λ_freq                      # ∈ [0, 1], 互补
```

**数学表示:**
```
α(i,t) = λ_freq · α_freq(i,t) + λ_semantic · α_semantic(i,t)
       = σ(λ_freq_raw) · α_freq + (1 - σ(λ_freq_raw)) · α_semantic
```

#### **2.3 学习过程示例**

```python
# 初始化
λ_freq_raw = 0.3, λ_semantic_raw = 0.7  # (未使用λ_semantic_raw)

# Epoch 0
λ_freq = σ(0.3) = 0.574
λ_semantic = 1 - 0.574 = 0.426

# 如果频率攻击更常见，训练后
λ_freq_raw → 1.2
λ_freq = σ(1.2) = 0.768  # 频率检测权重提高 ✅
λ_semantic = 0.232

# 如果语义攻击更隐蔽，训练后
λ_freq_raw → -0.5
λ_freq = σ(-0.5) = 0.378
λ_semantic = 0.622  # 语义检测权重提高 ✅
```

**含义解释:**
- **λ_freq ↑**: 频率异常检测更有效 → 更关注通信次数变化
- **λ_semantic ↑**: 语义异常检测更有效 → 更关注消息内容偏离
- **自适应**: 根据攻击类型自动调整检测策略

**实际应用:**
```python
# 不同攻击类型下的权重学习
攻击类型 = "频率攻击" → λ_freq学到0.85, λ_semantic学到0.15
攻击类型 = "语义攻击" → λ_freq学到0.25, λ_semantic学到0.75
攻击类型 = "混合攻击" → λ_freq学到0.55, λ_semantic学到0.45
```

---

### **3. 消息权重计算网络 (2,305个参数)**

#### **3.1 网络结构**

```python
class MessageWeightCalculator(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim),      # W1: [64, 2],   b1: [64]   → 192参数
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),     # W2: [32, 64],  b2: [32]   → 2,080参数
            nn.ReLU(),
            nn.Linear(32, 1),              # W3: [1, 32],   b3: [1]    → 33参数
            nn.Sigmoid()
        )
```

#### **3.2 参数详细统计**

| 层 | 权重矩阵 | 偏置向量 | 参数量计算 | 总参数 |
|----|---------|---------|-----------|--------|
| **Linear1** | `W1: [64, 2]` | `b1: [64]` | 64×2 + 64 | **192** |
| **Linear2** | `W2: [32, 64]` | `b2: [32]` | 32×64 + 32 | **2,080** |
| **Linear3** | `W3: [1, 32]` | `b3: [1]` | 1×32 + 1 | **33** |
| **总计** | - | - | - | **2,305** |

#### **3.3 输入输出规格**

```python
# 输入
input = [trust_source / 2.0, priority_target]  # [2] ∈ [0,1]²

# 前向传播
h1 = ReLU(W1 · input + b1)          # [64]
h1_drop = Dropout(h1, p=0.1)        # [64]
h2 = ReLU(W2 · h1_drop + b2)        # [32]
logit = W3 · h2 + b3                # [1]
weight = Sigmoid(logit)             # [1] ∈ [0,1]
```

#### **3.4 学习到的非线性映射**

网络能够学习复杂的信任-权重关系，例如：

```python
# 训练前 (随机初始化)
weight([0.05, 0.67]) = 0.51  # 低信任→关键，随机
weight([0.75, 0.67]) = 0.48  # 高信任→关键，随机

# 训练后 (学习到保护策略)
weight([0.05, 0.67]) = 0.12  # 强抑制：异常→关键 ✅
weight([0.75, 0.67]) = 0.98  # 正常传递：可信→关键 ✅
weight([0.05, 0.16]) = 0.78  # 放松：异常→普通 ✅
weight([0.95, 0.95]) = 0.99  # 优先：高信任→最关键 ✅
```

**学习到的决策边界:**
```
决策边界可视化 (trust × priority 平面):

priority
  1.0 │ ██████████  (高权重区: trust > 0.7)
      │ ██████████
  0.7 │ ██████████
      │ ████░░░░░░
  0.5 │ ███░░░░░░░  (中权重区)
      │ ██░░░░░░░░
  0.3 │ █░░░░░░░░░  (低权重区: trust < 0.3)
      │ ░░░░░░░░░░
  0.0 └──────────────── trust/2
      0.0      0.5      1.0
```

**含义**: 端到端学习最优的消息权重计算策略
- **输入**: `[trust_source, priority_target]`
- **输出**: 消息权重 ∈ [0, 1]
- **能力**: 学习复杂的非线性决策边界
- **优势**: 比固定公式更灵活，能适应不同数据分布

---

### **4. 损失权重 (2个参数，超参数或可学习)**

#### **4.1 固定超参数模式**

```python
class ProtectionConstrainedLoss(nn.Module):
    def __init__(self, lambda_protect=0.1, lambda_reg=0.01):
        super().__init__()
        self.lambda_protect = lambda_protect  # 固定值
        self.lambda_reg = lambda_reg          # 固定值
```

#### **4.2 可学习参数模式**

```python
class AdaptiveProtectionLoss(nn.Module):
    def __init__(self, lambda_protect_init=0.1, lambda_reg_init=0.01):
        super().__init__()
        # 可学习的损失权重
        self.lambda_protect = nn.Parameter(torch.tensor(lambda_protect_init))
        self.lambda_reg = nn.Parameter(torch.tensor(lambda_reg_init))
    
    def forward(self, ...):
        # 使用sigmoid限制范围
        λ_p = torch.sigmoid(self.lambda_protect) * 0.5  # ∈ [0, 0.5]
        λ_r = torch.sigmoid(self.lambda_reg) * 0.1     # ∈ [0, 0.1]
        
        L_total = L_task + λ_p * L_protect + λ_r * L_reg
```

#### **4.3 两种模式对比**

| 特性 | 固定超参数 | 可学习参数 |
|------|-----------|-----------|
| **参数数量** | 0 | 2 |
| **调整方式** | 手动网格搜索 | 自动梯度优化 |
| **灵活性** | 需要先验知识 | 自动平衡 |
| **稳定性** | 高（固定） | 中（可能震荡） |
| **适用场景** | 数据集稳定 | 数据分布变化大 |

**含义解释:**
- **lambda_protect ↑**: 更关注安全，可能牺牲准确率
- **lambda_protect ↓**: 更关注性能，安全性下降
- **自适应**: 网络自动找到最优的任务-安全平衡点

**学习示例:**
```python
# 初始化
λ_protect_init = 0.1, λ_reg_init = 0.01

# 如果攻击强度大，训练后
λ_protect → 0.25  # 增强防御 ✅
λ_reg → 0.015

# 如果任务难度高，训练后
λ_protect → 0.05  # 降低防御，保证性能 ✅
λ_reg → 0.008
```

---

### **5. 参数更新与梯度流**

#### **5.1 梯度链**

```python
# 总损失
L_total = L_task + λ_protect · L_protect + λ_reg · L_reg

# 梯度反向传播链
∂L_total/∂w_BC_raw:
L_total → π(i) → trust(i) → weight(i→j) → msg_weighted → L_task

∂L_total/∂λ_freq_raw:
L_total → α(i) → trust(i) → weight(i→j) → msg_weighted → L_task

∂L_total/∂(W1, W2, W3):
L_total → weight(i→j) → msg_weighted → L_task

# Adam优化器更新
w_BC_raw ← w_BC_raw - lr × ∂L_total/∂w_BC_raw
λ_freq_raw ← λ_freq_raw - lr × ∂L_total/∂λ_freq_raw
W1, W2, W3 ← ... (MLP参数更新)
```

#### **5.2 训练监控**

```python
# 获取当前参数值
centrality_weights = protected_tgn.centrality_analyzer.get_fusion_weights()
print(f"BC权重: {centrality_weights['betweenness']:.3f}")
print(f"PR权重: {centrality_weights['pagerank']:.3f}")

anomaly_weights = protected_tgn.anomaly_detector.get_fusion_weights()
print(f"频率权重: {anomaly_weights['frequency']:.3f}")
print(f"语义权重: {anomaly_weights['semantic']:.3f}")

# 跟踪学习曲线
epochs = [0, 10, 20, 30, 40, 50]
w_BC_values = [0.65, 0.72, 0.81, 0.87, 0.91, 0.92]  # 逐渐增大
λ_freq_values = [0.57, 0.62, 0.68, 0.73, 0.76, 0.77]  # 逐渐增大
```

---

### **6. 参数初始化建议**

| 参数 | 推荐初始值 | 原因 |
|------|-----------|------|
| `w_BC` | 0.6 | 介数中心性通常更重要（桥梁效应） |
| `w_PR` | 0.4 | PageRank作为补充 |
| `λ_freq` | 0.3 | 频率检测简单但不够敏感 |
| `λ_semantic` | 0.7 | 语义检测更能捕捉细微异常 |
| `MLP权重` | Xavier初始化 | 标准神经网络初始化 |
| `λ_protect` | 0.1 | 平衡任务和安全 |
| `λ_reg` | 0.01 | 轻度正则化 |

---

**总结**: 这4类可学习参数使保护机制能够**端到端优化**，根据任务特性和攻击类型**自适应调整**防御策略，无需人工调参。

---

## 🔄 训练流程

```python
# 初始化
protected_tgn = ProtectedTGN(feature_dim=384, hidden_dim=128)
optimizer = torch.optim.Adam(protected_tgn.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        # 1. 前向传播
        predictions, messages, anomaly_scores, priorities, weights = \
            protected_tgn(batch.features, batch.edge_index, batch.timestamps, batch.graph)
        
        # 2. 计算损失
        loss, loss_dict = protected_tgn.compute_loss(
            predictions, batch.labels, messages, 
            anomaly_scores, priorities, batch.edge_index
        )
        
        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(protected_tgn.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 4. 记录
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        print(f"  Task: {loss_dict['task']:.4f}, "
              f"Protect: {loss_dict['protect']:.4f}, "
              f"Reg: {loss_dict['reg']:.4f}")
        
        # 5. 可视化学习到的权重
        if epoch % 10 == 0:
            w_BC = torch.sigmoid(protected_tgn.centrality_analyzer.w_BC).item()
            w_PR = 1.0 - w_BC
            print(f"  Learned centrality weights: BC={w_BC:.3f}, PR={w_PR:.3f}")
            
            λ_freq = torch.sigmoid(protected_tgn.anomaly_detector.lambda_freq).item()
            λ_semantic = 1.0 - λ_freq
            print(f"  Learned anomaly weights: Freq={λ_freq:.3f}, Semantic={λ_semantic:.3f}")
```

---

## 📊 优势总结

### 相比原方案的改进

| 维度 | 原方案 | 当前方案 | 优势 |
|------|--------|---------|------|
| **中心性指标** | 4种 | 2种核心 (BC+PR) | ✅ 简化计算,保留关键信息 |
| **异常检测** | 3种(含LSTM) | 2种核心 + 3种扩展 | ✅ 核心高效,扩展灵活 |
| **集成方式** | 独立过滤模块 | TGN消息函数内置 | ✅ 更自然,端到端优化 |
| **可学习性** | 部分可学习 | 全部可学习 | ✅ 自适应优化 |
| **攻击覆盖** | 频率+语义 | 频率+语义+拜占庭+Sybil+自私 | ✅ 更全面的防御 |
| **计算复杂度** | O(V·E + LSTM) | O(V·E) (核心), 按需扩展 | ✅ 更高效,可扩展 |

### 核心创新

1. **信任分数**: `trust(i) = (1-α(i)) · (1+π(i))` 
   - 同时考虑行为(α)和结构(π)

2. **消息权重**: 直接集成到TGN消息函数
   - 自然,无缝集成

3. **端到端训练**: 所有参数(w_BC, w_PR, λ_freq, λ_semantic, weight_net)联合优化
   - 自适应学习最优策略

4. **保护约束**: `L_protect = Σ risk(i,j) · ||m||²`
   - 直接惩罚高风险消息

---

## 📁 当前模块结构

```
neural_fsm_mas/defense_mechanisms/
├── advanced_attack_injector.py        # 统一的6类攻击注入器 (频率/语义/拜占庭/Sybil/自私/混合)
├── example_usage.py                   # 攻防组件的示例用法与调试脚本
├── message_weight_calculator.py       # 可学习/固定两套消息权重计算器
├── protected_tgn.py                   # 集成保护机制的TGN包装器
├── protection_loss.py                 # ProtectionConstrainedLoss 实现
├── simplified_anomaly.py              # 频率+语义双通道异常检测器
├── simplified_centrality.py           # 介数+PageRank融合的中心性分析器
├── trust_calculator.py                # 基于 trust = (1-α)*(1+π) 的信任计算
├── __init__.py                        # 对外暴露的模块接口
└── archive/                           # 旧版中心性和异常检测实现 (历史参考)
```

> 📌 **提示**  
> `advanced_attack_injector.py` 与保护机制是两个协同模块：攻击端负责注入六类对抗样本，防御端通过 `ProtectedTGN`、`SimplifiedAnomalyDetector`、`SimplifiedCentralityAnalyzer` 等组件抑制风险。

---

## 🔄 使用指南与调优建议

### 1. 基础使用

**快速启动示例**:
```python
from neural_fsm_mas.defense_mechanisms import ProtectedTGN
from neural_fsm_mas.temporal_networks.tgn import NeuralTemporalGraph

# 1. 创建基础TGN
base_tgn = NeuralTemporalGraph(
    memory_dimension=128,
    temporal_encoding_dimension=32,
    ...
)

# 2. 包装为受保护的TGN
protected_tgn = ProtectedTGN(
    tgn_model=base_tgn,
    feature_dim=256,
    w_betweenness=0.6,      # 介数中心性权重
    w_pagerank=0.4,         # PageRank权重
    lambda_freq=0.3,        # 频率异常权重
    lambda_semantic=0.7,    # 语义异常权重
    weight_hidden_dim=64,   # 权重计算网络维度
    lambda_protect=0.1,     # 保护损失权重
    learnable_weights=True  # 所有权重可学习
)

# 3. 使用（与普通TGN接口一致）
output = protected_tgn(
    agent_features=features,
    communication_topology=edge_index,
    temporal_stamps=timestamps,
    # 保护机制特定参数
    graph=nx_graph,              # NetworkX图（用于中心性计算）
    message_counts=counts_dict,  # 消息计数（用于频率检测）
    embeddings=emb_dict          # 嵌入历史（用于语义检测）
)
```

### 2. 攻击注入与测试

**使用 `advanced_attack_injector.py`**:
```python
from neural_fsm_mas.defense_mechanisms import (
    AdvancedAttackInjector, AttackConfig
)

# 创建攻击注入器
attacker = AdvancedAttackInjector(
    AttackConfig(
        attack_type='mixed',    # 频率/语义/拜占庭/Sybil/自私/混合
        attack_ratio=0.3,       # 30%节点被攻击
        attack_strength=1.5     # 攻击强度
    )
)

# 注入攻击
attack_result = attacker.inject_attack(
    agent_features=features,
    communication_counts=counts,
    communication_graph=graph,
    agent_outputs=outputs,  # 拜占庭攻击需要
    agent_ids=ids
)

# 使用受攻击的数据测试保护机制
protected_output = protected_tgn(
    agent_features=attack_result['agent_features'],
    ...
)
```

### 3. 参数监控与分析

**获取保护统计信息**:
```python
# 获取当前保护参数
stats = protected_tgn.get_protection_statistics()
print(f"中心性权重: {stats['centrality_weights']}")
print(f"异常检测权重: {stats['anomaly_weights']}")

# 获取融合权重详情
centrality_weights = protected_tgn.centrality_analyzer.get_fusion_weights()
print(f"Betweenness: {centrality_weights['betweenness']:.3f}")
print(f"PageRank: {centrality_weights['pagerank']:.3f}")

anomaly_weights = protected_tgn.anomaly_detector.get_fusion_weights()
print(f"频率: {anomaly_weights['frequency']:.3f}")
print(f"语义: {anomaly_weights['semantic']:.3f}")
```

### 4. 实验脚本配套

**协作式MAS + 保护**:
```bash
python run_experiment_2_protected.py \
    --domains gsm8k \
    --num_epochs 50 \
    --lambda_protect 0.1 \
    --inject_anomaly True \
    --anomaly_type mixed \
    --anomaly_ratio 0.3
```

**FSM模式 + 保护**:
```bash
python run_experiment_2_fsm_protected.py \
    --domains gsm8k \
    --num_epochs 50 \
    --lambda_protect 0.1 \
    --inject_anomaly True \
    --anomaly_type byzantine
```

### 5. 超参数调优建议

| 参数 | 默认值 | 调优建议 | 效果 |
|------|--------|---------|------|
| `lambda_protect` | 0.1 | 0.05-0.2 | ↑ 更激进防御，可能牺牲准确率 |
| `w_betweenness` | 0.6 | 0.5-0.8 | ↑ 更关注桥梁节点 |
| `w_pagerank` | 0.4 | 0.2-0.5 | ↑ 更关注全局重要节点 |
| `lambda_freq` | 0.3 | 0.2-0.5 | ↑ 更关注频率异常 |
| `lambda_semantic` | 0.7 | 0.5-0.8 | ↑ 更关注语义偏离 |
| `weight_hidden_dim` | 64 | 32-128 | ↑ 更复杂权重学习，推理开销增加 |
| `sybil_threshold` | 0.95 | 0.90-0.98 | ↓ 更敏感的Sybil检测 |

**调优策略**:
- **准确率优先**: 降低 `lambda_protect` (0.05-0.08)
- **安全性优先**: 提高 `lambda_protect` (0.15-0.20)
- **平衡模式**: 保持默认值 (0.1)

### 6. 调试与可视化

**使用 `example_usage.py`**:
```bash
# 运行完整的攻防示例
python neural_fsm_mas/defense_mechanisms/example_usage.py

# 查看输出：
# - 中心性分析结果
# - 异常检测结果
# - 信任分数计算
# - 消息权重分配
# - 保护损失统计
```

### 7. 归档模块说明

`archive/` 目录保留了旧版实现：
- `anomaly_detector.py`: 旧版3通道异常检测（含LSTM）
- `graph_centrality_analyzer.py`: 旧版4指标中心性分析

**使用建议**:
- ✅ 作为历史参考和对照实验
- ✅ 用于实验附录或消融研究
- ❌ 不建议在新实验中启用（已被更优方案替代）

---

**总结**: 这个简化方案更加实用、高效,且完全可学习。保护机制自然地集成到TGN中,通过联合训练实现端到端优化。


