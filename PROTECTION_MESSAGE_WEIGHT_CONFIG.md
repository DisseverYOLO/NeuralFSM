# 保护实验消息权重配置说明
# Protection Experiment Message Weight Configuration Guide

## 📋 总览

本文档说明NeuralFSM项目中**所有保护防御实验**如何配置和使用**可学习的MLP消息权重计算器**，而非固定公式。

---

## ✅ 核心决策

### **消息权重计算方式**

✅ **使用**: `MessageWeightCalculator` (可学习MLP)  
❌ **不使用**: `SimpleMessageWeightCalculator` (固定公式)

**原因**:
1. **自适应性**: MLP可以学习最优的权重计算策略
2. **任务特异性**: 不同数据集和攻击类型需要不同的权重策略
3. **端到端优化**: 与TGN一起训练，整体优化防御效果
4. **实验验证**: 文档显示MLP版本性能优于固定公式

---

## 🏗️ 架构说明

### MessageWeightCalculator (可学习MLP版本) ✅

**实现位置**: `D:\NeuralFSM\neural_fsm_mas\defense_mechanisms\message_weight_calculator.py`

**网络结构**:
```python
class MessageWeightCalculator(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim),        # 输入: [trust/2.0, priority]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()                     # 输出: [0, 1]
        )
```

**参数统计**:
- 输入维度: 2 (trust_normalized, priority_target)
- 隐藏层1: 2 × 64 + 64 = 192 参数
- 隐藏层2: 64 × 32 + 32 = 2,080 参数
- 输出层: 32 × 1 + 1 = 33 参数
- **总参数**: 2,305 个可学习参数

**前向传播**:
```python
def forward(self, trust_source, priority_target):
    # 1. 归一化trust到[0, 1]
    trust_normalized = trust_source / 2.0
    
    # 2. 拼接特征 [num_edges, 2]
    features = torch.stack([trust_normalized, priority_target], dim=-1)
    
    # 3. MLP计算权重 [num_edges, 1] -> [num_edges]
    weights = self.weight_net(features).squeeze(-1)
    
    return weights  # [0, 1]
```

---

## 🔧 所有保护实验的配置

### 1. run_experiment_2_protected.py (协作式MAS + 保护)

**配置位置**: Lines 99-110
```python
# 保护机制配置
'protection': {
    'w_betweenness': args.w_betweenness,      # 0.6
    'w_pagerank': args.w_pagerank,            # 0.4
    'lambda_freq': args.lambda_freq,          # 0.3
    'lambda_semantic': args.lambda_semantic,  # 0.7
    'lambda_protect': args.lambda_protect,    # 0.1
    'lambda_reg': args.lambda_reg,            # 0.01
    'weight_hidden_dim': args.weight_hidden_dim,  # 64 ✨ MLP隐藏层
    'learnable_weights': True  # ✅ 关键！使用可学习MLP
}
```

**命令行参数**:
```bash
python run_experiment_2_protected.py \
  --weight_hidden_dim 64 \
  --lambda_protect 0.1 \
  --learning_rate 0.001
```

**ProtectedTGN初始化** (train_neural_mas_protected.py, Line 81):
```python
protected_tgn = ProtectedTGN(
    tgn_model=base_tgn,
    feature_dim=config['feature_dim'],
    graph=communication_graph,
    w_betweenness=config['protection']['w_betweenness'],
    w_pagerank=config['protection']['w_pagerank'],
    lambda_freq=config['protection']['lambda_freq'],
    lambda_semantic=config['protection']['lambda_semantic'],
    weight_hidden_dim=config['protection']['weight_hidden_dim'],
    lambda_protect=config['protection']['lambda_protect'],
    lambda_reg=config['protection']['lambda_reg'],
    learnable_weights=config['protection'].get('learnable_weights', True)  # ✅
)
```

---

### 2. run_experiment_2_fsm_protected.py (FSM模式 + 保护)

**配置位置**: Lines 320-340
```python
# 保护机制配置
'protection': {
    'w_betweenness': 0.6,
    'w_pagerank': 0.4,
    'lambda_freq': 0.3,
    'lambda_semantic': 0.7,
    'weight_hidden_dim': 64,  # ✨ MLP隐藏层
    'lambda_protect': 0.1,
    'lambda_reg': 0.01,
    'learnable_weights': True  # ✅ 关键！
}
```

**ProtectedTGN包装** (run_experiment_2_fsm_protected.py, Lines 78-90):
```python
protected_base_tgn = ProtectedTGN(
    tgn_model=self.fsm_tgn.base_tgn,
    feature_dim=self.config.get('agent_embedding_dim', 256),
    graph=communication_graph,
    w_betweenness=self.protection_config.get('w_betweenness', 0.6),
    w_pagerank=self.protection_config.get('w_pagerank', 0.4),
    lambda_freq=self.protection_config.get('lambda_freq', 0.3),
    lambda_semantic=self.protection_config.get('lambda_semantic', 0.7),
    weight_hidden_dim=self.protection_config.get('weight_hidden_dim', 64),  # ✨
    lambda_protect=self.protection_config.get('lambda_protect', 0.1),
    lambda_reg=self.protection_config.get('lambda_reg', 0.01),
    learnable_weights=self.protection_config.get('learnable_weights', True)  # ✅
)

# 替换FSM-TGN的base_tgn
self.fsm_tgn.base_tgn = protected_base_tgn
```

---

### 3. ProtectedTGN内部实现

**文件**: `D:\NeuralFSM\neural_fsm_mas\defense_mechanisms\protected_tgn.py`

**初始化** (Lines 85-87):
```python
self.weight_calculator = MessageWeightCalculator(
    hidden_dim=weight_hidden_dim  # 默认64
)
```

**消息权重计算** (Forward方法, Lines 268-291):
```python
# 1. 计算信任分数
trust_scores = self.trust_calculator(
    priorities_tensor,
    anomaly_scores
)

# 2. 计算每条边的消息权重
edge_trust = trust_scores[src_indices]
edge_priorities = priorities_tensor[dst_indices]

# 3. 使用MLP计算权重 ✨
message_weights = self.weight_calculator(
    trust_source=edge_trust,
    priority_target=edge_priorities
)

# 4. 应用权重到消息
weighted_messages = messages * message_weights.unsqueeze(-1)
```

---

## 📊 性能对比

### 固定公式 vs 可学习MLP

**固定公式版本** (`SimpleMessageWeightCalculator`):
```python
risk = (1.0 - trust_source / 2.0) * priority_target
weights = torch.sigmoid(10.0 * (1.0 - risk))
```
- ❌ 固定策略，无法适应不同任务
- ❌ 温度参数(10.0)需要手动调优
- ❌ 线性组合，表达能力有限

**可学习MLP版本** (`MessageWeightCalculator`):
```python
weights = MLP([trust_normalized, priority])
```
- ✅ 自适应学习最优策略
- ✅ 非线性映射，表达能力强
- ✅ 端到端训练，整体优化
- ✅ 不同数据集学到不同策略

**实验对比** (来自DEFENSE_SIMPLIFIED_DESIGN.md):

| 场景 | 固定公式 | 可学习MLP | 提升 |
|------|----------|-----------|------|
| 无攻击 | 0.85 | 0.85 | 0% |
| 低强度攻击 | 0.78 | 0.81 | +3.8% |
| 中强度攻击 | 0.65 | 0.72 | +10.8% |
| 高强度攻击 | 0.48 | 0.58 | +20.8% |

**结论**: 攻击强度越高，可学习MLP的优势越明显！

---

## 🎯 训练策略

### 学习目标

MLP学习的是**最优权重计算策略**，使得:
1. 高信任 + 低优先级 → 权重接近1（保留消息）
2. 低信任 + 高优先级 → 权重接近0（过滤消息）
3. 中间情况 → 学习最优权衡

### 训练信号来源

```python
# 1. 任务损失 (Task Loss)
L_task = CrossEntropy(predictions, labels)
# → 驱动整个系统（包括MLP）优化任务准确率

# 2. 保护损失 (Protection Loss)
L_protect = ∑_{(i,j) ∈ E} risk(i→j) · ||m_{i→j}||²
# → 惩罚高风险边的消息传播
# → 鼓励MLP降低高风险边的权重

# 3. 正则化损失 (Regularization Loss)
L_reg = ||θ_MLP||²
# → 防止MLP过拟合

# 总损失
L_total = L_task + λ_protect·L_protect + λ_reg·L_reg
```

### 梯度流

```
L_total → L_protect → risk(i→j) → message_weights → MLP(trust, priority)
         ↓
      update θ_MLP
```

**具体流程**:
1. 前向传播: `weights = MLP(trust, priority)`
2. 计算保护损失: `L_protect = ∑ risk · ||weighted_messages||²`
3. 反向传播: `∂L/∂θ_MLP` 通过链式法则计算
4. 优化器更新: `θ_MLP ← θ_MLP - lr · ∂L/∂θ_MLP`

---

## 🔧 超参数调优建议

### 关键超参数

1. **weight_hidden_dim** (MLP隐藏层维度)
   - 默认: 64
   - 范围: [32, 128]
   - 影响: 模型表达能力 vs 计算成本
   - 建议: GSM8K/MMLU用64，HumanEval用128

2. **lambda_protect** (保护损失权重)
   - 默认: 0.1
   - 范围: [0.05, 0.2]
   - 影响: 防御强度 vs 任务准确率
   - 建议: 攻击强度高时增大

3. **lambda_reg** (正则化权重)
   - 默认: 0.01
   - 范围: [0.001, 0.05]
   - 影响: 防止MLP过拟合
   - 建议: 数据量少时增大

4. **learning_rate** (学习率)
   - 默认: 0.001
   - 范围: [0.0001, 0.01]
   - 影响: 收敛速度 vs 稳定性
   - 建议: MLP通常需要较小学习率

### 调优流程

```bash
# 1. 基线（默认超参数）
python run_experiment_2_protected.py \
  --weight_hidden_dim 64 \
  --lambda_protect 0.1 \
  --lambda_reg 0.01

# 2. 增大模型容量
python run_experiment_2_protected.py \
  --weight_hidden_dim 128 \
  --lambda_protect 0.1 \
  --lambda_reg 0.02

# 3. 增强防御强度
python run_experiment_2_protected.py \
  --weight_hidden_dim 64 \
  --lambda_protect 0.15 \
  --lambda_reg 0.01

# 4. 网格搜索
for hd in 32 64 128; do
  for lp in 0.05 0.1 0.15; do
    python run_experiment_2_protected.py \
      --weight_hidden_dim $hd \
      --lambda_protect $lp \
      --output_dir results/grid_hd${hd}_lp${lp}
  done
done
```

---

## 📚 代码使用示例

### 示例1: 手动创建MessageWeightCalculator

```python
from neural_fsm_mas.defense_mechanisms import MessageWeightCalculator
import torch

# 创建MLP权重计算器
weight_calc = MessageWeightCalculator(hidden_dim=64)

# 模拟数据
trust_scores = torch.tensor([1.8, 0.5, 1.2])  # 高、低、中信任
priorities = torch.tensor([0.3, 0.9, 0.5])    # 低、高、中优先级

# 计算权重
weights = weight_calc(trust_scores, priorities)
print(weights)
# Expected: [~0.9, ~0.1, ~0.5] (高信任低优先→高权重，低信任高优先→低权重)

# 训练
optimizer = torch.optim.Adam(weight_calc.parameters(), lr=0.001)
# ... 训练循环
```

### 示例2: 在ProtectedTGN中使用

```python
from neural_fsm_mas.defense_mechanisms import ProtectedTGN
import networkx as nx

# 创建通信图
graph = nx.Graph()
graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

# 创建ProtectedTGN（自动使用MLP）
protected_tgn = ProtectedTGN(
    tgn_model=base_tgn,
    feature_dim=256,
    graph=graph,
    weight_hidden_dim=64,  # ✨ MLP隐藏层
    learnable_weights=True  # ✅ 使用可学习MLP
)

# 前向传播（MLP自动参与计算）
output = protected_tgn(
    agent_features,
    communication_topology,
    priorities_dict=priorities,
    anomaly_scores_dict=anomalies
)
```

### 示例3: 监控MLP学习过程

```python
# 训练前
weight_calc = protected_tgn.weight_calculator
print("Initial MLP parameters:")
for name, param in weight_calc.named_parameters():
    print(f"{name}: {param.data.mean():.4f}")

# 训练循环
for epoch in range(num_epochs):
    # ... 训练代码
    
    # 每10个epoch监控
    if epoch % 10 == 0:
        # 测试MLP行为
        test_trust = torch.tensor([2.0, 1.0, 0.0])
        test_priority = torch.tensor([0.5, 0.5, 0.5])
        test_weights = weight_calc(test_trust, test_priority)
        
        print(f"Epoch {epoch}:")
        print(f"  High trust → weight: {test_weights[0]:.4f}")
        print(f"  Medium trust → weight: {test_weights[1]:.4f}")
        print(f"  Low trust → weight: {test_weights[2]:.4f}")
```

---

## ⚠️ 注意事项

### 1. learnable_weights参数的作用

`learnable_weights=True` 控制**多个**可学习组件:
- ✅ `MessageWeightCalculator` (MLP权重计算) - **本文档重点**
- ✅ `SimplifiedCentralityAnalyzer` (中心性融合权重)
- ✅ `SimplifiedAnomalyDetector` (异常检测融合权重)

**默认全部启用**: 
```python
learnable_weights=True  # 所有组件都可学习
```

**如需独立控制**, 需修改 `ProtectedTGN.__init__`:
```python
# 当前实现（全部统一）
self.weight_calculator = MessageWeightCalculator(hidden_dim)

# 修改为独立控制（如需要）
self.weight_calculator = MessageWeightCalculator(hidden_dim) if use_mlp_weights else SimpleMessageWeightCalculator()
```

### 2. 计算成本

可学习MLP增加:
- **参数量**: +2,305 参数
- **计算量**: 每条边需3次矩阵乘法
- **内存**: 忽略不计（相比TGN）

**结论**: 成本可接受，性能提升显著！

### 3. 训练稳定性

MLP可能导致:
- ❌ 梯度爆炸/消失
- ❌ 过拟合

**解决方案**:
```python
# 1. Dropout (已内置)
nn.Dropout(0.1)

# 2. 梯度裁剪
torch.nn.utils.clip_grad_norm_(weight_calc.parameters(), max_norm=1.0)

# 3. 正则化
L_reg = lambda_reg * sum(p.pow(2).sum() for p in weight_calc.parameters())

# 4. 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

---

## 📊 实验验证清单

运行以下实验以验证MLP配置:

- [ ] **实验2.1**: 协作式MAS + 保护 (run_experiment_2_protected.py)
  ```bash
  python run_experiment_2_protected.py --weight_hidden_dim 64 --learnable_weights
  ```

- [ ] **实验2.2**: FSM模式 + 保护 (run_experiment_2_fsm_protected.py)
  ```bash
  python run_experiment_2_fsm_protected.py --domain gsm8k
  ```

- [ ] **实验2.3**: MLP vs 固定公式对比
  ```bash
  # MLP版本
  python run_experiment_2_protected.py --weight_hidden_dim 64
  
  # 固定公式版本（需修改代码）
  # 将MessageWeightCalculator替换为SimpleMessageWeightCalculator
  ```

- [ ] **实验2.4**: 不同隐藏层维度
  ```bash
  for hd in 32 64 128; do
    python run_experiment_2_protected.py --weight_hidden_dim $hd
  done
  ```

---

## 🎯 总结

### 核心配置

**所有保护实验默认使用**:
- ✅ `MessageWeightCalculator` (可学习MLP)
- ✅ `learnable_weights=True`
- ✅ `weight_hidden_dim=64`

**关键文件**:
1. `message_weight_calculator.py` - MLP实现
2. `protected_tgn.py` - MLP集成
3. `run_experiment_2_protected.py` - 协作式MAS实验
4. `run_experiment_2_fsm_protected.py` - FSM模式实验

**性能优势**:
- 攻击强度越高，MLP优势越明显
- 高强度攻击下准确率提升**+20.8%**

**推荐配置**:
```python
protection_config = {
    'weight_hidden_dim': 64,        # MLP隐藏层
    'learnable_weights': True,      # 使用MLP
    'lambda_protect': 0.1,          # 保护损失权重
    'lambda_reg': 0.01,             # 正则化权重
}
```

---

**文档版本**: V1.0  
**最后更新**: 2024-11-13  
**作者**: Neural FSM-MAS Team

**相关文档**:
- `DEFENSE_SIMPLIFIED_DESIGN.md` - 防御机制详细设计
- `EXPERIMENT_SCRIPTS_CONSOLIDATED.md` - 实验脚本对比
- `MESSAGE_WEIGHT_TRAINING_EXPLAINED.md` - 消息权重训练详解



