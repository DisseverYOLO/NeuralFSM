# NeuralFSM 简化保护机制实现指南

## 📋 您的需求总结

根据您的要求,我已经设计了一个**简化且实用**的保护机制:

### ✅ 核心改进

1. **简化中心性**: 只用2种 (介数BC + PageRank)
2. **简化异常检测**: 只用2种 (频率 + 语义)
3. **TGN集成**: 在消息函数中引入权重
4. **端到端训练**: 任务回报 + 保护正则项

---

## 🎯 核心公式 (您需要的)

### 1. 节点保护优先级
```python
π(i) = w_BC · BC(i) + w_PR · PR(i)

# w_BC, w_PR 是可学习参数
# BC: 介数中心性 (桥梁节点)
# PR: PageRank (全局重要性)
```

### 2. 异常分数
```python
α(i,t) = λ_freq · α_freq(i,t) + λ_semantic · α_semantic(i,t)

# λ_freq, λ_semantic 是可学习参数
# α_freq: Z-score频率异常
# α_semantic: 余弦相似度语义异常
```

### 3. 信任分数 (您提出的)
```python
trust(i,t) = (1 - α(i,t)) · (1 + π(i))

# 结合异常和重要性
# 范围: [0, 2]
```

### 4. 消息权重 (集成到TGN)
```python
# 在TGN的消息函数中
m'_{i→j} = w_{i→j} · m_{i→j}

# 权重计算 (您的想法)
w_{i→j} = f(trust(i), π(j))

# 可以用简单规则或神经网络学习
```

### 5. 保护约束损失 (联合训练)
```python
L_total = L_task + λ_protect · L_protect + λ_reg · L_reg

# L_task: MMLU任务损失 (准确率)
# L_protect: Σ α(i)·π(j) · ||m_{i→j}||²  (惩罚高风险消息)
# L_reg: L2正则化
```

---

## 🏗️ 架构设计

### 模块组织
```
neural_fsm_mas/defense_mechanisms/
├── simplified_centrality.py         ✅ 已实现 (BC+PR)
├── simplified_anomaly.py            🚧 待实现 (频率+语义)
├── trust_calculator.py              🚧 待实现
├── message_weight_calculator.py     🚧 待实现
├── protection_loss.py               🚧 待实现
├── protected_tgn.py                 🚧 待实现 (集成TGN)
└── __init__.py                      🚧 待实现
```

### 核心流程
```python
# 训练时
for batch in data_loader:
    # 1. 计算保护优先级 (静态,可缓存)
    priorities = centrality_analyzer(graph)
    
    # 2. 检测异常 (动态)
    anomalies = anomaly_detector(node_features)
    
    # 3. 计算信任分数
    trust = (1 - anomalies) * (1 + priorities)
    
    # 4. TGN前向传播 (带消息权重)
    predictions = protected_tgn(
        features, edge_index, 
        trust_scores=trust,
        priorities=priorities
    )
    
    # 5. 计算损失
    loss = L_task + 0.1 * L_protect + 0.01 * L_reg
    
    # 6. 反向传播
    loss.backward()
    optimizer.step()
```

---

## 💻 代码实现 (关键部分)

### A. 简化中心性分析器 (已完成✅)

```python
class SimplifiedCentralityAnalyzer(nn.Module):
    def __init__(self, w_betweenness=0.6, w_pagerank=0.4):
        super().__init__()
        # 可学习权重
        self.w_betweenness = nn.Parameter(torch.tensor(w_betweenness))
        self.w_pagerank = nn.Parameter(torch.tensor(w_pagerank))
    
    def compute_priority_scores(self, graph):
        # 计算BC和PR
        BC = nx.betweenness_centrality(graph)
        PR = nx.pagerank(graph)
        
        # 融合 (sigmoid归一化)
        w_bc = torch.sigmoid(self.w_betweenness)
        w_pr = torch.sigmoid(self.w_pagerank)
        
        priorities = {}
        for node in graph.nodes():
            priorities[node] = (
                w_bc * BC[node] + w_pr * PR[node]
            ) / (w_bc + w_pr)  # 归一化
        
        return priorities
```

**使用示例**:
```python
analyzer = SimplifiedCentralityAnalyzer()
priorities = analyzer.compute_priority_scores(graph)

# 查看学习到的权重
weights = analyzer.get_fusion_weights()
print(f"BC权重: {weights['betweenness']:.3f}")
print(f"PR权重: {weights['pagerank']:.3f}")
```

### B. 简化异常检测器 (待实现)

```python
class SimplifiedAnomalyDetector(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 可学习权重
        self.lambda_freq = nn.Parameter(torch.tensor(0.3))
        self.lambda_semantic = nn.Parameter(torch.tensor(0.7))
        
        # 历史记录
        self.freq_history = defaultdict(lambda: deque(maxlen=10))
        self.emb_history = defaultdict(lambda: deque(maxlen=10))
    
    def detect_frequency_anomaly(self, agent_id, current_count):
        """Z-score检测"""
        history = list(self.freq_history[agent_id])
        if len(history) < 2:
            return 0.0
        
        μ = np.mean(history)
        σ = np.std(history) + 1e-8
        z = abs(current_count - μ) / σ
        
        return torch.sigmoid(torch.tensor(z / 2.0)).item()
    
    def detect_semantic_anomaly(self, agent_id, current_emb):
        """余弦相似度检测"""
        history = list(self.emb_history[agent_id])
        if len(history) < 2:
            return 0.0
        
        center = torch.mean(torch.stack(history), dim=0)
        cos_sim = F.cosine_similarity(
            current_emb.unsqueeze(0),
            center.unsqueeze(0)
        ).item()
        
        return max(0.0, 1.0 - cos_sim)
    
    def compute_anomaly_score(self, agent_id, count, embedding):
        α_freq = self.detect_frequency_anomaly(agent_id, count)
        α_semantic = self.detect_semantic_anomaly(agent_id, embedding)
        
        # 可学习融合
        λ_f = torch.sigmoid(self.lambda_freq)
        λ_s = torch.sigmoid(self.lambda_semantic)
        
        α = (λ_f * α_freq + λ_s * α_semantic) / (λ_f + λ_s)
        return α
```

### C. 信任分数计算 (简单)

```python
class TrustCalculator(nn.Module):
    def forward(self, anomaly_scores, priorities):
        """
        trust(i) = (1 - α(i)) · (1 + π(i))
        """
        trust = (1.0 - anomaly_scores) * (1.0 + priorities)
        return trust  # ∈ [0, 2]
```

### D. 消息权重计算器 (可学习)

```python
class MessageWeightCalculator(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 小型MLP学习权重策略
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [trust_source, priority_target]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0,1]
        )
    
    def forward(self, trust_source, priority_target):
        """
        计算消息权重
        
        输入:
        - trust_source: 源节点信任分数 [num_edges]
        - priority_target: 目标节点保护优先级 [num_edges]
        
        输出:
        - weights: 消息权重 [num_edges]
        """
        # 归一化trust到[0,1]
        trust_norm = trust_source / 2.0
        
        # 拼接特征
        features = torch.stack([trust_norm, priority_target], dim=-1)
        
        # 计算权重
        weights = self.weight_net(features).squeeze(-1)
        
        return weights
```

### E. 保护约束损失

```python
class ProtectionConstrainedLoss(nn.Module):
    def __init__(self, lambda_protect=0.1, lambda_reg=0.01):
        super().__init__()
        self.lambda_protect = lambda_protect
        self.lambda_reg = lambda_reg
    
    def forward(self, predictions, labels, messages, 
                anomaly_scores, priorities, edge_index):
        """
        L_total = L_task + λ_p · L_protect + λ_r · L_reg
        """
        # 1. 任务损失
        L_task = F.cross_entropy(predictions, labels)
        
        # 2. 保护损失
        source_anomaly = anomaly_scores[edge_index[0]]  # [num_edges]
        target_priority = priorities[edge_index[1]]     # [num_edges]
        
        risk = source_anomaly * target_priority  # 风险分数
        message_strength = torch.norm(messages, dim=1) ** 2
        
        L_protect = torch.mean(risk * message_strength)
        
        # 3. 正则化 (在外部计算)
        # L_reg = sum(||θ||²) / len(params)
        
        # 4. 总损失
        L_total = L_task + self.lambda_protect * L_protect
        
        return L_total, {
            'task': L_task.item(),
            'protect': L_protect.item()
        }
```

### F. 集成到TGN

```python
class ProtectedTGN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, ...):
        super().__init__()
        
        # TGN组件
        self.tgn = NeuralTemporalGraph(...)
        
        # 保护组件
        self.centrality = SimplifiedCentralityAnalyzer()
        self.anomaly = SimplifiedAnomalyDetector(feature_dim)
        self.trust_calc = TrustCalculator()
        self.weight_calc = MessageWeightCalculator()
        
        # 损失
        self.protection_loss = ProtectionConstrainedLoss()
    
    def forward(self, node_features, edge_index, graph):
        # 1. 计算优先级
        priorities = self.centrality.compute_priority_scores(graph)
        priority_tensor = torch.tensor([priorities[i] for i in range(len(graph))])
        
        # 2. 计算异常分数
        anomaly_scores = self.anomaly.batch_compute(...)
        
        # 3. 计算信任分数
        trust_scores = self.trust_calc(anomaly_scores, priority_tensor)
        
        # 4. 计算消息权重
        src_trust = trust_scores[edge_index[0]]
        tgt_priority = priority_tensor[edge_index[1]]
        message_weights = self.weight_calc(src_trust, tgt_priority)
        
        # 5. TGN前向传播 (修改消息函数)
        output, messages = self.tgn_with_weights(
            node_features, edge_index, message_weights
        )
        
        return output, messages, anomaly_scores, priority_tensor
    
    def tgn_with_weights(self, features, edge_index, weights):
        """
        修改TGN的消息函数以应用权重
        """
        # 原始消息计算
        raw_messages = self.tgn.compute_messages(features, edge_index)
        
        # 应用权重
        protected_messages = raw_messages * weights.unsqueeze(-1)
        
        # 聚合和更新
        output = self.tgn.aggregate_and_update(protected_messages, edge_index)
        
        return output, protected_messages
```

---

## 🔄 完整训练循环

```python
# 初始化
model = ProtectedTGN(feature_dim=384, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(num_epochs):
    for batch in train_loader:
        # 前向
        pred, msgs, anomalies, priorities = model(
            batch.features, batch.edge_index, batch.graph
        )
        
        # 损失
        loss, loss_dict = model.protection_loss(
            pred, batch.labels, msgs, anomalies, priorities, batch.edge_index
        )
        
        # 加上参数正则化
        l2_reg = sum(p.norm() ** 2 for p in model.parameters()) / len(list(model.parameters()))
        loss = loss + 0.01 * l2_reg
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 日志
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}")
            print(f"  Task Loss: {loss_dict['task']:.4f}")
            print(f"  Protect Loss: {loss_dict['protect']:.4f}")
            print(f"  Total Loss: {loss.item():.4f}")
            
            # 查看学习到的权重
            cent_weights = model.centrality.get_fusion_weights()
            print(f"  Centrality: BC={cent_weights['betweenness']:.3f}, "
                  f"PR={cent_weights['pagerank']:.3f}")
```

---

## 📊 关键优势

| 特性 | 说明 | 优势 |
|------|------|------|
| **简化** | 只用BC+PR, 频率+语义 | ✅ 降低计算开销 |
| **集成** | 权重直接在TGN消息函数 | ✅ 自然,端到端 |
| **可学习** | 所有权重都可学习 | ✅ 自适应优化 |
| **实用** | 信任分数结合结构和行为 | ✅ 更全面 |

---

## 🚀 实现步骤

### 已完成 ✅
1. ✅ 设计文档 (`DEFENSE_SIMPLIFIED_DESIGN.md`)
2. ✅ 简化中心性分析器 (`simplified_centrality.py`)

### 待实现 🚧
3. [ ] 简化异常检测器 (`simplified_anomaly.py`)
4. [ ] 信任分数计算器 (`trust_calculator.py`)
5. [ ] 消息权重计算器 (`message_weight_calculator.py`)
6. [ ] 保护约束损失 (`protection_loss.py`)
7. [ ] ProtectedTGN类 (`protected_tgn.py`)
8. [ ] 修改训练脚本集成
9. [ ] 测试验证

### 预计时间
- 代码实现: 1-2天
- 集成测试: 1天
- 实验验证: 2-3天

---

## 📝 关键参数配置

```python
# 中心性权重 (可学习)
w_betweenness = 0.6  # 介数权重 (更重要)
w_pagerank = 0.4     # PageRank权重

# 异常检测权重 (可学习)
lambda_freq = 0.3      # 频率权重
lambda_semantic = 0.7  # 语义权重 (更重要)

# 损失权重
lambda_protect = 0.1   # 保护损失权重
lambda_reg = 0.01      # 正则化权重

# 模型架构
hidden_dim = 128       # 隐藏层维度
message_weight_hidden = 64  # 权重计算网络
```

---

## 💡 使用建议

1. **初期训练**: 先用较小的`lambda_protect`(如0.05)
2. **微调阶段**: 逐渐增大`lambda_protect`(如0.1-0.2)
3. **监控权重**: 定期查看学习到的中心性和异常检测权重
4. **可视化**: 使用`analyzer.visualize()`查看保护优先级分布

---

## 📚 文档清单

1. ✅ `DEFENSE_SIMPLIFIED_DESIGN.md` - 完整设计方案
2. ✅ `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` - 本实现指南
3. ✅ `simplified_centrality.py` - 中心性分析器代码
4. 🚧 其他模块待实现

---

**总结**: 这个简化方案完全符合您的需求,更加实用且高效。核心创新是**信任分数**和**TGN消息权重集成**,通过端到端训练自动优化所有参数。

有任何问题随时告诉我! 🚀


