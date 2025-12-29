# 🔄 实验一: FSM和通信拓扑优化技术指南

**Experiment 1: FSM and Communication Topology Optimization**

---

## 📋 目录

1. [实验概述](#实验概述)
2. [核心技术](#核心技术)
3. [系统架构](#系统架构)
4. [关键组件详解](#关键组件详解)
5. [优化目标与方法](#优化目标与方法)
6. [训练流程](#训练流程)
7. [实验设置](#实验设置)
8. [评估指标](#评估指标)
9. [使用方法](#使用方法)
10. [常见问题](#常见问题)

---

## 🎯 实验概述

### **实验目标**

实验一是NeuralFSM项目的**基础实验**（Baseline），目标是通过神经网络学习：

1. **最优的FSM状态转移路径** - 学习哪些状态转移序列能产生最好的推理结果
2. **最优的智能体通信拓扑** - 学习哪些智能体之间应该通信，如何通信

### **与实验二的区别**

| 维度 | 实验一 (Baseline) | 实验二 (Protected) |
|------|-------------------|-------------------|
| **目标** | 纯粹优化性能 | 优化性能 + 鲁棒性 |
| **训练器** | `NeuralMASTrainer` | `ProtectedNeuralMASTrainer` |
| **TGN** | `NeuralTemporalGraph` | `ProtectedTGN` |
| **损失函数** | 策略梯度损失 | 策略梯度 + 保护约束 |
| **防御机制** | ❌ 无 | ✅ 有 (中心性+异常检测) |

### **核心创新点**

✅ **动态拓扑学习** - 不是固定的通信结构，而是可学习的  
✅ **时序建模** - 考虑多轮交互的历史信息  
✅ **端到端训练** - 直接从任务反馈优化拓扑  
✅ **策略梯度优化** - 使用REINFORCE算法

---

## 🧠 核心技术

### **1. 神经时序图网络 (Neural Temporal Graph, NTG)**

NTG是实验一的核心技术，用于建模智能体之间的动态通信关系。

#### **技术特点**

```
传统GNN:  固定图结构 + 静态消息传递
    ↓
NTG:      动态图结构 + 时序消息传递 + 记忆机制
```

#### **核心公式**

1. **时序编码**
   ```
   h_temporal(t) = sin(t/10000^(2i/d)) + cos(t/10000^(2i/d))
   ```

2. **智能体状态更新**
   ```
   h_i^(t) = GRU(h_i^(t-1), m_i^(t))
   ```
   其中 `m_i^(t)` 是聚合的消息

3. **消息聚合**
   ```
   m_i^(t) = Σ_{j∈N(i)} α_{ij} · Φ(h_j^(t), e_{ij})
   ```
   其中 `α_{ij}` 是注意力权重

4. **通信概率**
   ```
   P(e_{ij} | h_i, h_j) = σ(MLP([h_i, h_j]))
   ```

### **2. 有限状态机 (FSM) 优化**

#### **FSM结构**

```
状态定义:
- Initial State: 问题分析
- Reasoning States: 数学推理, 分析推理, 对抗推理等
- Final State: 最终决策

状态转移:
- 通过TGN学习最优转移序列
- 每个状态对应一个或多个智能体的激活
```

#### **状态转移学习**

```python
# 伪代码
for each problem:
    current_state = INITIAL
    for round in range(max_rounds):
        # 1. TGN预测通信拓扑
        comm_probs = tgn.compute_communication_probabilities(agent_features)
        comm_topology = sample_topology(comm_probs)
        
        # 2. 执行智能体推理
        reasoning_result = execute_multi_agent_reasoning(
            agents, comm_topology, current_state
        )
        
        # 3. 状态转移
        next_state = fsm_transition(current_state, reasoning_result)
        current_state = next_state
    
    # 4. 计算奖励
    reward = evaluate_final_answer(reasoning_result, ground_truth)
    
    # 5. 策略梯度更新
    loss = -reward * log_prob
    loss.backward()
```

### **3. 策略梯度优化 (REINFORCE)**

#### **核心思想**

```
目标: 最大化期望回报
J(θ) = E[R(τ)]

梯度:
∇J(θ) = E[R(τ) · ∇log π_θ(τ)]

其中:
- τ: 轨迹 (状态序列 + 动作序列)
- R(τ): 累积奖励
- π_θ: 策略网络 (TGN)
```

#### **实现细节**

```python
# 计算策略梯度损失
def compute_policy_gradient_loss(log_probs, rewards):
    """
    Args:
        log_probs: 每一步的对数概率 [num_steps]
        rewards: 任务奖励 (MMLU准确率) [1]
    
    Returns:
        loss: 策略梯度损失
    """
    # 标准化奖励 (减少方差)
    normalized_reward = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # REINFORCE损失
    loss = -normalized_reward * log_probs.sum()
    
    return loss
```

---

## 🏗️ 系统架构

### **整体架构图**

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuralMASTrainer                         │
│                   (实验一训练器)                              │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼────────┐      ┌──────▼──────┐
        │ MultiAgent     │      │ Neural      │
        │ Topology       │◄────►│ Temporal    │
        │ Manager        │      │ Graph (TGN) │
        └───────┬────────┘      └──────┬──────┘
                │                       │
        ┌───────▼────────┐      ┌──────▼──────┐
        │ Reasoning      │      │ Memory      │
        │ Agents (5)     │      │ Bank        │
        │ - Math         │      │ - Agent     │
        │ - Analytical   │      │   States    │
        │ - Decision     │      │ - History   │
        │ - Code         │      └─────────────┘
        │ - Adversarial  │
        └────────────────┘
                │
        ┌───────▼────────┐
        │ FSM States     │
        │ - Initial      │
        │ - Reasoning    │
        │ - Final        │
        └────────────────┘
                │
        ┌───────▼────────┐
        │ MMLU Dataset   │
        │ - Questions    │
        │ - Choices      │
        │ - Answers      │
        └────────────────┘
```

### **数据流**

```
问题输入
   ↓
[文本嵌入] → 智能体特征向量
   ↓
[TGN] → 预测通信拓扑概率
   ↓
[采样] → 具体的通信边
   ↓
[多智能体推理] → 每个智能体生成推理结果
   ↓
[FSM状态转移] → 决定下一步激活哪些智能体
   ↓
[多轮交互] → 反复通信和推理
   ↓
[最终决策] → 选择答案
   ↓
[评估] → 计算准确率 (奖励)
   ↓
[策略梯度] → 更新TGN参数
```

---

## 🔧 关键组件详解

### **1. NeuralTemporalGraph (TGN)**

**文件**: `neural_fsm_mas/temporal_networks/neural_temporal_graph.py`

#### **核心功能**

```python
class NeuralTemporalGraph(nn.Module):
    """
    神经时序图网络
    
    核心功能:
    1. 学习智能体之间的通信概率
    2. 维护智能体的时序记忆
    3. 聚合邻居消息
    4. 更新智能体状态
    """
    
    def __init__(self, 
                 agent_feature_dim: int,
                 memory_dimension: int = 128,
                 temporal_dimension: int = 32,
                 message_dimension: int = 64):
        # 智能体记忆库
        self.agent_memory_bank = AgentMemoryBank(memory_dimension)
        
        # 时间编码器
        self.temporal_encoder = TemporalEncoder(temporal_dimension)
        
        # 通信聚合器
        self.communication_aggregator = CommunicationAggregator(
            agent_feature_dim + memory_dimension + temporal_dimension,
            message_dimension
        )
        
        # 状态更新器 (GRU)
        self.agent_state_updater = nn.GRUCell(
            message_dimension,
            agent_feature_dim + memory_dimension + temporal_dimension
        )
```

#### **前向传播流程**

```python
def forward(self, agent_features, communication_topology, 
            temporal_stamps=None, agent_indices=None):
    """
    Args:
        agent_features: [num_agents, feature_dim] 智能体特征
        communication_topology: [2, num_edges] 通信边
        temporal_stamps: [num_agents] 时间戳
        agent_indices: [num_agents] 智能体索引
    
    Returns:
        updated_features: [num_agents, feature_dim] 更新后的特征
    """
    # 1. 获取记忆
    memory = self.agent_memory_bank.retrieve_memory(agent_indices)
    
    # 2. 时序编码
    temporal_encoding = self.temporal_encoder(temporal_stamps)
    
    # 3. 拼接特征
    combined_features = torch.cat([agent_features, memory, temporal_encoding], dim=-1)
    
    # 4. 消息聚合
    messages = self.communication_aggregator(
        combined_features, communication_topology
    )
    
    # 5. 状态更新
    updated_states = self.agent_state_updater(messages, combined_features)
    
    # 6. 更新记忆
    self.agent_memory_bank.update_memory(agent_indices, updated_states)
    
    return updated_states[:, :self.agent_feature_dim]
```

### **2. MultiAgentTopologyManager**

**文件**: `neural_fsm_mas/agent_topology/multi_agent_topology.py`

#### **核心功能**

```python
class MultiAgentTopologyManager:
    """
    多智能体拓扑管理器
    
    核心功能:
    1. 管理智能体的角色和能力
    2. 采样通信拓扑
    3. 执行多智能体推理
    4. FSM状态管理
    """
    
    def __init__(self, 
                 num_agents: int,
                 agent_types: List[str],
                 tgn_model: NeuralTemporalGraph):
        self.num_agents = num_agents
        self.agent_types = agent_types
        self.tgn = tgn_model
        
        # 创建智能体
        self.agents = self._create_agents()
        
        # FSM状态
        self.fsm_states = self._initialize_fsm()
```

#### **通信拓扑采样**

```python
def sample_communication_topology(self, 
                                  agent_features: torch.Tensor,
                                  current_state: str,
                                  temperature: float = 1.0):
    """
    基于TGN预测采样通信拓扑
    
    Args:
        agent_features: 智能体特征
        current_state: 当前FSM状态
        temperature: 采样温度 (控制探索)
    
    Returns:
        edge_index: [2, num_edges] 采样的边
        log_probs: 对数概率 (用于策略梯度)
    """
    # 1. 计算所有可能边的概率
    probs = self.tgn.compute_communication_probabilities(agent_features)
    # probs: [num_agents, num_agents]
    
    # 2. 考虑FSM状态约束
    state_mask = self._get_state_communication_mask(current_state)
    probs = probs * state_mask
    
    # 3. 温度采样
    probs = probs / temperature
    probs = torch.softmax(probs.flatten(), dim=0)
    
    # 4. 采样k条边
    num_edges = min(self.num_agents * 2, len(probs))
    edge_indices = torch.multinomial(probs, num_edges, replacement=False)
    
    # 5. 转换为edge_index格式
    edge_index = self._indices_to_edge_index(edge_indices, self.num_agents)
    
    # 6. 计算log_probs
    log_probs = torch.log(probs[edge_indices] + 1e-10).sum()
    
    return edge_index, log_probs
```

#### **多智能体推理执行**

```python
async def execute_multi_agent_reasoning(self,
                                       problem: str,
                                       communication_topology: torch.Tensor,
                                       current_state: str):
    """
    执行多智能体协作推理
    
    流程:
    1. 根据拓扑确定激活的智能体
    2. 每个智能体独立推理
    3. 通过通信边交换信息
    4. 汇总推理结果
    
    Args:
        problem: 问题文本
        communication_topology: 通信拓扑
        current_state: 当前FSM状态
    
    Returns:
        reasoning_result: 推理结果
        next_state: 下一个FSM状态
        reasoning_log_probs: 推理的对数概率
    """
    # 1. 确定激活的智能体
    active_agents = self._get_active_agents_for_state(current_state)
    
    # 2. 每个智能体并行推理
    agent_outputs = []
    for agent_id in active_agents:
        # 获取邻居的信息
        neighbor_info = self._get_neighbor_messages(
            agent_id, communication_topology
        )
        
        # 智能体推理
        output = await self.agents[agent_id].reason(
            problem=problem,
            neighbor_messages=neighbor_info,
            context=self.fsm_context[current_state]
        )
        agent_outputs.append(output)
    
    # 3. 汇总结果
    reasoning_result = self._aggregate_reasoning(agent_outputs)
    
    # 4. FSM状态转移
    next_state = self._fsm_transition(current_state, reasoning_result)
    
    # 5. 计算log_probs
    reasoning_log_probs = self._compute_reasoning_log_probs(agent_outputs)
    
    return reasoning_result, next_state, reasoning_log_probs
```

### **3. NeuralMASTrainer**

**文件**: `neural_fsm_mas/train_neural_mas.py`

#### **训练循环**

```python
async def train(self,
                num_epochs: int = 20,
                batch_size: int = 8,
                learning_rate: float = 0.001):
    """
    训练循环
    
    流程:
    1. 遍历训练数据
    2. 执行多轮交互
    3. 计算奖励
    4. 策略梯度更新
    """
    optimizer = torch.optim.Adam(self.tgn.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        for batch in self.data_loader:
            # 1. 批处理数据
            problems = batch['questions']
            ground_truths = batch['answers']
            
            batch_log_probs = []
            batch_rewards = []
            
            for problem, ground_truth in zip(problems, ground_truths):
                # 2. 嵌入问题
                agent_features = self.embed_problem(problem)
                
                # 3. 多轮交互
                current_state = "initial"
                total_log_prob = 0.0
                
                for round in range(self.num_interaction_rounds):
                    # 采样通信拓扑
                    comm_topology, topology_log_prob = \
                        self.topology_manager.sample_communication_topology(
                            agent_features, current_state
                        )
                    
                    # 执行推理
                    reasoning_result, next_state, reasoning_log_prob = \
                        await self.topology_manager.execute_multi_agent_reasoning(
                            problem, comm_topology, current_state
                        )
                    
                    # 累积log_prob
                    total_log_prob += topology_log_prob + reasoning_log_prob
                    
                    # 状态转移
                    current_state = next_state
                    
                    # 更新特征
                    agent_features = self.tgn(
                        agent_features, comm_topology, 
                        temporal_stamps=torch.tensor([round])
                    )
                
                # 4. 计算奖励
                final_answer = reasoning_result['answer']
                reward = 1.0 if final_answer == ground_truth else 0.0
                
                batch_log_probs.append(total_log_prob)
                batch_rewards.append(reward)
            
            # 5. 批量更新
            batch_log_probs = torch.stack(batch_log_probs)
            batch_rewards = torch.tensor(batch_rewards)
            
            # 策略梯度损失
            loss = self._compute_policy_gradient_loss(
                batch_log_probs, batch_rewards
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += batch_rewards.mean().item()
        
        # 打印统计
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_accuracy:.4f}")
```

---

## 🎯 优化目标与方法

### **优化目标**

```
最大化: E[R(τ)]

其中:
- R(τ): 任务奖励 (MMLU准确率)
- τ: 轨迹
  - 状态序列: s_0, s_1, ..., s_T
  - 动作序列: a_0, a_1, ..., a_T
  - 通信拓扑: G_0, G_1, ..., G_T
```

### **可学习的参数**

| 组件 | 参数 | 维度 |
|------|------|------|
| 时序编码器 | `temporal_encoder` | 32 |
| 消息聚合器 | `communication_aggregator` | 64 |
| 状态更新器 | `agent_state_updater` (GRU) | 128 |
| 记忆库 | `agent_memory_bank` | 128 |
| 图神经网络层 | `graph_neural_layers` | 多层 |

### **优化策略**

1. **探索 vs 利用平衡**
   ```python
   # 使用温度参数控制
   temperature = max(0.1, 1.0 * (0.99 ** epoch))
   probs = probs / temperature
   ```

2. **奖励标准化**
   ```python
   # 减少方差
   normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)
   ```

3. **梯度裁剪**
   ```python
   # 防止梯度爆炸
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

---

## 📊 实验设置

### **数据集: MMLU**

**MMLU** (Massive Multitask Language Understanding)
- **任务**: 多选题回答
- **领域**: 57个学科领域
- **难度**: 从初中到专家级
- **格式**: 
  ```json
  {
    "question": "What is the capital of France?",
    "choices": ["London", "Paris", "Berlin", "Madrid"],
    "answer": "B"
  }
  ```

### **默认超参数**

```python
# 模型参数
num_agents = 5                      # 智能体数量
agent_feature_dim = 128             # 特征维度
memory_dimension = 128              # 记忆维度
temporal_dimension = 32             # 时序维度
message_dimension = 64              # 消息维度

# 训练参数
num_epochs = 20                     # 训练轮数
batch_size = 8                      # 批大小
learning_rate = 0.001               # 学习率
num_interaction_rounds = 3          # 交互轮数

# 采样参数
initial_temperature = 1.0           # 初始温度
temperature_decay = 0.99            # 温度衰减
```

### **智能体类型**

1. **Mathematical Reasoning** - 数学推理
2. **Analytical Reasoning** - 分析推理
3. **Decision Making** - 决策制定
4. **Code Generation** - 代码生成
5. **Adversarial Reasoning** - 对抗推理

---

## 📈 评估指标

### **主要指标**

1. **准确率 (Accuracy)**
   ```
   Accuracy = (正确答案数) / (总问题数)
   ```

2. **平均奖励 (Average Reward)**
   ```
   Avg Reward = Σ R(τ) / N
   ```

3. **训练损失 (Training Loss)**
   ```
   Loss = -E[R(τ) · log π_θ(τ)]
   ```

### **辅助指标**

4. **通信效率**
   ```
   Comm Efficiency = (激活边数) / (总可能边数)
   ```

5. **状态转移熵**
   ```
   State Entropy = -Σ P(s_t+1|s_t) log P(s_t+1|s_t)
   ```

6. **收敛速度**
   ```
   Convergence = Epoch达到90%最佳准确率
   ```

---

## 🚀 使用方法

### **基础使用**

```bash
python run_experiment_1_baseline.py \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --num_agents 5 \
    --num_interaction_rounds 3
```

### **完整参数**

```bash
python run_experiment_1_baseline.py \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --num_agents 5 \
    --agent_feature_dim 128 \
    --memory_dimension 128 \
    --temporal_dimension 32 \
    --message_dimension 64 \
    --num_interaction_rounds 3 \
    --initial_temperature 1.0 \
    --temperature_decay 0.99 \
    --dataset mmlu \
    --output_dir experiments/baseline \
    --save_checkpoints \
    --checkpoint_interval 5
```

### **快速测试**

```bash
# 1分钟快速测试
python run_experiment_1_baseline.py \
    --num_epochs 1 \
    --batch_size 4 \
    --num_samples 20
```

### **编程接口**

```python
from neural_fsm_mas.train_neural_mas import NeuralMASTrainer
import asyncio

async def main():
    # 创建训练器
    trainer = NeuralMASTrainer(
        num_agents=5,
        agent_feature_dim=128,
        num_interaction_rounds=3,
        dataset='mmlu'
    )
    
    # 训练
    await trainer.train(
        num_epochs=20,
        batch_size=8,
        learning_rate=0.001
    )
    
    # 保存模型
    trainer.save_checkpoint('experiments/baseline/final_model.pt')
    
    # 评估
    accuracy = await trainer.evaluate(test_data)
    print(f"Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📂 输出文件

### **文件结构**

```
experiments/baseline/
├── checkpoints/
│   ├── epoch_5.pt
│   ├── epoch_10.pt
│   ├── epoch_15.pt
│   └── epoch_20.pt
│
├── logs/
│   ├── training.log           # 训练日志
│   ├── communication_stats.json  # 通信统计
│   └── state_transitions.json    # 状态转移统计
│
└── results.json               # 最终结果
```

### **results.json 示例**

```json
{
  "experiment_name": "baseline",
  "hyperparameters": {
    "num_agents": 5,
    "num_epochs": 20,
    "batch_size": 8,
    "learning_rate": 0.001
  },
  "final_metrics": {
    "accuracy": 0.654,
    "average_reward": 0.654,
    "final_loss": 0.423
  },
  "training_curves": {
    "epochs": [1, 2, 3, ..., 20],
    "accuracy": [0.45, 0.52, 0.58, ..., 0.654],
    "loss": [1.23, 0.98, 0.76, ..., 0.423]
  },
  "communication_stats": {
    "avg_edges_per_round": 6.2,
    "communication_efficiency": 0.62,
    "most_active_agents": [0, 2, 4]
  }
}
```

---

## ❓ 常见问题

### **Q1: 为什么准确率提升缓慢?**

**A**: 可能的原因:
1. 学习率过大或过小
2. 温度衰减过快,探索不足
3. 批大小太小,梯度估计不准
4. 交互轮数不够

**解决方案**:
```bash
# 尝试调整超参数
python run_experiment_1_baseline.py \
    --learning_rate 0.0005 \
    --initial_temperature 1.5 \
    --temperature_decay 0.995 \
    --batch_size 16 \
    --num_interaction_rounds 5
```

### **Q2: 内存不足怎么办?**

**A**: 减少模型大小:
```bash
python run_experiment_1_baseline.py \
    --num_agents 3 \
    --agent_feature_dim 64 \
    --memory_dimension 64 \
    --batch_size 4
```

### **Q3: 如何可视化通信拓扑?**

**A**: 使用可视化工具:
```python
from visualize_results import visualize_communication_topology

visualize_communication_topology(
    'experiments/baseline/logs/communication_stats.json'
)
```

### **Q4: 如何与实验二对比?**

**A**: 运行两个实验后使用:
```bash
python visualize_results.py --compare baseline protected
```

---

## 📚 相关文档

- **QUICK_START_GUIDE.md** - 快速开始
- **FINAL_IMPLEMENTATION_SUMMARY.md** - 完整实现
- **DEFENSE_SIMPLIFIED_DESIGN.md** - 实验二设计
- **EVALUATION_SYSTEM_EXPLAINED.md** - 评估系统

---

## 🎓 技术参考

### **相关论文**

1. **Graph Neural Networks**
   - Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017

2. **Temporal Graph Networks**
   - Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs", ICML 2020

3. **Multi-Agent Reinforcement Learning**
   - Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017

4. **Policy Gradient Methods**
   - Sutton et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation", NeurIPS 1999

---

**文档版本**: 1.0  
**最后更新**: 2025-11-03  
**维护者**: Neural FSM Team

🎉 **准备好开始实验一了！** 🎉

