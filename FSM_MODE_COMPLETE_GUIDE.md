># 🎯 FSM模式完整指南
# Complete Guide to FSM Mode in NeuralFSM

---

## 📚 **目录**

1. [概述](#概述)
2. [架构说明](#架构说明)
3. [核心组件](#核心组件)
4. [使用指南](#使用指南)
5. [训练流程](#训练流程)
6. [API参考](#api参考)
7. [实验示例](#实验示例)
8. [常见问题](#常见问题)

---

## 📖 **概述**

### **什么是FSM模式？**

FSM（Finite State Machine，有限状态机）模式是NeuralFSM项目的核心创新，它将多智能体系统建模为一个有限状态机：

- **每个状态**由一个特定智能体负责
- **智能体输出**决定状态转移
- **监听关系**构成隐式的智能体通信图
- **TGN网络**学习最优的状态转移和监听关系

### **与协作式MAS的区别**

| 特性 | 协作式MAS（原有） | FSM模式（新增） |
|-----|-----------------|---------------|
| **执行方式** | 所有智能体多轮并行交互 | 每个状态由一个智能体串行处理 |
| **通信方式** | 全局共享记忆 | 监听关系（显式通信路径） |
| **控制流** | 固定轮数 (`num_rounds`) | 状态转移 (`<STATE_TRANS>`) |
| **终止条件** | 轮数结束 | 达到最终状态 (`is_final`) |
| **优势** | 充分并行，信息共享 | 结构化推理，可解释性强 |

### **核心理念**

```
任务输入
  ↓
State 0: [Agent_0 负责]
  → Agent_0 推理
  → 输出: "分析... <STATE_TRANS>: 1"
  → 传递给监听者 [Agent_1, Agent_2] 📡
  ↓
State 1: [Agent_1 负责]
  → Agent_1 接收前序消息
  → Agent_1 推理
  → 输出: "设计... <STATE_TRANS>: 2"
  → 传递给监听者 [Agent_2] 📡
  ↓
State 2: [Agent_2 负责, is_final=True]
  → Agent_2 接收前序消息
  → Agent_2 推理
  → 输出: "<|submit|> 最终答案"
  ↓
任务完成 ✅
```

---

## 🏗️ **架构说明**

### **系统架构图**

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuralFSM System                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         FSMStateManager                             │   │
│  │  - 管理状态定义                                      │   │
│  │  - 状态→智能体映射                                   │   │
│  │  - 监听关系（通信路径）                              │   │
│  │  - 状态转移判断                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↕                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    MultiAgentTopologyManager                        │   │
│  │  - initialize_fsm_from_description()                │   │
│  │  - execute_fsm_reasoning()                          │   │
│  │  - 兼容旧模式: execute_multi_agent_reasoning()       │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↕                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         FSMTemporalGraph (TGN)                      │   │
│  │  - FSMTransitionPredictor (状态转移概率)            │   │
│  │  - FSMListenerPredictor (监听关系权重)              │   │
│  │  - FSMStateAgentMatcher (状态-智能体匹配)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↕                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       AgentExecutionNode (智能体节点)               │   │
│  │  - add_predecessor_message() (接收前序消息)         │   │
│  │  - async_execute_reasoning() (执行推理)             │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↕                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       FSM领域提示集                                  │   │
│  │  - FSMGSMPromptSet (GSM8K)                          │   │
│  │  - FSMMMLUPromptSet (MMLU)                          │   │
│  │  - FSMHumanEvalPromptSet (HumanEval)                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 **核心组件**

### **1. FSMStateManager**

**文件**: `neural_fsm_mas/agent_topology/fsm_state_manager.py`

**功能**:
- 管理FSM状态定义
- 维护状态→智能体映射
- 管理监听关系（通信路径图）
- 提取状态转移标记 `<STATE_TRANS>: X`
- 检查最终答案 `<|submit|> Answer`

**核心方法**:

```python
# 添加状态
manager.add_state(
    state_id=0,
    state_name="Analyze",
    responsible_agent_id="agent_0",
    is_initial=True
)

# 设置监听关系
manager.set_listeners(state_id=0, listener_agent_ids=["agent_1", "agent_2"])

# 提取状态转移
next_state = manager.extract_state_transition(output)

# 执行转移
manager.transition_to_state(next_state_id)

# 获取监听关系图
listener_matrix = manager.get_listener_graph()  # [num_states, num_agents]
```

---

### **2. FSMTemporalGraph (FSM-TGN)**

**文件**: `neural_fsm_mas/temporal_networks/fsm_tgn.py`

**功能**:
- 学习状态转移概率矩阵
- 学习监听关系权重矩阵
- 学习状态-智能体最优匹配

**三个核心预测器**:

#### **A. FSMTransitionPredictor**
```python
# 预测状态转移概率
transition_probs = predictor(
    current_state_features,  # [1, state_dim]
    context_features         # [1, state_dim]
)
# 输出: [1, num_states] - 转移到各状态的概率
```

#### **B. FSMListenerPredictor**
```python
# 预测监听关系权重
listener_weights = predictor(
    state_features,   # [num_states, state_dim]
    agent_features    # [num_agents, agent_dim]
)
# 输出: [num_states, num_agents] - 监听权重矩阵
```

#### **C. FSMStateAgentMatcher**
```python
# 计算状态-智能体兼容性
compatibility = matcher(
    state_features,   # [num_states, state_dim]
    agent_features    # [num_agents, agent_dim]
)
# 输出: [num_states, num_agents] - 兼容性分数
```

**训练目标**:
```python
# 策略梯度（REINFORCE）
loss = - log_prob(state_transitions) * reward(accuracy)

# 奖励函数
reward = 1.0 if answer_correct else 0.0

# 优化：
# - 最大化任务准确率
# - 最小化状态转移次数
# - 优化监听关系效率
```

---

### **3. MultiAgentTopologyManager (FSM扩展)**

**文件**: `neural_fsm_mas/agent_topology/multi_agent_topology.py`

**新增方法**:

#### **初始化FSM**
```python
def initialize_fsm_from_description(self, fsm_description: Dict):
    """
    从FSM描述初始化有限状态机
    
    Args:
        fsm_description: {
            'states': [
                {'id': 0, 'name': 'Analyze', 'agent': 'agent_0', 'is_initial': True},
                {'id': 1, 'name': 'Solve', 'agent': 'agent_1'},
                {'id': 2, 'name': 'Verify', 'agent': 'agent_2', 'is_final': True}
            ],
            'listeners': {
                0: ['agent_1', 'agent_2'],  # State 0 → agent_1, agent_2
                1: ['agent_2'],              # State 1 → agent_2
                2: []                        # State 2 是最终状态
            }
        }
    """
```

#### **执行FSM推理**
```python
async def execute_fsm_reasoning(self,
                                task_input: Dict[str, str],
                                max_transitions: int = 10) -> Tuple[str, torch.Tensor]:
    """
    执行FSM模式推理
    
    流程:
    1. 从初始状态开始
    2. 执行当前状态负责的智能体
    3. 提取状态转移或最终答案
    4. 将输出传递给监听智能体
    5. 转移到下一状态
    6. 重复直到最终状态
    
    Returns:
        (final_answer, reasoning_log_probs)
    """
```

---

### **4. FSM领域提示集**

**文件**: `neural_fsm_mas/domain_prompts/prompt_manager.py`

**三个领域的FSM提示集**:

#### **FSMGSMPromptSet (数学题)**
```python
fsm_states = {
    0: 'Problem Understanding',
    1: 'Solution Planning',
    2: 'Calculation',
    3: 'Verification' (Final)
}
```

#### **FSMMMLUPromptSet (选择题)**
```python
fsm_states = {
    0: 'Question Analysis',
    1: 'Knowledge Retrieval',
    2: 'Option Evaluation',
    3: 'Final Decision' (Final)
}
```

#### **FSMHumanEvalPromptSet (代码生成)**
```python
fsm_states = {
    0: 'Requirement Analysis',
    1: 'Algorithm Design',
    2: 'Code Implementation',
    3: 'Code Review' (Final)
}
```

**FSM提示模板**:
```python
def get_answer_prompt(self, question: str, role: str,
                     state_id: int = None,
                     predecessor_messages: List[str] = None) -> str:
    """
    生成包含FSM指导的提示
    
    提示包含:
    1. 前序消息上下文（来自监听的状态）
    2. 角色特定的任务提示
    3. FSM状态转移指导
    4. 可能的下一个状态列表
    """
```

---

## 📘 **使用指南**

### **快速开始**

#### **1. 定义FSM结构**

```python
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager

# 创建拓扑管理器
topology = MultiAgentTopologyManager(
    task_domain="gsm8k",
    language_model_name="gpt-4o-mini",
    agent_role_names=["Agent_0", "Agent_1", "Agent_2", "Agent_3"],
    decision_strategy="final_decision"
)

# 定义FSM
fsm_description = {
    'states': [
        {'id': 0, 'name': 'Analyze', 'agent': 'agent_0', 'is_initial': True},
        {'id': 1, 'name': 'Plan', 'agent': 'agent_1'},
        {'id': 2, 'name': 'Execute', 'agent': 'agent_2'},
        {'id': 3, 'name': 'Verify', 'agent': 'agent_3', 'is_final': True}
    ],
    'listeners': {
        0: ['agent_1', 'agent_2'],
        1: ['agent_2', 'agent_3'],
        2: ['agent_3'],
        3: []
    }
}

# 初始化FSM
topology.initialize_fsm_from_description(fsm_description)
```

#### **2. 执行FSM推理**

```python
import asyncio

# 准备输入
task_input = {
    'task': 'Solve: A shop has 20 apples...',
    'domain': 'gsm8k'
}

# 执行
answer, log_probs = await topology.execute_fsm_reasoning(
    task_input=task_input,
    max_transitions=10
)

print(f"Answer: {answer}")
```

#### **3. 使用FSM-TGN学习FSM结构**

```python
from neural_fsm_mas.temporal_networks.fsm_tgn import FSMTemporalGraph

# 创建FSM-TGN
fsm_tgn = FSMTemporalGraph(
    agent_feature_dim=256,
    state_feature_dim=256,
    num_states=4,
    num_agents=4
)

# 构建特征
agent_features = torch.randn(4, 256)
context_features = torch.randn(256)

# 学习FSM结构
fsm_structure = fsm_tgn.get_fsm_structure(
    agent_features=agent_features,
    context_features=context_features,
    agent_ids=['agent_0', 'agent_1', 'agent_2', 'agent_3'],
    listener_threshold=0.15
)

# 使用学习到的结构
topology.initialize_fsm_from_description(fsm_structure)
```

---

## 🎓 **训练流程**

### **实验1：FSM模式训练**

**脚本**: `run_experiment_1_fsm.py`

#### **基本用法**

```bash
# 训练GSM8K
python run_experiment_1_fsm.py --domain gsm8k --num_epochs 10

# 训练MMLU
python run_experiment_1_fsm.py --domain mmlu --num_epochs 20

# 训练所有领域
python run_experiment_1_fsm.py --domain all --num_epochs 15
```

#### **完整参数**

```bash
python run_experiment_1_fsm.py \
    --domain gsm8k \
    --llm_name gpt-4o-mini \
    --num_epochs 10 \
    --batch_size 4 \
    --num_states 4 \
    --max_transitions 10 \
    --listener_threshold 0.15 \
    --learning_rate 1e-4 \
    --num_train_samples 100 \
    --num_eval_samples 50 \
    --output_dir results/fsm_exp1 \
    --save_interval 5
```

#### **训练流程**

```python
# 1. 初始化训练器
trainer = FSMNeuralMASTrainer(config)

# 2. 准备数据
trainer.prepare_training_data(domains=['gsm8k'])

# 3. 训练循环
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(epoch)
    # metrics: {'avg_reward', 'avg_loss', 'accuracy'}

# 4. 评估
eval_results = await trainer.evaluate_all_domains()

# 5. 保存模型
trainer.save_checkpoint('final_model.pt')
```

#### **训练输出**

```
==================================================
🔄 Epoch 1/10
==================================================

📚 训练领域: gsm8k
🔧 初始化FSM模式 for domain: gsm8k
✅ FSM结构已初始化

FSM State Manager Summary:
  Total States: 4
  Current State: 0
  States:
    State 0: Problem Understanding [INITIAL]
      Agent: agent_0
      Listeners: agent_1, agent_2
    State 1: Solution Planning
      Agent: agent_1
      Listeners: agent_2, agent_3
    ...

  Batch 0: Avg Reward = 0.750
  Batch 10: Avg Reward = 0.825
  ...

📊 Epoch 1 完成:
   平均奖励: 0.780
   平均损失: 0.352
   准确率: 78.00%
```

---

## 📚 **API参考**

### **FSMStateManager API**

```python
class FSMStateManager:
    def __init__(self, agent_ids: List[str])
    
    def add_state(self, state_id: int, state_name: str, 
                 responsible_agent_id: str,
                 is_initial: bool = False,
                 is_final: bool = False)
    
    def set_listeners(self, state_id: int, listener_agent_ids: List[str])
    
    def extract_state_transition(self, output: str) -> Optional[int]
    
    def check_final_answer(self, output: str) -> Optional[str]
    
    def transition_to_state(self, target_state_id: int) -> bool
    
    def get_listener_graph(self) -> torch.Tensor  # [num_states, num_agents]
    
    def reset(self)
```

### **FSMTemporalGraph API**

```python
class FSMTemporalGraph(nn.Module):
    def __init__(self,
                 agent_feature_dim: int = 256,
                 state_feature_dim: int = 256,
                 num_states: int = 4,
                 num_agents: int = 4,
                 hidden_dim: int = 128)
    
    def forward(self,
                agent_features: torch.Tensor,
                communication_topology: torch.Tensor,
                current_state_id: Optional[int] = None,
                context_features: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]
    
    def get_fsm_structure(self,
                         agent_features: torch.Tensor,
                         context_features: torch.Tensor,
                         agent_ids: List[str],
                         listener_threshold: float = 0.15
                         ) -> Dict[str, Any]
```

### **MultiAgentTopologyManager FSM API**

```python
class MultiAgentTopologyManager:
    def initialize_fsm_from_description(self, fsm_description: Dict)
    
    async def execute_fsm_reasoning(self,
                                    task_input: Dict[str, str],
                                    max_transitions: int = 10
                                    ) -> Tuple[str, torch.Tensor]
    
    # 兼容旧模式
    async def execute_multi_agent_reasoning(self,
                                          task_input: Dict[str, str],
                                          num_interaction_rounds: int = 3
                                          ) -> List[Any]
```

---

## 💡 **实验示例**

### **示例1：简单FSM推理**

```python
import asyncio
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager

async def simple_fsm_example():
    # 创建拓扑
    topology = MultiAgentTopologyManager(
        task_domain="gsm8k",
        language_model_name="gpt-4o-mini",
        agent_role_names=["Analyzer", "Solver", "Verifier"],
        decision_strategy="final_decision"
    )
    
    # 定义3状态FSM
    fsm_desc = {
        'states': [
            {'id': 0, 'name': 'Analyze', 'agent': 'agent_0', 'is_initial': True},
            {'id': 1, 'name': 'Solve', 'agent': 'agent_1'},
            {'id': 2, 'name': 'Verify', 'agent': 'agent_2', 'is_final': True}
        ],
        'listeners': {
            0: ['agent_1'],
            1: ['agent_2'],
            2: []
        }
    }
    
    topology.initialize_fsm_from_description(fsm_desc)
    
    # 执行
    answer, _ = await topology.execute_fsm_reasoning({
        'task': '5 + 3 = ?',
        'domain': 'gsm8k'
    })
    
    print(f"Answer: {answer}")

asyncio.run(simple_fsm_example())
```

### **示例2：训练FSM-TGN**

```python
import torch
import torch.optim as optim
from neural_fsm_mas.temporal_networks.fsm_tgn import FSMTemporalGraph

# 初始化
fsm_tgn = FSMTemporalGraph(
    agent_feature_dim=256,
    state_feature_dim=256,
    num_states=4,
    num_agents=4
)

optimizer = optim.Adam(fsm_tgn.parameters(), lr=1e-4)

# 训练循环
for epoch in range(100):
    # 构建特征
    agent_features = torch.randn(4, 256)
    context_features = torch.randn(256)
    comm_topology = torch.zeros(2, 0, dtype=torch.long)
    
    # 前向传播
    outputs = fsm_tgn(agent_features, comm_topology, 
                     current_state_id=0, 
                     context_features=context_features)
    
    # 计算损失（示例：使用交叉熵）
    transition_probs = outputs['transition_probs']
    target_state = torch.tensor([1])  # 目标状态
    
    loss = F.cross_entropy(transition_probs.unsqueeze(0), target_state)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## ❓ **常见问题**

### **Q1: FSM模式和协作式MAS哪个更好？**

**A**: 各有优势：

- **FSM模式**：
  - ✅ 结构化推理，可解释性强
  - ✅ 适合多步骤任务（数学题、代码生成）
  - ✅ 通信路径清晰
  - ❌ 串行执行，可能较慢

- **协作式MAS**：
  - ✅ 并行执行，速度快
  - ✅ 充分信息共享
  - ❌ 缺乏结构，可解释性弱

**建议**: 根据任务特性选择，或两者结合使用。

---

### **Q2: 如何确定最优的状态数量？**

**A**: 三种方法：

1. **领域知识**: 根据任务类型手动设计
   - 数学题: 4状态（理解→计划→计算→验证）
   - 代码: 4状态（分析→设计→实现→审查）

2. **TGN学习**: 使用FSMTemporalGraph自动学习

3. **超参数搜索**: 尝试不同状态数，选择最优

---

### **Q3: 监听关系如何决定？**

**A**: 三种方式：

1. **手动定义**: 基于任务逻辑设计通信路径
   ```python
   listeners = {
       0: ['agent_1', 'agent_2'],  # State 0 输出给 1 和 2
       1: ['agent_2'],              # State 1 输出给 2
   }
   ```

2. **TGN学习**: FSMListenerPredictor自动学习
   ```python
   listener_weights = fsm_tgn.listener_predictor(state_features, agent_features)
   ```

3. **混合方式**: 手动约束 + TGN优化

---

### **Q4: 如何处理状态转移失败？**

**A**: 系统有多重保障：

1. **重试机制**: `max_retry_attempts=3`
2. **超时保护**: `max_execution_time=600`
3. **最大转移限制**: `max_transitions=10`
4. **错误恢复**: 返回错误信息，记录日志

---

### **Q5: 可以在FSM中跳转到任意状态吗？**

**A**: 可以！智能体可以输出任何状态ID：

```python
# Agent可以决定:
<STATE_TRANS>: 2  # 跳到State 2
<STATE_TRANS>: 0  # 回到State 0 (循环)
<STATE_TRANS>: 3  # 跳到最终状态
```

但建议：
- 训练时约束为前向转移（提高效率）
- 推理时允许回退（提高鲁棒性）

---

### **Q6: 如何集成到现有项目？**

**A**: 非常简单，新旧模式兼容：

```python
# 旧模式（协作式MAS）- 保持不变
result = await topology.execute_multi_agent_reasoning(
    task_input, 
    num_interaction_rounds=3
)

# 新模式（FSM）- 新增功能
topology.initialize_fsm_from_description(fsm_desc)
result = await topology.execute_fsm_reasoning(
    task_input,
    max_transitions=10
)
```

---

## 📄 **相关文档**

- **架构说明**: `FSM_ARCHITECTURE_REFACTORING.md`
- **架构对比**: `ARCHITECTURE_CLARIFICATION.md`
- **快速开始**: `QUICK_START.md`
- **数据集使用**: `DATASET_USAGE_GUIDE.md`
- **保护机制**: `PROTECTION_MECHANISM_FINAL_REPORT.md`

---

## 🎉 **总结**

FSM模式为NeuralFSM项目带来了：

1. ✅ **清晰的推理结构** - 状态机建模
2. ✅ **可解释的通信路径** - 监听关系
3. ✅ **端到端学习** - TGN优化
4. ✅ **向后兼容** - 保留协作式MAS
5. ✅ **灵活扩展** - 支持多领域

**开始使用**:

```bash
# 运行测试
python test_fsm_mode.py

# 开始训练
python run_experiment_1_fsm.py --domain gsm8k --num_epochs 10
```

**祝实验顺利！** 🚀

