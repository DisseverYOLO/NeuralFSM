# 多数据集训练指南
## Multi-Dataset Training Guide

---

## 📊 支持的数据集

### 1. MMLU (Massive Multitask Language Understanding)
- **描述**: 多学科知识问答
- **领域**: STEM、人文、社会科学等57个学科
- **任务类型**: 多选题 (A/B/C/D)
- **数据量**: 15,908题
- **状态**: ✅ 已实现

### 2. GSM8K (Grade School Math 8K)
- **描述**: 小学数学问题
- **领域**: 数学推理
- **任务类型**: 开放式数学题
- **数据量**: 8,000+题
- **状态**: ✅ 已实现

### 3. HumanEval
- **描述**: Python代码生成
- **领域**: 编程
- **任务类型**: 函数实现
- **数据量**: 164题
- **状态**: 🔄 待实现

---

## 🔤 文本嵌入功能

### 实现原理（参考GDesigner）

**GDesigner实现**:
```python
# GDesigner/llm/profile_embedding.py
from sentence_transformers import SentenceTransformer

def get_sentence_embedding(sentence):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return embeddings
```

**我们的实现**:
```python
# neural_fsm_mas/embeddings/text_embedding.py
class TextEmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_agents(self, agents: List[dict]) -> torch.Tensor:
        """将智能体描述嵌入为向量"""
        agent_texts = [f"{agent['name']}: {agent['system_prompt']}" 
                      for agent in agents]
        embeddings = self.model.encode(agent_texts, convert_to_tensor=True)
        return embeddings
    
    def encode_states(self, states: List[dict]) -> torch.Tensor:
        """将状态描述嵌入为向量"""
        state_texts = [f"{state['name']}: {state['action']}" 
                      for state in states]
        embeddings = self.model.encode(state_texts, convert_to_tensor=True)
        return embeddings
```

### 嵌入流程

```
智能体/状态描述（文本）
    ↓
Sentence Transformer编码
    ↓
嵌入向量 (384维)
    ↓
TGN输入特征
    ↓
节点表示学习
    ↓
连接概率矩阵
```

### 使用的嵌入模型

| 模型名称 | 维度 | 语言支持 | 用途 |
|---------|------|---------|------|
| `all-MiniLM-L6-v2` (默认) | 384 | 英文 | 快速、高效 |
| `all-mpnet-base-v2` | 768 | 英文 | 更高质量 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 多语言 | 支持中文 |

---

## 🚀 使用方法

### 方法1: 使用多数据集训练脚本（推荐）

#### MMLU训练
```bash
cd neural_fsm_mas

python examples/run_multi_dataset_training.py \
    --dataset mmlu \
    --num_episodes 50 \
    --learning_rate 0.001 \
    --num_train_samples 200 \
    --num_val_samples 100
```

#### GSM8K训练
```bash
python examples/run_multi_dataset_training.py \
    --dataset gsm8k \
    --num_episodes 30 \
    --learning_rate 0.001 \
    --num_train_samples 100 \
    --num_val_samples 50
```

#### HumanEval训练（待实现）
```bash
python examples/run_multi_dataset_training.py \
    --dataset humaneval \
    --num_episodes 20 \
    --num_train_samples 80 \
    --num_val_samples 40
```

---

### 方法2: 使用单独的训练脚本

#### MMLU
```bash
python examples/run_mmlu_training.py
```

#### GSM8K
```bash
python examples/run_gsm8k_training.py
```

---

### 方法3: Python代码调用

```python
from neural_fsm_mas.train_fsm_mas import NeuralFSMTrainer
from neural_fsm_mas.training_data.mmlu_data_processor import MMLUDataProcessor
from neural_fsm_mas.training_data.gsm8k_data_processor import GSM8KDataProcessor

# 加载数据
processor = GSM8KDataProcessor("datasets/gsm8k/gsm8k.jsonl")
processor.load_gsm8k_data()
processor.process_gsm8k_data()
train_data, val_data = processor.split_train_val()

# 格式化
formatted_train = processor.format_for_mas_training(train_data[:100])
formatted_val = processor.format_for_mas_training(val_data[:50])

# 初始化训练器
trainer = NeuralFSMTrainer(
    use_neural_learning=True,
    memory_dimension=128,
    temporal_dimension=32
)

# 训练
results = trainer.train_fsm_mas_with_tgn(
    task_description="Solve math problems",
    training_data=formatted_train,
    validation_data=formatted_val,
    num_episodes=30
)

print(f"训练准确率: {results['final_training_accuracy']:.2%}")
print(f"验证准确率: {results['final_validation_accuracy']:.2%}")
```

---

## 📝 命令行参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--dataset` | str | **必填** | 数据集名称 (mmlu/gsm8k/humaneval) |
| `--data_path` | str | 自动 | 数据集路径（可选，默认使用配置路径） |
| `--num_episodes` | int | 30 | 训练Episodes数量 |
| `--learning_rate` | float | 0.001 | 学习率 |
| `--policy_gradient_weight` | float | 0.7 | 策略梯度损失权重 (α) |
| `--reconstruction_weight` | float | 0.3 | MSE重构损失权重 (β) |
| `--num_train_samples` | int | 100 | 训练样本数量 |
| `--num_val_samples` | int | 50 | 验证样本数量 |
| `--memory_dimension` | int | 128 | TGN记忆维度 |
| `--temporal_dimension` | int | 32 | TGN时间维度 |
| `--random_seed` | int | 42 | 随机种子 |

---

## 📂 数据集准备

### MMLU

**下载方式**:
```bash
# 方法1: 使用官方下载脚本
cd datasets/MMLU
python download.py

# 方法2: 手动下载
# 访问: https://github.com/hendrycks/test
```

**数据结构**:
```
datasets/MMLU/data/
├── test/
│   ├── abstract_algebra_test.csv
│   ├── anatomy_test.csv
│   └── ...
├── dev/
│   ├── abstract_algebra_dev.csv
│   └── ...
└── val/
    ├── abstract_algebra_val.csv
    └── ...
```

---

### GSM8K

**下载方式**:
```bash
# 访问: https://github.com/openai/grade-school-math
# 下载: gsm8k.jsonl
```

**数据格式**:
```jsonl
{"question": "...", "answer": "...\n#### 42"}
{"question": "...", "answer": "...\n#### 123"}
```

**数据结构**:
```
datasets/gsm8k/
└── gsm8k.jsonl
```

---

### HumanEval (待实现)

**下载方式**:
```bash
# 访问: https://github.com/openai/human-eval
```

**数据结构**:
```
datasets/humaneval/
└── humaneval-py.jsonl
```

---

## 🎯 训练示例

### 示例1: 快速测试（小数据集）

```bash
python examples/run_multi_dataset_training.py \
    --dataset gsm8k \
    --num_episodes 10 \
    --num_train_samples 20 \
    --num_val_samples 10 \
    --learning_rate 0.001
```

**预期输出**:
```
🚀 NeuralFSM - 多数据集训练系统
================================================================================

📋 训练配置:
  数据集: GSM8K
  训练Episodes: 10
  训练样本数: 20
  验证样本数: 10
  ...

📂 加载数据集: GSM8K
✅ 数据加载完成:
   训练数据: 20 条
   验证数据: 10 条

🤖 初始化NeuralFSM训练器...
🔤 初始化文本嵌入模型...
✅ 嵌入模型加载完成，维度: 384
✅ 训练器初始化完成

🎯 开始训练...
--------------------------------------------------------------------------------
Episode 1/10:
  🚀 开始生成FSM多智能体系统...
  🤖 Step 1: 生成智能体角色描述...
  ✅ 生成了 3 个智能体
  ...
  
📊 训练完成！最终结果:
================================================================================
✅ 最终训练准确率: 75.00%
✅ 最终验证准确率: 70.00%
✅ 策略梯度损失: 0.3250
✅ MSE重构损失: 0.0542
✅ 组合损失: 0.2438

🎉 GSM8K 训练完成！
```

---

### 示例2: 完整训练（推荐配置）

```bash
# MMLU训练
python examples/run_multi_dataset_training.py \
    --dataset mmlu \
    --num_episodes 50 \
    --num_train_samples 200 \
    --num_val_samples 100 \
    --learning_rate 0.001 \
    --policy_gradient_weight 0.7 \
    --reconstruction_weight 0.3 \
    --memory_dimension 128 \
    --temporal_dimension 32

# GSM8K训练
python examples/run_multi_dataset_training.py \
    --dataset gsm8k \
    --num_episodes 30 \
    --num_train_samples 100 \
    --num_val_samples 50 \
    --learning_rate 0.001 \
    --policy_gradient_weight 0.7 \
    --reconstruction_weight 0.3
```

---

## 📊 训练输出

### 训练日志
```
Episode 5/30:
  智能体交互日志:
    State_0 (ProblemAnalysis):
      - Agent_0 (MathSolver): Analyzing problem structure...
      - Agent_1 (LogicAgent): Identifying key variables...
    
    State_1 (Calculation):
      - Agent_0 (MathSolver): Computing: 2x + 3 = 7 → x = 2
    
  训练准确率: 12/20 (60.00%)
  验证准确率: 7/10 (70.00%)
  
  损失:
    策略梯度损失: 0.4000
    MSE重构损失: 0.0623
    组合损失: 0.2987
  
  梯度范数: 2.3456
```

### 结果文件

**训练记录**: `fsm_mas_training_results_gsm8k_20250122_143052.json`
```json
{
  "task_description": "Solve math problems",
  "num_episodes": 30,
  "final_training_accuracy": 0.75,
  "final_validation_accuracy": 0.70,
  "training_history": [
    {
      "episode": 1,
      "training_accuracy": 0.55,
      "validation_accuracy": 0.60,
      "policy_loss": 0.45,
      "reconstruction_loss": 0.082,
      "combined_loss": 0.3396
    },
    ...
  ]
}
```

**优化系统**: `optimized_fsm_mas_system_gsm8k_20250122_143052.json`
```json
{
  "agents": [...],
  "fsm": {...},
  "optimized_state_topology": {
    "edges": [
      {"from_state": "0", "to_state": "1", "probability": 0.85},
      ...
    ]
  },
  "optimized_communication_topology": {
    "edges": [
      {"from_agent": "0", "to_agent": "1", "probability": 0.92},
      ...
    ]
  }
}
```

---

## 🔍 嵌入可视化示例

### 智能体嵌入
```python
from neural_fsm_mas.embeddings import get_embedding_model

model = get_embedding_model()

agents = [
    {"name": "MathSolver", "system_prompt": "You are an expert in mathematics..."},
    {"name": "LogicAgent", "system_prompt": "You excel at logical reasoning..."}
]

embeddings = model.encode_agents(agents)
print(f"智能体嵌入: {embeddings.shape}")  # [2, 384]
print(f"相似度: {torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)}")
```

### 状态嵌入
```python
states = [
    {"name": "ProblemAnalysis", "action": "Analyze the problem..."},
    {"name": "Solution", "action": "Compute the final answer..."}
]

embeddings = model.encode_states(states)
print(f"状态嵌入: {embeddings.shape}")  # [2, 384]
```

---

## 🆚 与GDesigner的对比

| 特性 | GDesigner | NeuralFSM |
|-----|-----------|-----------|
| **文本嵌入** | ✅ `profile_embedding.py` | ✅ `text_embedding.py` (增强版) |
| **嵌入模型** | SentenceTransformer | SentenceTransformer (可配置) |
| **智能体表示** | 角色描述 → 向量 | 名称+系统提示词 → 向量 |
| **状态表示** | 不适用 | 名称+动作 → 向量 |
| **查询嵌入** | `construct_new_features` | `combine_features_with_query` |
| **数据集** | GSM8K, HumanEval | MMLU, GSM8K, HumanEval |
| **TGN输入** | 嵌入向量 | 嵌入向量 |

---

## ✅ 验证清单

- [x] 文本嵌入模块实现 (`text_embedding.py`)
- [x] 智能体描述→向量嵌入
- [x] 状态描述→向量嵌入
- [x] TGN使用嵌入向量作为输入
- [x] MMLU数据处理器
- [x] GSM8K数据处理器
- [x] 多数据集训练脚本
- [x] 训练脚本预留（MMLU、GSM8K、HumanEval）
- [ ] HumanEval数据处理器（待实现）

---

## 📞 常见问题

### Q1: 如何选择嵌入模型？

**A**: 根据任务需求选择:
- 英文任务: `all-MiniLM-L6-v2` (默认，快速)
- 高质量需求: `all-mpnet-base-v2` (慢，但更准确)
- 中文/多语言: `paraphrase-multilingual-MiniLM-L12-v2`

### Q2: 嵌入维度如何影响训练？

**A**: 
- 384维（默认）: 平衡速度和效果
- 768维: 更丰富的表示，但计算量更大
- TGN会自动适应嵌入维度

### Q3: 如何添加新数据集？

**A**: 
1. 在`training_data/`中创建新的数据处理器
2. 在`run_multi_dataset_training.py`中添加配置
3. 实现数据加载和格式化方法

---

**文档完成时间**: 2025-01-XX  
**版本**: v1.0

