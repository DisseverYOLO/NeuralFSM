# NeuralFSM 统一数据集集成指南
# Unified Dataset Integration Guide

## ✅ **核心理念**

所有数据集（MMLU、GSM8K、HumanEval）现已**完全集成**到NeuralFSM的训练框架中。

你的优化方法（FSM生成、智能体描述、TGN学习）**对所有数据集通用**，无需为每个数据集编写单独的脚本。

---

## 🏗️ **统一架构**

### **1. 统一数据处理器**
文件: `neural_fsm_mas/training_data/unified_data_processor.py`

```python
from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor

# 一个处理器支持所有数据集
processor = UnifiedDataProcessor("./datasets")

# 加载任意数据集
mmlu_data = processor.load_dataset('mmlu', 'test')
gsm8k_data = processor.load_dataset('gsm8k', 'train')
humaneval_data = processor.load_dataset('humaneval', 'test')

# 获取任务描述（用于生成FSM和智能体）
task_desc = processor.get_task_description('gsm8k')
```

### **2. 统一训练器**
文件: `neural_fsm_mas/train_neural_mas.py`

```python
from neural_fsm_mas.train_neural_mas import NeuralMASTrainer

# 创建训练器
trainer = NeuralMASTrainer(
    config=config,
    dataset_root="./datasets",
    output_dir="./results"
)

# 准备所有数据集
trainer.prepare_training_data(domains=['mmlu', 'gsm8k', 'humaneval'])

# 为每个数据集自动创建合适的拓扑
topology = trainer.create_domain_topology(domain='humaneval')
```

### **3. 统一领域提示集**
文件: `neural_fsm_mas/domain_prompts/prompt_manager.py`

每个数据集都有预定义的智能体配置：

- **MMLU**: 7个智能体（Knowledge Expert, Subject Specialist, ...）
- **GSM8K**: 4个智能体（Math Problem Solver, Calculation Verifier, ...）
- **HumanEval**: 5个智能体（Code Designer, Code Writer, ...）

---

## 🚀 **使用方法**

### **方法1: 训练单个数据集**

```bash
# 训练MMLU
python run_experiment_1_baseline.py --domains mmlu --num_epochs 50

# 训练GSM8K
python run_experiment_1_baseline.py --domains gsm8k --num_epochs 30

# 训练HumanEval
python run_experiment_1_baseline.py --domains humaneval --num_epochs 20
```

### **方法2: 训练多个数据集**

```bash
# 同时训练所有数据集
python run_experiment_1_baseline.py \
    --domains mmlu gsm8k humaneval \
    --num_epochs 50 \
    --batch_size 16
```

### **方法3: 在Python代码中使用**

```python
import asyncio
from neural_fsm_mas.train_neural_mas import NeuralMASTrainer

config = {
    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_rounds': 3,
    'memory_dim': 128,
    'time_dim': 32,
    'llm_name': 'gpt-4o-mini',  # 默认模型
    'use_protection': False,
    'domains': ['mmlu', 'gsm8k', 'humaneval']  # 指定数据集
}

trainer = NeuralMASTrainer(
    config=config,
    dataset_root="./datasets",
    output_dir="./results/my_experiment"
)

# 训练
async def train():
    await trainer.train_all_domains()

asyncio.run(train())
```

---

## 📊 **工作流程（对所有数据集通用）**

```
1. 加载数据
   └─> UnifiedDataProcessor.load_dataset(domain)

2. 获取任务描述
   └─> processor.get_task_description(domain)

3. 生成/加载智能体配置
   ├─> 优先: DomainPromptManager.get_manager(domain)
   └─> 备选: Generate_Agent_Description(task_desc)

4. 生成FSM
   └─> Generate_State_Description(task_desc)
       └─> LLM生成 <STATE_TRANS> 标记

5. 创建多智能体拓扑
   └─> MultiAgentTopologyManager(
           agent_role_names=agents,
           use_neural_temporal_graph=True
       )

6. TGN学习
   ├─> 学习通信权重
   ├─> 学习FSM状态转移
   └─> 策略梯度优化 (REINFORCE)

7. 评估
   └─> 在测试集上评估准确率
```

**关键点**: 这个流程对MMLU、GSM8K、HumanEval完全相同！

---

## 🎯 **添加新数据集**

只需3步：

### **Step 1: 添加数据加载方法**
在 `unified_data_processor.py` 中：

```python
def _load_new_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
    """加载新数据集"""
    data_file = self.dataset_root / "new_dataset" / "data.jsonl"
    
    all_data = []
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            all_data.append({
                'domain': 'new_dataset',
                'question': item['question'],
                'answer': item['answer'],
                'split': split,
                'category': 'New Category'
            })
    
    return all_data
```

### **Step 2: 添加领域提示集**
在 `prompt_manager.py` 中：

```python
class NewDatasetPromptSet(DomainPromptSet):
    def __init__(self):
        super().__init__("new_dataset")
        
        self.role_descriptions = {
            "Agent1": "...",
            "Agent2": "..."
        }
        
        self.role_connections = [
            ("Agent1", "Agent2")
        ]

# 注册
DomainPromptRegistry.register_domain("new_dataset", NewDatasetPromptSet())
```

### **Step 3: 添加任务描述**
在 `unified_data_processor.py` 的 `get_task_description` 方法中：

```python
'new_dataset': """
Task: Description of your task

Goals:
1. ...
2. ...

Available tools: ...
"""
```

**完成！** 现在可以直接运行：
```bash
python run_experiment_1_baseline.py --domains new_dataset --num_epochs 50
```

---

## 📁 **数据集位置**

```
datasets/
├── mmlu/
│   └── data/
│       ├── test/           # 18,738 样本
│       ├── val/            # 2,041 样本
│       ├── dev/            # 428 样本
│       └── auxiliary_train/ # 99,842 样本
│
├── gsm8k/
│   └── gsm8k.jsonl         # 1,319 样本
│
└── humaneval/
    └── humaneval-py.jsonl  # 161 样本
```

---

## 🔧 **配置示例**

### **训练所有数据集 (带不同配置)**

```bash
python run_experiment_1_baseline.py \
    --domains mmlu gsm8k humaneval \
    --dataset_root ./datasets \
    --output_dir ./results/multi_dataset_exp \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --num_rounds 3 \
    --memory_dim 128 \
    --time_dim 32 \
    --llm_name gpt-4o-mini  # 可选，默认为gpt-4o-mini
```

### **快速测试**

```bash
# 每个数据集10个epochs快速测试
python run_experiment_1_baseline.py \
    --domains mmlu gsm8k humaneval \
    --num_epochs 10 \
    --batch_size 8
```

---

## 📊 **结果输出**

训练结果会自动保存：

```
results/
├── experiment1_baseline/
│   ├── mmlu/
│   │   ├── training_history.json
│   │   ├── model_checkpoint.pt
│   │   └── test_results.json
│   │
│   ├── gsm8k/
│   │   └── ...
│   │
│   └── humaneval/
│       └── ...
```

---

## ✅ **验证集成**

运行测试：

```bash
# 验证数据集加载
python verify_datasets.py

# 验证统一处理器
cd neural_fsm_mas/training_data
python unified_data_processor.py

# 快速训练测试
python run_experiment_1_baseline.py \
    --domains mmlu \
    --num_epochs 1 \
    --batch_size 4
```

---

## 🎓 **总结**

### **之前的问题**
- ❌ 需要为每个数据集编写单独的脚本
- ❌ 数据格式不统一
- ❌ 难以添加新数据集

### **现在的解决方案**
- ✅ 一个统一的训练框架支持所有数据集
- ✅ 统一的数据格式和接口
- ✅ 添加新数据集只需3步
- ✅ FSM生成和智能体描述对所有数据集通用
- ✅ TGN学习方法对所有数据集通用

### **核心优势**
1. **代码复用**: 同一套代码处理所有数据集
2. **易于扩展**: 添加新数据集非常简单
3. **统一评估**: 相同的评估指标和输出格式
4. **自动适配**: 根据数据集自动选择合适的智能体配置

---

**结论**: 你完全正确！我们不需要 `run_experiment_with_local_data.py` 这样的独立脚本。所有数据集现在都完美集成到你的优化框架中了。✅

**相关文档**: 
- `QUICK_START.md` - 快速开始
- `DATASET_USAGE_GUIDE.md` - 详细数据集指南
- `FSM_STATE_TRANSITION_EXPLAINED.md` - FSM工作原理

