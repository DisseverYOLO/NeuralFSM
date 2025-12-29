# NeuralFSM 数据集使用指南
# Dataset Usage Guide

本指南说明如何使用NeuralFSM项目中的本地数据集运行实验。

---

## 📦 **数据集概览**

NeuralFSM项目现已包含三个完整的数据集:

### 1. **MMLU (Massive Multitask Language Understanding)**
- **位置**: `datasets/mmlu/data/`
- **样本数**: 
  - Test: 18,738 样本 (57个学科)
  - Validation: 2,041 样本
  - Dev: 428 样本
  - Auxiliary Train: 99,842 样本
- **格式**: CSV文件 (question, choice_a, choice_b, choice_c, choice_d, answer)
- **任务**: 多选题问答

### 2. **GSM8K (Grade School Math 8K)**
- **位置**: `datasets/gsm8k/gsm8k.jsonl`
- **样本数**: 1,319 样本
- **格式**: JSONL (question, answer with reasoning)
- **任务**: 小学数学应用题

### 3. **HumanEval**
- **位置**: `datasets/humaneval/humaneval-py.jsonl`
- **样本数**: 161 样本
- **格式**: JSONL (name, prompt, test, entry_point)
- **任务**: Python代码生成

---

## ✅ **快速开始**

### **第一步: 验证数据集**

确认所有数据集已正确加载:

```bash
python verify_datasets.py
```

**预期输出**:
```
✅ GSM8K: 1,319 样本
✅ HumanEval: 161 样本
✅ MMLU: 18,738 test样本 (57个学科)
🎉 所有数据集验证通过！
```

---

### **第二步: 运行实验**

#### **选项1: 运行单个数据集**

```bash
# MMLU数据集
python run_experiment_with_local_data.py --domain mmlu --max_samples 100

# GSM8K数据集
python run_experiment_with_local_data.py --domain gsm8k --max_samples 50

# HumanEval数据集
python run_experiment_with_local_data.py --domain humaneval --max_samples 50
```

#### **选项2: 运行所有数据集**

```bash
python run_experiment_with_local_data.py --domain all --max_samples 100
```

#### **完整参数说明**

```bash
python run_experiment_with_local_data.py \
    --domain mmlu \                    # 数据集: mmlu/gsm8k/humaneval/all
    --dataset_root ./datasets \        # 数据集根目录
    --output_dir ./results/exp1 \      # 输出目录
    --max_samples 100 \                # 每个领域的最大样本数
    --split test \                     # 数据分割: train/test/val
    --batch_size 8 \                   # 批次大小
    --num_rounds 3 \                   # 智能体交互轮数
    --llm_name gpt-4o-mini             # 语言模型 (默认: gpt-4o-mini)
```

---

## 📊 **统一数据加载器**

### **Python API使用**

```python
from datasets.dataset_loader import DatasetLoader

# 初始化加载器
loader = DatasetLoader(dataset_root="./datasets")

# 加载MMLU数据
mmlu_samples = loader.load_mmlu(split='test', max_samples=100)

# 加载GSM8K数据
gsm8k_samples = loader.load_gsm8k(max_samples=50)

# 加载HumanEval数据
humaneval_samples = loader.load_humaneval(max_samples=50)

# 加载所有领域
all_data = loader.load_all_domains(max_samples_per_domain=50)
```

### **统一数据格式**

所有数据集都被转换为统一的 `DataSample` 格式:

```python
@dataclass
class DataSample:
    question: str          # 问题文本
    answer: str           # 标准答案
    domain: str           # 领域: 'mmlu'/'gsm8k'/'humaneval'
    metadata: Dict        # 额外信息 (学科、选项、推理过程等)
```

### **示例: MMLU样本**

```python
DataSample(
    question="What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid",
    answer="B",
    domain="mmlu",
    metadata={
        'subject': 'geography',
        'choices': ['London', 'Paris', 'Berlin', 'Madrid'],
        'split': 'test',
        'raw_question': 'What is the capital of France?'
    }
)
```

### **示例: GSM8K样本**

```python
DataSample(
    question="Janet's ducks lay 16 eggs per day...",
    answer="18",
    domain="gsm8k",
    metadata={
        'full_answer': "Janet sells 16 - 3 - 4 = 9 duck eggs...\n#### 18",
        'split': 'train'
    }
)
```

---

## 🎯 **领域提示集**

每个数据集都有专门优化的智能体角色配置:

### **MMLU领域**
- Knowledge Expert (知识专家)
- Subject Specialist (学科专家)
- Critical Analyzer (批判性分析师)
- Mathematician (数学家)
- Scientist (科学家)
- Humanities Scholar (人文学者)
- Social Scientist (社会科学家)

### **GSM8K领域**
- Math Problem Solver (数学问题解决者)
- Problem Analyzer (问题分析师)
- Calculation Verifier (计算验证师)
- Solution Critic (解决方案评审员)

### **HumanEval领域**
- Code Designer (代码设计师)
- Code Writer (代码编写者)
- Code Reviewer (代码审查员)
- Test Engineer (测试工程师)
- Algorithm Expert (算法专家)

---

## 🔬 **运行完整训练实验**

### **实验1: Baseline (无保护机制)**

```bash
python run_experiment_1_baseline.py \
    --mmlu_data_path ./datasets/mmlu \
    --output_dir ./results/experiment1_baseline \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001
```

### **实验2: With Protection (带保护机制)**

```bash
python run_experiment_2_protected.py \
    --mmlu_data_path ./datasets/mmlu \
    --output_dir ./results/experiment2_protected \
    --num_epochs 50 \
    --batch_size 16 \
    --protection_weight 0.3
```

### **鲁棒性测试 (攻击实验)**

```bash
python run_robustness_test.py \
    --attack_type frequency \         # frequency/semantic/mixed
    --attack_ratio 0.3 \               # 异常节点比例
    --output_dir ./results/robustness_test
```

---

## 📁 **输出结果**

实验结果会保存在指定的输出目录中:

```
results/
├── experiment1_baseline/
│   ├── mmlu_test_results.json      # 测试结果
│   ├── training_history.json       # 训练历史
│   └── model_checkpoints/          # 模型检查点
│
├── experiment2_protected/
│   ├── mmlu_test_results.json
│   ├── protection_analysis.json    # 保护机制分析
│   └── ...
│
└── local_data_experiments/
    ├── mmlu_test_results.json      # 各数据集的评估结果
    ├── gsm8k_test_results.json
    └── humaneval_test_results.json
```

### **结果文件格式**

```json
{
  "domain": "mmlu",
  "num_samples": 100,
  "correct_count": 72,
  "accuracy": 0.72,
  "results": [
    {
      "question": "...",
      "ground_truth": "B",
      "predicted_answer": "B",
      "is_correct": true,
      "reasoning_log": {...}
    },
    ...
  ]
}
```

---

## 🛠️ **自定义数据加载**

### **加载特定MMLU学科**

```python
# 只加载数学和物理相关学科
math_subjects = [
    'abstract_algebra', 'college_mathematics', 
    'high_school_mathematics', 'elementary_mathematics'
]

mmlu_math = loader.load_mmlu(
    split='test', 
    subjects=math_subjects, 
    max_samples=200
)
```

### **创建自定义批次**

```python
# 创建训练批次
batches = loader.create_training_batches(
    samples=gsm8k_samples,
    batch_size=16,
    shuffle=True
)

# 遍历批次
for batch in batches:
    # 处理每个批次
    for sample in batch:
        print(f"问题: {sample.question}")
        print(f"答案: {sample.answer}")
```

---

## 📈 **评估指标**

### **MMLU评估**
- **准确率**: 选项匹配 (A/B/C/D)
- **指标**: 整体准确率、各学科准确率

### **GSM8K评估**
- **准确率**: 数值答案匹配 (误差< 0.01)
- **指标**: 整体准确率、推理步骤正确性

### **HumanEval评估**
- **Pass@k**: 前k个生成中至少1个通过测试的比例
- **指标**: Pass@1, Pass@10, Pass@100

---

## 🚀 **性能优化建议**

### **1. 控制样本数量**
```bash
# 快速测试: 每个领域10个样本
python run_experiment_with_local_data.py --domain all --max_samples 10

# 完整评估: 每个领域1000个样本
python run_experiment_with_local_data.py --domain all --max_samples 1000
```

### **2. 调整批次大小**
- GPU充足: `--batch_size 32`
- GPU受限: `--batch_size 4`
- CPU运行: `--batch_size 1`

### **3. 减少交互轮数**
- 快速测试: `--num_rounds 1`
- 标准设置: `--num_rounds 3`
- 深度推理: `--num_rounds 5`

---

## ❓ **常见问题**

### **Q1: 数据集在哪里？**
A: 所有数据集都在 `datasets/` 目录下:
- `datasets/mmlu/data/`
- `datasets/gsm8k/gsm8k.jsonl`
- `datasets/humaneval/humaneval-py.jsonl`

### **Q2: 如何验证数据集完整性？**
A: 运行验证脚本:
```bash
python verify_datasets.py
```

### **Q3: 如何添加新数据集？**
A: 在 `datasets/dataset_loader.py` 中添加新的加载方法:
```python
def load_new_dataset(self, split='train', max_samples=None):
    # 实现加载逻辑
    pass
```

### **Q4: 结果保存在哪里？**
A: 默认保存在 `results/` 目录，可通过 `--output_dir` 参数自定义。

### **Q5: 支持哪些语言模型？**
A: 支持OpenAI API兼容的模型:
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- 或自定义API端点

---

## 📚 **相关文档**

- **数据集加载器**: `datasets/dataset_loader.py`
- **领域提示管理**: `neural_fsm_mas/domain_prompts/prompt_manager.py`
- **实验运行器**: `run_experiment_with_local_data.py`
- **数据集验证**: `verify_datasets.py`

---

## 🎓 **下一步**

1. ✅ **验证数据集**: `python verify_datasets.py`
2. ✅ **运行快速测试**: `python run_experiment_with_local_data.py --domain mmlu --max_samples 10`
3. ✅ **分析结果**: 查看 `results/` 目录中的JSON文件
4. ✅ **完整训练**: 运行 `run_experiment_1_baseline.py` 进行端到端训练

---

**最后更新**: 2025-11-06  
**版本**: 1.0  
**维护者**: NeuralFSM团队

