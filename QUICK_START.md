# NeuralFSM 快速开始指南
# Quick Start Guide

本指南将帮助你在5分钟内开始使用NeuralFSM项目。

---

## ✅ **前置条件检查**

### **1. Python环境**
确保已安装Python 3.9-3.11:
```bash
python --version
# 应该输出: Python 3.9.x 或 3.10.x 或 3.11.x
```

### **2. 必要的库**
安装项目依赖:
```bash
pip install -r requirements.txt
```

### **3. 数据集**
所有数据集已包含在项目中:
- ✅ MMLU: 18,738 测试样本 (57个学科)
- ✅ GSM8K: 1,319 样本
- ✅ HumanEval: 161 样本

---

## 🚀 **三步快速运行**

### **第一步: 验证数据集**

```bash
python verify_datasets.py
```

**预期输出**:
```
✅ GSM8K: 通过
✅ HumanEval: 通过
✅ MMLU: 通过
🎉 所有数据集验证通过！
```

---

### **第二步: 运行快速测试**

测试MMLU数据集 (10个样本):
```bash
python run_experiment_with_local_data.py --domain mmlu --max_samples 10
```

测试所有数据集 (每个10个样本):
```bash
python run_experiment_with_local_data.py --domain all --max_samples 10
```

**预期输出**:
```
🧪 运行实验: MMLU
📥 加载 mmlu 数据集 (test)...
✅ 加载 MMLU test: 10 样本
🎯 开始评估 10 个样本...
✅ 评估完成!
   准确率: 70.00% (7/10)
```

---

### **第三步: 查看结果**

结果保存在 `results/local_data_experiments/`:
```bash
# Windows
type results\local_data_experiments\mmlu_test_results.json

# Linux/Mac
cat results/local_data_experiments/mmlu_test_results.json
```

---

## 🎯 **常用命令**

### **运行特定数据集**

```bash
# MMLU (多选题)
python run_experiment_with_local_data.py --domain mmlu --max_samples 50

# GSM8K (数学题)
python run_experiment_with_local_data.py --domain gsm8k --max_samples 30

# HumanEval (代码生成)
python run_experiment_with_local_data.py --domain humaneval --max_samples 20
```

### **运行完整实验**

```bash
# 实验1: Baseline (无保护机制)
python run_experiment_1_baseline.py \
    --mmlu_data_path ./datasets/mmlu \
    --num_epochs 10 \
    --batch_size 8

# 实验2: Protected (带保护机制)
python run_experiment_2_protected.py \
    --mmlu_data_path ./datasets/mmlu \
    --num_epochs 10 \
    --protection_weight 0.3
```

### **鲁棒性测试**

```bash
# 频率攻击
python run_robustness_test.py --attack_type frequency --attack_ratio 0.3

# 语义攻击
python run_robustness_test.py --attack_type semantic --attack_ratio 0.3

# 混合攻击
python run_robustness_test.py --attack_type mixed --attack_ratio 0.3
```

---

## 📊 **数据集概览**

| 数据集 | 样本数 | 任务类型 | 评估指标 |
|--------|--------|----------|----------|
| **MMLU** | 18,738 | 多选题 (57个学科) | 准确率 |
| **GSM8K** | 1,319 | 小学数学应用题 | 数值匹配 |
| **HumanEval** | 161 | Python代码生成 | Pass@k |

---

## 🛠️ **常见问题**

### **Q1: ModuleNotFoundError**
**问题**: `ModuleNotFoundError: No module named 'torch'`

**解决**:
```bash
pip install -r requirements.txt
```

### **Q2: 数据集未找到**
**问题**: `FileNotFoundError: GSM8K数据集未找到`

**解决**:
```bash
# 检查数据集目录
ls datasets/gsm8k/
ls datasets/humaneval/
ls datasets/mmlu/data/

# 重新验证
python verify_datasets.py
```

### **Q3: CUDA/GPU问题**
**问题**: GPU不可用或CUDA错误

**解决**:
```bash
# CPU模式运行 (自动回退)
python run_experiment_with_local_data.py --domain mmlu --max_samples 10

# 检查PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### **Q4: OpenAI API密钥**
**问题**: OpenAI API调用失败

**解决**:
```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
set OPENAI_API_KEY=your-api-key-here       # Windows CMD
$env:OPENAI_API_KEY="your-api-key-here"    # Windows PowerShell
```

---

## 📁 **项目结构**

```
NeuralFSM/
├── datasets/                      # 数据集目录
│   ├── gsm8k/gsm8k.jsonl         # GSM8K数据
│   ├── humaneval/humaneval-py.jsonl  # HumanEval数据
│   ├── mmlu/data/                # MMLU数据
│   └── dataset_loader.py         # 统一数据加载器
│
├── neural_fsm_mas/                # 核心代码
│   ├── agent_topology/           # 智能体拓扑
│   ├── temporal_networks/        # TGN网络
│   ├── domain_prompts/           # 领域提示
│   ├── defense_mechanisms/       # 防御机制
│   └── train_neural_mas.py       # 训练器
│
├── run_experiment_with_local_data.py  # 本地数据实验
├── run_experiment_1_baseline.py       # 实验1 (无保护)
├── run_experiment_2_protected.py      # 实验2 (有保护)
├── run_robustness_test.py             # 鲁棒性测试
├── verify_datasets.py                 # 数据集验证
│
├── requirements.txt               # 依赖列表
└── DATASET_USAGE_GUIDE.md        # 详细使用指南
```

---

## 🎓 **下一步**

1. ✅ **阅读完整指南**: [DATASET_USAGE_GUIDE.md](./DATASET_USAGE_GUIDE.md)
2. ✅ **理解FSM状态转移**: [FSM_STATE_TRANSITION_EXPLAINED.md](./FSM_STATE_TRANSITION_EXPLAINED.md)
3. ✅ **查看实验1文档**: [EXPERIMENT1_ACADEMIC_PAPER.md](./EXPERIMENT1_ACADEMIC_PAPER.md)
4. ✅ **运行完整训练**: 
   ```bash
   python run_experiment_1_baseline.py --num_epochs 50
   ```

---

## 💡 **性能优化提示**

### **快速测试 (1-2分钟)**
```bash
python run_experiment_with_local_data.py \
    --domain mmlu \
    --max_samples 5 \
    --num_rounds 1
```

### **标准评估 (10-20分钟)**
```bash
python run_experiment_with_local_data.py \
    --domain mmlu \
    --max_samples 100 \
    --num_rounds 3
```

### **完整训练 (数小时)**
```bash
python run_experiment_1_baseline.py \
    --num_epochs 50 \
    --batch_size 16
```

---

## 📞 **获取帮助**

- **文档索引**: [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md)
- **数据集指南**: [DATASET_USAGE_GUIDE.md](./DATASET_USAGE_GUIDE.md)
- **安装指南**: [requirements.txt](./requirements.txt)
- **项目总结**: [FINAL_PROJECT_STATUS.md](./FINAL_PROJECT_STATUS.md)

---

**祝你使用愉快！ 🎉**

