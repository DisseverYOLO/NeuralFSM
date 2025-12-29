# 🧪 实验脚本使用指南
# Experiment Scripts Guide

---

## 📋 **脚本总览**

### **训练脚本分类**

```
训练脚本层次结构：

基础训练器（Training Core）
├── train_neural_mas.py          [协作式MAS训练器]
├── train_neural_mas_protected.py [协作式MAS + 保护]
└── train_fsm_mas.py              [MetaAgent自动生成FSM] (独立)

实验运行脚本（Experiment Runners）
├── run_experiment_1_baseline.py      [实验1A: 协作式MAS]
├── run_experiment_1_fsm.py           [实验1B: FSM模式] 🆕
├── run_experiment_2_protected.py     [实验2A: 协作式MAS + 保护]
└── run_experiment_2_fsm_protected.py [实验2B: FSM + 保护] 🆕
```

---

## 📚 **各脚本详细说明**

### **1. 基础训练器（不直接运行）**

#### **`train_neural_mas.py`** 

**功能**：协作式MAS基础训练器

**特点**：
- 所有智能体多轮并行交互
- 使用固定交互轮数 (`num_rounds=3`)
- 不使用FSM状态机
- 支持多数据集（MMLU、GSM8K、HumanEval）

**模式**：
```python
# 协作式MAS模式
await topology.execute_multi_agent_reasoning(
    task_input, 
    num_interaction_rounds=3  # 所有智能体交互3轮
)
```

**何时使用**：
- ✅ 作为基类被其他训练器继承
- ❌ 不建议直接运行（使用 `run_experiment_1_baseline.py` 代替）

---

#### **`train_neural_mas_protected.py`**

**功能**：协作式MAS + 保护机制

**特点**：
- 继承 `train_neural_mas.py`
- 添加 `ProtectedTGN` 包装
- 计算保护损失
- 记录学习到的保护权重

**保护机制**：
- 中心性计算（Betweenness + PageRank）
- 异常检测（频率 + 语义）
- 信任计算
- 消息权重衰减
- 保护损失

**何时使用**：
- ✅ 作为基类被 `run_experiment_2_protected.py` 调用
- ❌ 不建议直接运行

---

#### **`train_fsm_mas.py`**

**功能**：MetaAgent风格的FSM自动生成和训练

**特点**：
- 使用 `baseclass/FSM_Gen.py` 自动生成智能体
- 自动生成FSM状态和转移规则
- 随机采样拓扑图
- **独立的实验路径**

**适用场景**：
- 研究自动FSM生成
- 探索随机拓扑采样
- MetaAgent集成实验

**与新FSM模式的区别**：
| 特性 | train_fsm_mas.py | 新FSM模式 |
|-----|-----------------|----------|
| FSM来源 | 自动生成 | 手动定义/TGN学习 |
| 智能体 | 自动生成 | 预定义角色 |
| 拓扑 | 随机采样 | TGN学习 |
| 用途 | 自动化探索 | 精确控制 |

**何时使用**：
- ✅ 独立运行研究自动FSM生成
- ✅ 命令：`python train_fsm_mas.py --mode demo`
- ❌ 与新FSM模式**不冲突**，是互补关系

---

### **2. 实验运行脚本（直接运行）**

#### **`run_experiment_1_baseline.py`** ✅

**实验**: 实验1A - 协作式MAS（无保护）

**架构**：
- 协作式MAS：所有智能体多轮交互
- 无保护机制
- 作为Baseline

**运行命令**：
```bash
# 基本运行
python run_experiment_1_baseline.py \
    --dataset_root ./datasets \
    --domains mmlu gsm8k \
    --num_epochs 50

# 快速测试
python run_experiment_1_baseline.py \
    --domains mmlu \
    --num_epochs 5 \
    --batch_size 8
```

**输出**：
- `results/experiment1_baseline/experiment1_summary.json`
- 训练历史和模型权重

---

#### **`run_experiment_1_fsm.py`** 🆕 ✅

**实验**: 实验1B - FSM模式（无保护）

**架构**：
- FSM模式：每个状态一个智能体
- 状态转移：`<STATE_TRANS>: X`
- 监听关系：通信路径图
- 无保护机制

**运行命令**：
```bash
# 训练GSM8K
python run_experiment_1_fsm.py \
    --domain gsm8k \
    --num_states 4 \
    --num_epochs 10

# 训练所有领域
python run_experiment_1_fsm.py \
    --domain all \
    --num_states 4 \
    --num_epochs 20 \
    --batch_size 4
```

**输出**：
- `results/fsm_experiment1/final_evaluation.json`
- FSM结构和训练历史

---

#### **`run_experiment_2_protected.py`** ✅

**实验**: 实验2A - 协作式MAS + 保护机制

**架构**：
- 协作式MAS：所有智能体多轮交互
- ✅ 保护机制：ProtectedTGN
- 对比 `run_experiment_1_baseline.py`

**运行命令**：
```bash
# 基本运行
python run_experiment_2_protected.py \
    --mmlu_data_path ./datasets/mmlu \
    --num_epochs 50

# 自定义保护参数
python run_experiment_2_protected.py \
    --mmlu_data_path ./datasets/mmlu \
    --w_betweenness 0.7 \
    --w_pagerank 0.3 \
    --lambda_protect 0.15 \
    --num_epochs 30
```

**输出**：
- `results/experiment2_protected/experiment2_summary.json`
- 学习到的保护权重
- 与baseline的对比

---

#### **`run_experiment_2_fsm_protected.py`** 🆕 ✅

**实验**: 实验2B - FSM模式 + 保护机制

**架构**：
- FSM模式：状态机推理
- ✅ 保护机制：ProtectedTGN
- 对比 `run_experiment_1_fsm.py`

**运行命令**：
```bash
# 基本运行
python run_experiment_2_fsm_protected.py \
    --domain gsm8k \
    --num_states 4 \
    --num_epochs 10

# 完整保护参数
python run_experiment_2_fsm_protected.py \
    --domain all \
    --num_states 4 \
    --w_betweenness 0.6 \
    --w_pagerank 0.4 \
    --lambda_protect 0.1 \
    --num_epochs 20 \
    --baseline_results results/fsm_experiment1/final_evaluation.json
```

**输出**：
- `results/fsm_experiment2_protected/final_evaluation.json`
- FSM结构 + 保护权重
- 与FSM baseline的对比

---

## 🧪 **完整实验矩阵**

### **2×2 实验设计**

|  | **协作式MAS** | **FSM模式** |
|---|-------------|-----------|
| **无保护（Baseline）** | 实验1A: `run_experiment_1_baseline.py` | 实验1B: `run_experiment_1_fsm.py` 🆕 |
| **带保护** | 实验2A: `run_experiment_2_protected.py` | 实验2B: `run_experiment_2_fsm_protected.py` 🆕 |

### **实验目的**

1. **实验1A vs 实验1B**：对比**协作式MAS vs FSM**（无保护）
   - 哪种架构更有效？
   - FSM的可解释性优势

2. **实验1A vs 实验2A**：验证**保护机制**对协作式MAS的效果
   - 保护机制能提升多少？

3. **实验1B vs 实验2B**：验证**保护机制**对FSM的效果
   - FSM + 保护的协同效果

4. **实验2A vs 实验2B**：对比**两种架构+保护**
   - 最优组合是什么？

---

## 🚀 **快速开始**

### **完整实验流程**

```bash
# 1. 实验1A: 协作式MAS Baseline
python run_experiment_1_baseline.py \
    --domains mmlu gsm8k \
    --num_epochs 50 \
    --output_dir results/exp1a_baseline

# 2. 实验1B: FSM Baseline 🆕
python run_experiment_1_fsm.py \
    --domain all \
    --num_states 4 \
    --num_epochs 50 \
    --output_dir results/exp1b_fsm

# 3. 实验2A: 协作式MAS + 保护
python run_experiment_2_protected.py \
    --mmlu_data_path ./datasets/mmlu \
    --num_epochs 50 \
    --baseline_results results/exp1a_baseline/experiment1_summary.json \
    --output_dir results/exp2a_protected

# 4. 实验2B: FSM + 保护 🆕
python run_experiment_2_fsm_protected.py \
    --domain all \
    --num_states 4 \
    --num_epochs 50 \
    --baseline_results results/exp1b_fsm/final_evaluation.json \
    --output_dir results/exp2b_fsm_protected
```

### **快速验证（测试模式）**

```bash
# 每个实验只运行5个epoch，快速验证
python run_experiment_1_baseline.py --domains mmlu --num_epochs 5
python run_experiment_1_fsm.py --domain gsm8k --num_epochs 5
python run_experiment_2_protected.py --mmlu_data_path ./datasets/mmlu --num_epochs 5
python run_experiment_2_fsm_protected.py --domain gsm8k --num_epochs 5
```

---

## 📊 **结果对比**

### **结果文件位置**

| 实验 | 结果文件 |
|-----|---------|
| 实验1A | `results/exp1a_baseline/experiment1_summary.json` |
| 实验1B | `results/exp1b_fsm/final_evaluation.json` |
| 实验2A | `results/exp2a_protected/experiment2_summary.json` |
| 实验2B | `results/exp2b_fsm_protected/final_evaluation.json` |

### **对比分析**

```python
# 加载结果
import json

with open('results/exp1a_baseline/experiment1_summary.json') as f:
    exp1a = json.load(f)

with open('results/exp1b_fsm/final_evaluation.json') as f:
    exp1b = json.load(f)

with open('results/exp2a_protected/experiment2_summary.json') as f:
    exp2a = json.load(f)

with open('results/exp2b_fsm_protected/final_evaluation.json') as f:
    exp2b = json.load(f)

# 对比准确率
print(f"实验1A (协作式MAS): {exp1a['average_accuracy']:.4f}")
print(f"实验1B (FSM): {exp1b['average_accuracy']:.4f}")
print(f"实验2A (协作式MAS+保护): {exp2a['average_accuracy']:.4f}")
print(f"实验2B (FSM+保护): {exp2b['average_accuracy']:.4f}")
```

---

## ❓ **常见问题**

### **Q1: 应该运行哪些脚本？**

**A**: 运行4个实验脚本（`run_experiment_*.py`），**不需要**直接运行 `train_*.py`

### **Q2: `train_fsm_mas.py` 和 `run_experiment_1_fsm.py` 的区别？**

**A**: 
- `train_fsm_mas.py`: MetaAgent风格，**自动生成**FSM
- `run_experiment_1_fsm.py`: 新FSM模式，**手动定义**FSM，TGN学习

两者是**互补关系**，不冲突。

### **Q3: `train_neural_mas.py` 需要修改吗？**

**A**: **不需要**。它已经兼容FSM模式：
- `use_fsm_mode=False` → 协作式MAS
- `use_fsm_mode=True` → FSM模式

### **Q4: 保护机制兼容FSM模式吗？**

**A**: **兼容**。通过 `run_experiment_2_fsm_protected.py` 使用。

### **Q5: 如何选择实验？**

**A**: 根据研究目标：

| 研究目标 | 运行的实验 |
|---------|----------|
| 对比架构 | 实验1A + 实验1B |
| 验证保护机制 | 实验1A + 实验2A |
| 最优组合 | 全部4个实验 |
| FSM+保护效果 | 实验1B + 实验2B |

---

## 🎯 **推荐工作流**

### **学术论文实验流程**

1. **基础实验**（建立Baseline）
   ```bash
   python run_experiment_1_baseline.py --domains mmlu gsm8k --num_epochs 50
   python run_experiment_1_fsm.py --domain all --num_epochs 50
   ```

2. **保护机制实验**（验证有效性）
   ```bash
   python run_experiment_2_protected.py --num_epochs 50
   python run_experiment_2_fsm_protected.py --domain all --num_epochs 50
   ```

3. **结果分析**
   - 对比4组实验结果
   - 生成表格和图表
   - 分析保护机制的效果

4. **消融实验**（可选）
   - 调整保护参数
   - 调整FSM状态数
   - 测试不同数据集

---

## 📝 **总结**

### **运行实验的正确方式**

✅ **推荐**：
- `run_experiment_1_baseline.py` - 协作式MAS
- `run_experiment_1_fsm.py` - FSM模式 🆕
- `run_experiment_2_protected.py` - 协作式MAS + 保护
- `run_experiment_2_fsm_protected.py` - FSM + 保护 🆕

❌ **不推荐直接运行**：
- `train_neural_mas.py` - 基类，被其他脚本调用
- `train_neural_mas_protected.py` - 基类，被其他脚本调用

✅ **可独立运行**（不同研究方向）：
- `train_fsm_mas.py` - MetaAgent自动FSM生成

---

**完整实验矩阵**: 4个实验脚本覆盖所有组合 ✅

**兼容性**: 所有模式相互兼容，可自由切换 ✅

**易用性**: 清晰的命令行参数，简单易用 ✅

---

**🎉 现在你有了完整的实验工具包！** 🚀

