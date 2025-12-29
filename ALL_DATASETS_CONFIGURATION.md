# 所有数据集配置说明
# All Datasets Configuration Guide

## 📊 支持的数据集

### 完整列表 (6个)

1. **GSM8K** - Grade School Math 数学题解答
2. **MMLU** - Massive Multitask Language Understanding (57个子类别)
3. **HumanEval** - Python代码生成
4. **HotpotQA** - 多跳问答推理 ✨ NEW
5. **ALFWorld** - 具身智能体交互任务 ✨ NEW
6. **MATH** - 高级数学竞赛题 ✨ NEW

---

## 🎯 统一实验脚本

**主实验脚本**: `run_experiment_1_fsm_complete.py`

### 运行所有数据集

```bash
# 运行单个数据集
python run_experiment_1_fsm_complete.py --domains gsm8k

# 运行多个数据集
python run_experiment_1_fsm_complete.py --domains gsm8k hotpotqa math

# 运行所有数据集
python run_experiment_1_fsm_complete.py --domains gsm8k mmlu humaneval hotpotqa alfworld math
```

---

## 📋 数据集特定配置

### 1. GSM8K

**数据路径**: `./datasets/gsm8k/gsm8k.jsonl`

**特点**:
- 数学应用题
- 多步推理
- 数值答案

**FSM配置**:
- 状态数: 7-10
- 智能体: Problem Analyzer, Math Solver, Calculation Verifier, Solution Critic
- FSM模式: 单FSM（所有问题共享）

**推荐参数**:
```bash
python run_experiment_1_fsm_complete.py \
  --domains gsm8k \
  --batch_size 16 \
  --num_epochs 50 \
  --max_total_steps 15
```

---

### 2. MMLU

**数据路径**: `./datasets/mmlu/data/`

**特点**:
- 57个学科类别
- 选择题（A/B/C/D）
- 知识问答

**FSM配置**:
- **特殊模式**: 类别级FSM（每个类别一套FSM）
- 状态数: 4-8（根据类别）
- 智能体: Knowledge Expert, Subject Specialist, Critical Analyzer, Decision Maker
- FSM模式: 57套FSM（每个类别1套）

**推荐参数**:
```bash
python run_experiment_1_fsm_complete.py \
  --domains mmlu \
  --mmlu_use_category_fsm \
  --batch_size 16 \
  --num_epochs 50 \
  --max_total_steps 10
```

**MMLU类别级FSM**:
- 启用: `--mmlu_use_category_fsm` (默认启用)
- 每个类别自动加载对应的FSM
- 如果FSM不存在，会在训练时动态生成

**生成所有MMLU类别FSM**:
```bash
# 使用专用脚本（待创建）
python generate_mmlu_category_fsms.py \
  --output_dir ./fsm_cache/mmlu \
  --use_azure False
```

---

### 3. HumanEval

**数据路径**: `./datasets/humaneval/humaneval-py.jsonl`

**特点**:
- Python代码生成
- 函数实现任务
- 需要测试验证

**FSM配置**:
- 状态数: 7-11
- 智能体: Code Designer, Code Writer, Code Reviewer, Test Engineer
- FSM模式: 单FSM（所有问题共享）

**推荐参数**:
```bash
python run_experiment_1_fsm_complete.py \
  --domains humaneval \
  --batch_size 8 \
  --num_epochs 50 \
  --max_total_steps 20
```

---

### 4. HotpotQA ✨ NEW

**数据路径**: `./datasets/hotpotqa/hotpotqa.jsonl`

**特点**:
- 多跳问答推理
- 跨文档信息整合
- 支撑事实标注
- 难度分级: easy/medium/hard

**FSM配置**:
- 状态数: 7-11
- 智能体: Question Analyzer, Document Retriever, Fact Extractor, Multi-hop Reasoner, Answer Synthesizer, Consistency Verifier
- FSM模式: 单FSM（所有问题共享）

**推荐参数**:
```bash
python run_experiment_1_fsm_complete.py \
  --domains hotpotqa \
  --batch_size 8 \
  --num_epochs 50 \
  --max_total_steps 20 \
  --max_visits_per_state 4
```

**特殊配置**:
- 长上下文处理: 自动截断到10个文档
- 支撑事实验证: 支持supporting_facts提取和验证
- 问题类型: bridge（桥接）或comparison（比较）

---

### 5. ALFWorld ✨ NEW

**数据路径**: `./datasets/alfworld/test.jsonl`

**特点**:
- 具身智能体交互
- 多步骤动作序列
- 环境反馈驱动
- 子目标正则匹配

**FSM配置**:
- 状态数: 6-11
- 智能体: Task Parser, Environment Explorer, Action Planner, Action Executor, State Monitor, Recovery Agent, Goal Verifier
- FSM模式: 单FSM（所有问题共享）

**推荐参数**:
```bash
python run_experiment_1_fsm_complete.py \
  --domains alfworld \
  --batch_size 4 \
  --num_epochs 50 \
  --max_total_steps 30 \
  --max_visits_per_state 5
```

**特殊配置**:
- 动作序列验证: 基于subgoals的正则表达式匹配
- 环境交互: 需要模拟环境反馈
- 任务类型: pick_and_place, pick_clean_then_place, pick_heat_then_place等

---

### 6. MATH ✨ NEW

**数据路径**: `./datasets/math/math.jsonl`

**特点**:
- 高级数学竞赛题
- 详细解答过程
- LaTeX格式答案
- 按subject分类: Algebra, Geometry, Number Theory等
- 难度分级: 1-5

**FSM配置**:
- 状态数: 8-12
- 智能体: Problem Analyzer, Math Concept Expert, Strategy Designer, Step-by-step Solver, Symbolic Calculator, Solution Verifier, Alternative Approach Agent, LaTeX Formatter
- FSM模式: 单FSM（所有问题共享）

**推荐参数**:
```bash
python run_experiment_1_fsm_complete.py \
  --domains math \
  --batch_size 8 \
  --num_epochs 50 \
  --max_total_steps 25 \
  --max_visits_per_state 4
```

**特殊配置**:
- LaTeX处理: 自动清理和格式化LaTeX公式
- 分层采样: 按subject分层划分训练/验证/测试集
- 答案提取: 从solution中提取boxed答案

---

## 🔧 通用配置参数

### FSM缓存配置

```bash
# 使用FSM缓存（默认启用）
--use_fsm_cache

# FSM缓存目录
--fsm_cache_dir ./fsm_cache

# 如果缓存不存在，自动生成FSM（默认启用）
--generate_fsm_if_missing

# MMLU使用类别级FSM（默认启用）
--mmlu_use_category_fsm
```

### 训练参数

```bash
# 训练轮数
--num_epochs 50

# 批次大小（根据数据集调整）
--batch_size 16  # GSM8K, MMLU
--batch_size 8   # HotpotQA, MATH, HumanEval
--batch_size 4   # ALFWorld

# 学习率
--learning_rate 0.001
```

### FSM执行参数

```bash
# 每个状态最大访问次数（循环避免）
--max_visits_per_state 3

# Episode最大总步数
--max_total_steps 20

# 自动修复FSM（默认启用）
--auto_fix_fsm

# 使用LLM生成缺失的转移条件
--use_llm_for_conditions
```

### 损失函数权重

```bash
# 策略梯度权重 (α)
--policy_gradient_weight 1.0

# 状态转移权重 (β)
--transition_loss_weight 0.3

# 监听路径权重 (γ)
--listener_loss_weight 0.2

# LLM成本权重 (δ)
--cost_loss_weight 0.1
```

---

## 📊 数据集对比表

| 数据集 | 类型 | 状态数 | 智能体数 | 批次大小 | 最大步数 | FSM模式 |
|--------|------|--------|----------|----------|----------|---------|
| GSM8K | 数学题 | 7-10 | 4 | 16 | 15 | 单FSM |
| MMLU | 选择题 | 4-8 | 4 | 16 | 10 | 类别FSM (57套) |
| HumanEval | 代码生成 | 7-11 | 4 | 8 | 20 | 单FSM |
| HotpotQA | 多跳QA | 7-11 | 6-7 | 8 | 20 | 单FSM |
| ALFWorld | 具身AI | 6-11 | 7 | 4 | 30 | 单FSM |
| MATH | 竞赛数学 | 8-12 | 8 | 8 | 25 | 单FSM |

---

## 🚀 快速开始示例

### 示例1: 运行GSM8K

```bash
python run_experiment_1_fsm_complete.py \
  --domains gsm8k \
  --num_epochs 50 \
  --batch_size 16 \
  --output_dir ./results/gsm8k_experiment
```

### 示例2: 运行HotpotQA（新数据集）

```bash
python run_experiment_1_fsm_complete.py \
  --domains hotpotqa \
  --num_epochs 50 \
  --batch_size 8 \
  --max_total_steps 20 \
  --generate_fsm_if_missing \
  --output_dir ./results/hotpotqa_experiment
```

### 示例3: 运行MATH（新数据集）

```bash
python run_experiment_1_fsm_complete.py \
  --domains math \
  --num_epochs 50 \
  --batch_size 8 \
  --max_total_steps 25 \
  --generate_fsm_if_missing \
  --output_dir ./results/math_experiment
```

### 示例4: 运行MMLU（类别级FSM）

```bash
python run_experiment_1_fsm_complete.py \
  --domains mmlu \
  --num_epochs 50 \
  --batch_size 16 \
  --mmlu_use_category_fsm \
  --generate_fsm_if_missing \
  --output_dir ./results/mmlu_experiment
```

### 示例5: 运行所有数据集

```bash
python run_experiment_1_fsm_complete.py \
  --domains gsm8k mmlu humaneval hotpotqa alfworld math \
  --num_epochs 50 \
  --batch_size 16 \
  --generate_fsm_if_missing \
  --mmlu_use_category_fsm \
  --output_dir ./results/all_datasets_experiment
```

---

## ⚠️ 注意事项

### 1. MMLU特殊处理

- **类别级FSM**: 每个类别需要独立的FSM
- **动态加载**: 训练时根据问题类别自动加载对应FSM
- **生成成本**: 57个类别 × ~$0.20 = ~$11.4 (GPT-4o-mini)
- **缓存管理**: FSM缓存存储在 `./fsm_cache/mmlu/{category_name}/`

### 2. 数据集路径

确保所有数据集文件在正确位置：
```
datasets/
├── gsm8k/
│   └── gsm8k.jsonl
├── mmlu/
│   └── data/
│       ├── test/
│       ├── val/
│       └── auxiliary_train/
├── humaneval/
│   └── humaneval-py.jsonl
├── hotpotqa/
│   └── hotpotqa.jsonl
├── alfworld/
│   └── test.jsonl
└── math/
    └── math.jsonl
```

### 3. FSM缓存

- **首次运行**: 会自动生成FSM（如果启用`--generate_fsm_if_missing`）
- **后续运行**: 自动使用缓存，节省时间和成本
- **缓存位置**: `./fsm_cache/{dataset_name}/`
- **缓存验证**: 自动验证缓存完整性

### 4. 内存和计算资源

- **ALFWorld**: 需要更多内存（长动作序列）
- **HotpotQA**: 需要处理长上下文（多文档）
- **MATH**: 需要处理LaTeX公式
- **MMLU**: 需要管理57套FSM

### 5. LLM成本

- **FSM生成**: 每个数据集约$0.20-0.50 (GPT-4o-mini)
- **MMLU**: 57个类别 × $0.20 = ~$11.4
- **训练成本**: 取决于episode数量和LLM调用次数

---

## 📚 相关文档

- `SIX_DATASETS_INTEGRATION_PLAN.md` - 完整集成方案
- `INTEGRATION_PROGRESS.md` - 进度跟踪
- `neural_fsm_mas/training_data/unified_data_processor.py` - 数据处理器
- `neural_fsm_mas/fsm_cache_manager.py` - FSM缓存管理
- `baseclass/Enhanced_FSM_Gen.py` - FSM生成器

---

**文档版本**: V1.0  
**最后更新**: 2024-11-13  
**作者**: Neural FSM-MAS Team

