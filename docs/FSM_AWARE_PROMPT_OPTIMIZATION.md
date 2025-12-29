# FSM感知的提示词优化系统
# FSM-Aware Prompt Optimization System

## 🎯 核心理念

**不是简单的模板替换，而是深度结合FSM架构的智能提示词生成系统**

传统提示词优化：
```
模板A: "Please solve this problem step by step"
模板B: "Let's think step by step"
→ 简单选择性能最好的模板
```

FSM感知提示词优化：
```
[Your Role: Problem Analyzer]
Analyze carefully and identify key information:

[Current State: Problem Understanding]
State Goal: Parse the problem statement and identify key variables...

[Collaborating with: Solution Planner, Verifier]
Your output will be reviewed by these agents.

[Original Question]
...
```

## 🏗️ 系统架构

### 1. FSM结构注册
在创建FSM系统后，自动注册FSM结构信息：

```python
# 注册时提取：
- 智能体角色（agent_role）
- 智能体描述（agent_description）
- 状态名称（state_name）
- 状态描述（description）

optimizer.register_fsm_structure(fsm_config, agents)
```

### 2. 角色策略分类
根据智能体角色自动分类为5种类型：

| 角色类型 | 关键词 | 提示词策略 | 示例角色 |
|---------|--------|-----------|----------|
| **Analyzer** | analyz, examin, understand | 强调分析和理解 | Problem Analyzer, Data Examiner |
| **Planner** | plan, design, strateg | 强调步骤和策略 | Solution Planner, Strategy Designer |
| **Executor** | solv, calculat, execut | 强调准确和效率 | Problem Solver, Calculator |
| **Verifier** | verif, check, validat | 强调检查和纠错 | Solution Verifier, Answer Checker |
| **Integrator** | integrat, synthes, summariz | 强调整合和总结 | Result Integrator, Final Synthesizer |

### 3. 动态提示词生成
运行时根据多维上下文信息生成提示词：

```python
optimized_prompt, metadata = optimizer.get_prompt(
    question=问题,
    agent_role=当前智能体角色,
    state_name=当前状态名称,
    state_description=当前状态描述,
    listeners=当前监听者列表,
    transition_history=状态转移历史
)
```

**生成的提示词包含：**
1. **角色指导**：基于智能体类型的特定指导
2. **状态上下文**：当前状态的目标和描述
3. **协作信息**：其他智能体的监听关系
4. **历史警告**：对频繁失败的转移给予特别提醒
5. **原始问题**：用户的实际问题
6. **结尾指导**：基于角色的输出要求

### 4. 性能追踪与优化

#### 追踪维度：
- **状态-角色性能**：每个(状态, 角色)组合的准确率
- **失败转移记录**：记录频繁失败的状态转移
- **角色总体表现**：每个角色的整体准确率

#### 自动优化：
- 识别表现最差的状态-角色组合
- 对失败转移>=2次的路径发出警告
- 在提示词中注入历史失败信息

## 📊 示例场景

### 场景1：GSM8K数学问题

**FSM状态序列：**
```
State_0 (Problem Understanding) → 
State_1 (Solution Planning) → 
State_2 (Calculation) → 
State_3 (Verification) → 
State_4 (Final Answer)
```

**State_0的提示词（Problem Analyzer）：**
```
[Your Role: Problem Analyzer]
Analyze carefully and identify key information:

[Current State: Problem Understanding]
State Goal: Parse the problem statement and extract numerical values, 
variables, and mathematical relationships...

[Collaborating with: Solution Planner]
Your output will be reviewed by these agents.

John has 5 apples. He gives 2 to Mary. How many does he have left?

Provide your analysis/solution clearly.
```

**State_2的提示词（Calculator）：**
```
[Your Role: Problem Calculator]
Execute the solution accurately:

[Current State: Calculation]
State Goal: Perform mathematical calculations according to the solution plan...

[Collaborating with: Solution Verifier]
Your output will be reviewed by these agents.

[Note: Be extra careful with this step - it has failed 2 times before]

计算：5 - 2 = ?

Provide your analysis/solution clearly.
```

### 场景2：性能报告

```json
{
  "sample_count": 150,
  "role_performance": {
    "Problem Analyzer": {"accuracy": 0.92, "total": 50},
    "Solution Planner": {"accuracy": 0.85, "total": 50},
    "Problem Calculator": {"accuracy": 0.78, "total": 50},
    "Solution Verifier": {"accuracy": 0.95, "total": 50}
  },
  "worst_performers": [
    {
      "state": "Calculation",
      "role": "Problem Calculator",
      "accuracy": 0.78,
      "total": 50
    }
  ],
  "most_failed_transitions": [
    [("Solution Planning", "Calculation"), 8],
    [("Calculation", "Verification"), 5]
  ]
}
```

## 🚀 使用方式

### 启用FSM感知提示词优化

```bash
python run_experiment_1_fsm_complete.py \
  --domains gsm8k \
  --llm_name gpt-4o-mini \
  --num_epochs 5 \
  --enable_prompt_optimization \
  --enable_role_based \
  --enable_state_context \
  --enable_collaboration \
  --enable_history_aware
```

### 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_role_based` | True | 启用基于角色的提示词定制 |
| `enable_state_context` | True | 启用状态上下文注入 |
| `enable_collaboration` | True | 启用协作信息注入 |
| `enable_history_aware` | True | 启用历史感知优化 |
| `switch_interval` | 30 | 性能评估间隔（样本数） |

## 🎯 优势对比

| 特性 | 传统模板切换 | FSM感知优化 |
|------|-------------|-------------|
| 提示词来源 | 预定义模板 | 动态生成 |
| 上下文感知 | ❌ | ✅ FSM状态、角色、监听者 |
| 角色定制 | ❌ | ✅ 5种角色策略 |
| 历史学习 | ❌ | ✅ 失败转移警告 |
| 协作提示 | ❌ | ✅ 监听者信息 |
| NeuralFSM适配 | ❌ 通用 | ✅ 专门设计 |

## 📈 预期效果

1. **更精准的角色指导**：不同角色的智能体获得符合其职责的提示词
2. **更丰富的上下文**：利用FSM结构信息提供更多背景
3. **更智能的优化**：基于历史表现动态调整提示词
4. **更好的协作**：明确告知智能体之间的监听关系

## 🔍 实现细节

### 关键代码位置

1. **提示词优化器**：`neural_fsm_mas/prompt_optimization/__init__.py`
2. **FSM结构注册**：`train_fsm_mas_v2.py::create_domain_fsm_system()`
3. **提示词生成**：训练循环中的 `optimizer.get_prompt()`
4. **结果记录**：训练循环中的 `optimizer.record_result()`

### 扩展方向

1. **Few-shot示例选择**：根据当前状态和角色选择相似的成功案例
2. **动态模板生成**：使用LLM生成针对性的提示词模板
3. **多Agent协同优化**：考虑整个FSM路径的提示词协同优化
4. **强化学习集成**：将提示词优化作为策略空间的一部分

## 总结

FSM感知的提示词优化系统**充分利用了NeuralFSM的架构特点**，不是简单的模板替换，而是**深度集成**了：
- ✅ FSM的状态结构
- ✅ 智能体的角色分工
- ✅ 运行时的上下文信息
- ✅ 历史执行的经验

这才是真正适合NeuralFSM项目的提示词优化方案！

