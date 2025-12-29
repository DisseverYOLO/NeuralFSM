# 状态描述动态优化系统（修订版）
# State Description Dynamic Optimization System (Revised)

## 🎯 核心理念

**双维度优化：基于准确率和效率，动态增强状态描述**

### 关键洞察

在NeuralFSM中：
1. ❌ **无法单独追踪每个状态的成功率** - 只知道最终答案是否正确
2. ✅ **可以追踪状态在失败问题中的出现频率** - 准确率维度
3. ✅ **可以追踪达到最大转移次数的情况** - 效率维度

## 📊 工作原理

### ✨ 重要更新：正确追踪状态重复访问

**状态序列保留重复访问**：
- 使用 `List[str]` 而不是 `Set[str]`
- 如果某状态在一次执行中被访问多次，每次都计入统计
- 这样能更准确识别"反复访问的问题状态"

### 1. 准确率维度：追踪失败问题
```python
# 问题1（失败）访问的状态序列：[Problem Understanding, Calculation, Verification, Calculation]
#   → Calculation访问了2次（可能存在问题：反复计算）
# 问题2（失败）访问的状态序列：[Problem Understanding, Calculation]
# 问题3（成功）访问的状态序列：[Problem Understanding, Calculation, Verification]

# 统计结果（包括重复访问）：
state_total_visits = {
    'Problem Understanding': 3,  # 总共访问3次
    'Calculation': 4,            # 总共访问4次（问题1访问了2次！）
    'Verification': 2            # 总共访问2次
}

state_in_failures = {
    'Problem Understanding': 2,  # 在失败问题中出现2次
    'Calculation': 3,            # 在失败问题中出现3次（问题1贡献了2次）
    'Verification': 1            # 在失败问题中出现1次
}

# 失败率计算：
# Calculation: 3 / 4 = 75% ⚠️ 很高！
# Problem Understanding: 2 / 3 = 67%
# Verification: 1 / 2 = 50%
```

### 2. 效率维度：追踪最大转移次数
```python
# 达到最大转移次数的问题访问的状态序列
# 问题A（达到最大转移）：[State1, State2, State3, State2, State3, State2, ...]
#   → State2和State3反复出现，说明可能陷入循环

state_in_max_transitions = {
    'State2': 5,  # 在低效执行中出现5次（可能陷入循环）
    'State3': 4   # 在低效执行中出现4次
}
```

### 3. 触发优化的两种情况

#### 情况A：准确率问题（失败率）

**失败率计算公式**：
```
失败率 = 该状态在失败问题中出现的次数 / 该状态被访问的总次数

示例：
- Calculation状态总共被访问了 40 次
- 其中在 18 次失败的问题中出现过
- 失败率 = 18 / 40 = 45%
```

**触发条件**：
1. 该状态在失败问题中出现次数 >= 5次（默认）
2. **且** 失败率 > 30%（即该状态访问时，有30%以上的概率问题会失败）
3. → 添加**准确性增强** ⚠️

#### 情况B：效率问题（低效率）

**低效率计算公式**：
```
低效率 = 该状态在达到最大转移次数的问题中出现的次数 / 该状态被访问的总次数

示例：
- Solution Planning状态总共被访问了 40 次
- 其中在 10 次达到最大转移次数的问题中出现过
- 低效率 = 10 / 40 = 25%
```

**触发条件**：
1. 达到最大转移次数的问题总数 >= 3次（默认）
2. **且** 该状态在这些低效执行中出现次数 >= 3次
3. **且** 低效率 > 20%（即该状态访问时，有20%以上的概率会导致低效执行）
4. → 添加**效率增强** ⚡

#### 完整示例说明

假设有100个问题，某个状态 `Calculation` 的统计数据：

| 指标 | 数值 | 详细说明 |
|------|------|---------|
| 总访问次数 | 80 | 在100个问题中，80次问题访问了此状态 |
| | | |
| **准确率维度** | | |
| 失败问题总数 | 22 | 100个问题中，有22个最终答案错误 |
| 在失败中出现 | 18 | 在这22个失败问题中，18次访问了`Calculation`状态 |
| **失败率** | **18/80 = 22.5%** | **含义：访问此状态的问题中，22.5%会失败** |
| | | |
| **效率维度** | | |
| 达到最大转移总数 | 15 | 100个问题中，15次达到最大转移次数限制 |
| 在低效中出现 | 12 | 在这15次低效问题中，12次访问了`Calculation`状态 |
| **低效率** | **12/80 = 15%** | **含义：访问此状态的问题中，15%会低效执行** |

**触发条件判断**：
- ❌ 在失败中出现 18次 >= 5次 ✓，但失败率 22.5% < 30% → **不触发准确率增强**
- ❌ 达到最大转移总数 15次 >= 3次 ✓，在低效中出现 12次 >= 3次 ✓，但低效率 15% < 20% → **不触发效率增强**
- **结论：不需要优化此状态**

#### 触发优化的示例

假设 `Calculation` 状态的统计变为：

| 指标 | 数值 | 判断 |
|------|------|------|
| 总访问次数 | 40 | - |
| 在失败中出现 | 18 | >= 5次 ✓ |
| **失败率** | **18/40 = 45%** | **> 30% ✓** |
| 达到最大转移总数 | 15 | >= 3次 ✓ |
| 在低效中出现 | 10 | >= 3次 ✓ |
| **低效率** | **10/40 = 25%** | **> 20% ✓** |

**触发条件判断**：
- ✅ 失败率 45% > 30%，在失败中出现 18次 >= 5次 → **触发准确率增强** ⚠️
- ✅ 低效率 25% > 20%，达到最大转移 15次 >= 3次，在低效中出现 10次 >= 3次 → **触发效率增强** ⚡
- **结论：添加组合增强** 🔥

### 4. 动态增强示例

#### 准确率增强
```
⚠️ ACCURACY ALERT: This state appears in 7 failed attempts (45% failure rate).
CRITICAL: Double-check your work and verify all details carefully.

[原始状态描述]
```

#### 效率增强
```
⚡ EFFICIENCY ALERT: This state appears in 5 cases where maximum transitions 
were reached (35% of visits). Focus on being CONCISE and DIRECT - avoid unnecessary steps.

[原始状态描述]
```

#### 组合增强（同时存在准确率和效率问题）
```
⚠️ ACCURACY ALERT: This state appears in 8 failed attempts (50% failure rate).
CRITICAL: Double-check your work and verify all details carefully.
⚡ EFFICIENCY ALERT: This state appears in 4 cases where maximum transitions 
were reached. Focus on being CONCISE and DIRECT.

[原始状态描述]
```

## 🚀 使用方式

### 启用优化
```bash
python run_experiment_1_fsm_complete.py \
  --domains gsm8k \
  --llm_name gpt-4o-mini \
  --num_epochs 5 \
  --enable_prompt_optimization \
  --max_transitions_threshold 3 \
  --accuracy_enhancement_threshold 5 \
  --efficiency_enhancement_threshold 3
```

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_prompt_optimization` | False | 启用状态描述优化 |
| `--max_transitions_threshold` | 3 | 达到N次最大转移后触发效率优化 |
| `--accuracy_enhancement_threshold` | 5 | 失败N次后触发准确率优化 |
| `--efficiency_enhancement_threshold` | 3 | 低效执行N次后触发效率优化 |

## 📈 性能报告

### 实验结束时显示
```
✨ 状态描述优化摘要
================================================================================

📊 GSM8K:
  • 总问题数: 100
  • 整体准确率: 78.0%
  • 达到最大转移: 12次 (12.0%)
  • 被增强状态数: 3 / 8

  ⚠️  问题状态（按严重程度排序）:
    • Calculation:
        失败率: 45.0% (18次)
        低效率: 25.0% (10次)
        状态: 🔥 已增强(准确率+效率)
    
    • Solution Planning:
        失败率: 30.0% (12次)
        低效率: 35.0% (14次)
        状态: ⚡ 已增强(效率)
    
    • Verification:
        失败率: 35.0% (14次)
        低效率: 10.0% (4次)
        状态: ✅ 已增强(准确率)
```

## 💡 实际效果示例

### 场景：GSM8K数学问题

**初始情况（前30个问题）：**
- 总准确率：70%
- 达到最大转移次数：8次
- `Calculation` 状态在6次失败中出现
- `Solution Planning` 在5次低效执行中出现

**第30个问题后触发优化：**
```
  ⚠️  状态 'Calculation' 在 6 次失败中出现（失败率 40.0%）- 添加准确性增强
  ⚠️  状态 'Solution Planning' 在 5 次低效执行中出现 - 添加效率增强
```

**增强后的描述：**

**Calculation状态（准确率增强）：**
```
⚠️ NOTICE: This state has appeared in 6 failed attempts (40% failure rate).
Please focus on accuracy and avoid common mistakes.

Perform mathematical calculations step by step:
1. Extract numerical values from the solution plan
2. Execute calculations in the correct order
3. Verify intermediate results
4. Present the final numerical answer
```

**Solution Planning状态（效率增强）：**
```
⚡ EFFICIENCY ALERT: This state appears in 5 cases where maximum transitions 
were reached (31% of visits). Focus on being CONCISE and DIRECT - avoid unnecessary steps.

Develop a clear and efficient solution strategy:
1. Identify the problem type
2. Determine the required operations
3. Outline the calculation steps
4. Establish verification criteria
```

**后30个问题的结果：**
- 总准确率提升到：82%
- 达到最大转移次数降低到：4次
- 系统持续监控并可能调整增强策略

## 🎯 优势

| 特性 | 旧版（不准确） | 新版（准确） |
|------|--------------|------------|
| 追踪方式 | ❌ 假设可以追踪单个状态成功率 | ✅ 基于整体问题结果追踪状态 |
| 触发机制 | 单一准确率阈值 | ✅ 双维度（准确率+效率） |
| 效率考虑 | ❌ 无 | ✅ 追踪达到最大转移次数 |
| 问题诊断 | 准确率低的状态 | ✅ 失败频发 + 低效频发的状态 |
| 增强策略 | 单一提示 | ✅ 针对性提示（准确率/效率/组合） |

## 🔍 实现细节

### 核心逻辑
```python
# 1. 记录执行结果
optimizer.record_execution(
    visited_states={'Problem Understanding', 'Calculation', 'Verification'},
    is_correct=False,  # 答案错误 → 这些状态都记为"出现在失败中"
    reached_max_transitions=True  # 达到最大转移 → 这些状态都记为"出现在低效执行中"
)

# 2. 每10个问题评估一次
if total_questions % 10 == 0:
    update_enhancements()  # 检查是否需要添加/更新增强

# 3. 动态增强描述
enhanced_desc, was_enhanced = optimizer.get_enhanced_description(
    state_name='Calculation',
    original_description='Perform calculations...'
)

# 4. 更新状态描述
if was_enhanced:
    state.description = enhanced_desc
```

### 增强触发条件详解

#### 准确率增强触发条件
```python
if (state_in_failures[state_name] >= accuracy_enhancement_threshold and
    failure_rate > 0.3):
    add_accuracy_enhancement()
```

#### 效率增强触发条件
```python
if (max_transitions_count >= max_transitions_threshold and
    state_in_max_transitions[state_name] >= efficiency_enhancement_threshold and
    inefficiency_rate > 0.2):
    add_efficiency_enhancement()
```

## 🎯 关键修正

### ✅ 修正1：正确的追踪方式
- ❌ 旧：假设可以单独追踪每个状态的成功率
- ✅ 新：基于整个问题的结果，追踪状态在失败/低效执行中的出现频率

### ✅ 修正2：增加效率维度
- ❌ 旧：只考虑准确率
- ✅ 新：同时考虑准确率和效率（达到最大转移次数）

### ✅ 修正3：更实用的触发机制
- ❌ 旧：单一准确率阈值（如0.7）
- ✅ 新：基于实际出现次数和比率的双维度判断

## 总结

这个修订版系统：
- ✅ **更准确**：基于正确的追踪逻辑
- ✅ **更全面**：考虑准确率和效率两个维度
- ✅ **更实用**：基于实际出现次数触发，不依赖不存在的单状态成功率
- ✅ **更智能**：可以识别并组合多种问题类型

**完美适配NeuralFSM的实际架构！** 🚀
