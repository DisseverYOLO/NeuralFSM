# NeuralFSM 消融实验设计文档
# NeuralFSM Ablation Study Design Documentation

**版本**: V2.0  
**日期**: 2024-11-13  
**状态**: 设计阶段

---

## 📋 目录

1. [实验概述](#1-实验概述)
2. [核心组件消融实验](#2-核心组件消融实验)
3. [防御保护消融实验](#3-防御保护消融实验)
4. [损失函数消融实验](#4-损失函数消融实验)

---

## 1. 实验概述

### 1.1 实验目标

消融实验的目标是系统地验证NeuralFSM系统中各个组件的贡献，量化每个组件对系统性能的影响。通过逐步移除或替换各个组件，我们能够：

1. **验证核心组件的有效性**: 证明TGN、FSM、提示词优化等核心组件的必要性
2. **量化各组件的贡献**: 测量每个组件对性能提升的具体贡献
3. **识别关键设计决策**: 发现哪些设计决策对系统性能影响最大

### 1.2 实验配置

**数据集**: GSM8K、MATH、HumanEval  
**评估指标**: 任务准确率（Accuracy）  
**实验设置**: 每个配置运行3次，报告平均值和标准差  
**基线对比**: 完整系统（Full System）作为对比基线

---

## 2. 核心组件消融实验

本实验验证TGN、FSM和提示词优化三个核心组件对系统性能的贡献。

### 2.1 实验配置说明

- **完整系统（Full System）**: 使用TGN + FSM + 提示词优化
- **TGN消融**: 移除TGN，使用固定规则进行状态转移和监听关系
- **FSM消融**: 移除FSM结构，使用简单的多轮对话模式
- **提示词优化消融**: 禁用提示词优化，使用固定提示词

### 2.2 核心组件消融实验表

| 实验ID | 实验名称 | 关键配置 | GSM8K准确率 (%) | MATH准确率 (%) | HumanEval准确率 (%) | 平均准确率 (%) |
|--------|---------|---------|----------------|---------------|-------------------|--------------|
| **Full** | **完整系统** | `use_tgn=True, use_fsm=True, enable_prompt_optimization=True` | **-** | **-** | **-** | **-** |
| A1 | TGN消融 | `use_tgn=False, transition_strategy='sequential', listener_strategy='all_to_all'` | - | - | - | - |
| A2 | FSM消融 | `use_fsm=False, execution_mode='dialogue'` | - | - | - | - |
| A3 | 提示词优化消融 | `enable_prompt_optimization=False, prompt_type='fixed'` | - | - | - | - |
| A4 | TGN + FSM消融 | `use_tgn=False, use_fsm=False` | - | - | - | - |
| A5 | TGN + 提示词优化消融 | `use_tgn=False, enable_prompt_optimization=False` | - | - | - | - |
| A6 | FSM + 提示词优化消融 | `use_fsm=False, enable_prompt_optimization=False` | - | - | - | - |
| A7 | 全部消融（基线） | `use_tgn=False, use_fsm=False, enable_prompt_optimization=False` | - | - | - | - |

**说明**:
- 表格中的准确率将在实验执行后填入
- 格式: `平均值 ± 标准差`（例如：`85.3 ± 1.2`）
- 完整系统（Full）作为对比基线，其准确率将作为100%基准
- 其他实验的准确率将与完整系统对比，计算相对性能下降

**预期结果**:
- TGN消融应该导致准确率显著下降（预期下降10-20%）
- FSM消融应该导致准确率显著下降（预期下降5-15%）
- 提示词优化消融应该导致准确率下降（预期下降2-5%）
- 多个组件同时消融应该导致更大的性能下降

---

## 3. 防御保护消融实验

本实验验证保护机制中各个组件的贡献，包括异常检测、中心性分析和消息权重计算。实验在有攻击和无攻击两种场景下进行。

### 3.1 实验配置说明

- **完整保护系统（Full Protection）**: 使用异常检测 + 中心性分析 + 可学习MLP消息权重
- **异常检测消融**: 移除异常检测机制（`lambda_freq=0, lambda_semantic=0`）
- **中心性分析消融**: 移除中心性分析（`w_betweenness=0, w_pagerank=0`）
- **消息权重消融**: 使用固定公式替代可学习MLP（`learnable_weights=False`）
- **无保护系统**: 完全移除保护机制（`use_protection=False`）

**攻击场景**:
- **频率攻击**: 某些智能体发送大量消息（攻击强度：20%智能体，消息频率增加5倍）
- **语义攻击**: 某些智能体发送语义异常的消息（攻击强度：20%智能体，语义相似度<0.3）
- **恶意攻击**: 某些智能体故意发送错误信息（攻击强度：20%智能体，故意输出错误答案）

### 3.2 防御保护消融实验表（无攻击场景）

| 实验ID | 实验名称 | 关键配置 | GSM8K准确率 (%) | MATH准确率 (%) | HumanEval准确率 (%) | 平均准确率 (%) |
|--------|---------|---------|----------------|---------------|-------------------|--------------|
| **Full-Protect** | **完整保护系统** | `use_protection=True, lambda_freq=0.3, lambda_semantic=0.7, w_betweenness=0.6, w_pagerank=0.4, learnable_weights=True` | **-** | **-** | **-** | **-** |
| B1 | 异常检测消融 | `lambda_freq=0, lambda_semantic=0` | - | - | - | - |
| B2 | 中心性分析消融 | `w_betweenness=0, w_pagerank=0` | - | - | - | - |
| B3 | 消息权重消融（固定公式） | `learnable_weights=False` | - | - | - | - |
| B4 | 异常检测 + 中心性消融 | `lambda_freq=0, lambda_semantic=0, w_betweenness=0, w_pagerank=0` | - | - | - | - |
| B5 | 异常检测 + 消息权重消融 | `lambda_freq=0, lambda_semantic=0, learnable_weights=False` | - | - | - | - |
| B6 | 中心性 + 消息权重消融 | `w_betweenness=0, w_pagerank=0, learnable_weights=False` | - | - | - | - |
| B7 | 无保护系统 | `use_protection=False` | - | - | - | - |

### 3.3 防御保护消融实验表（有攻击场景）

| 实验ID | 实验名称 | 关键配置 | 攻击类型 | GSM8K准确率 (%) | MATH准确率 (%) | HumanEval准确率 (%) | 平均准确率 (%) |
|--------|---------|---------|---------|----------------|---------------|-------------------|--------------|
| **Full-Protect** | **完整保护系统** | `use_protection=True, lambda_freq=0.3, lambda_semantic=0.7, w_betweenness=0.6, w_pagerank=0.4, learnable_weights=True` | **频率攻击** | **-** | **-** | **-** | **-** |
| **Full-Protect** | **完整保护系统** | `use_protection=True, lambda_freq=0.3, lambda_semantic=0.7, w_betweenness=0.6, w_pagerank=0.4, learnable_weights=True` | **语义攻击** | **-** | **-** | **-** | **-** |
| **Full-Protect** | **完整保护系统** | `use_protection=True, lambda_freq=0.3, lambda_semantic=0.7, w_betweenness=0.6, w_pagerank=0.4, learnable_weights=True` | **恶意攻击** | **-** | **-** | **-** | **-** |
| B1 | 异常检测消融 | `lambda_freq=0, lambda_semantic=0` | 频率攻击 | - | - | - | - |
| B1 | 异常检测消融 | `lambda_freq=0, lambda_semantic=0` | 语义攻击 | - | - | - | - |
| B1 | 异常检测消融 | `lambda_freq=0, lambda_semantic=0` | 恶意攻击 | - | - | - | - |
| B2 | 中心性分析消融 | `w_betweenness=0, w_pagerank=0` | 频率攻击 | - | - | - | - |
| B2 | 中心性分析消融 | `w_betweenness=0, w_pagerank=0` | 语义攻击 | - | - | - | - |
| B2 | 中心性分析消融 | `w_betweenness=0, w_pagerank=0` | 恶意攻击 | - | - | - | - |
| B3 | 消息权重消融（固定公式） | `learnable_weights=False` | 频率攻击 | - | - | - | - |
| B3 | 消息权重消融（固定公式） | `learnable_weights=False` | 语义攻击 | - | - | - | - |
| B3 | 消息权重消融（固定公式） | `learnable_weights=False` | 恶意攻击 | - | - | - | - |
| B4 | 异常检测 + 中心性消融 | `lambda_freq=0, lambda_semantic=0, w_betweenness=0, w_pagerank=0` | 频率攻击 | - | - | - | - |
| B4 | 异常检测 + 中心性消融 | `lambda_freq=0, lambda_semantic=0, w_betweenness=0, w_pagerank=0` | 语义攻击 | - | - | - | - |
| B4 | 异常检测 + 中心性消融 | `lambda_freq=0, lambda_semantic=0, w_betweenness=0, w_pagerank=0` | 恶意攻击 | - | - | - | - |
| B5 | 异常检测 + 消息权重消融 | `lambda_freq=0, lambda_semantic=0, learnable_weights=False` | 频率攻击 | - | - | - | - |
| B5 | 异常检测 + 消息权重消融 | `lambda_freq=0, lambda_semantic=0, learnable_weights=False` | 语义攻击 | - | - | - | - |
| B5 | 异常检测 + 消息权重消融 | `lambda_freq=0, lambda_semantic=0, learnable_weights=False` | 恶意攻击 | - | - | - | - |
| B6 | 中心性 + 消息权重消融 | `w_betweenness=0, w_pagerank=0, learnable_weights=False` | 频率攻击 | - | - | - | - |
| B6 | 中心性 + 消息权重消融 | `w_betweenness=0, w_pagerank=0, learnable_weights=False` | 语义攻击 | - | - | - | - |
| B6 | 中心性 + 消息权重消融 | `w_betweenness=0, w_pagerank=0, learnable_weights=False` | 恶意攻击 | - | - | - | - |
| B7 | 无保护系统 | `use_protection=False` | 频率攻击 | - | - | - | - |
| B7 | 无保护系统 | `use_protection=False` | 语义攻击 | - | - | - | - |
| B7 | 无保护系统 | `use_protection=False` | 恶意攻击 | - | - | - | - |

**说明**:
- 无攻击场景下，保护机制版本应该与无保护版本性能相近（预期差异<2%）
- 有攻击场景下，保护机制版本应该显著优于无保护版本（预期提升20-40%）
- 可学习MLP应该优于固定公式，特别是在复杂攻击场景下（预期提升5-10%）
- 异常检测和中心性分析应该互补，结合使用效果最好

**预期结果**:
- 无攻击场景：所有配置的性能应该相近，保护机制不会显著影响正常性能
- 有攻击场景：完整保护系统应该显著优于消融版本，无保护系统性能应该最低
- 可学习MLP在复杂攻击场景下应该优于固定公式

---

## 4. 损失函数消融实验

本实验验证多目标损失函数中各个损失项的贡献，包括策略损失、状态转移损失、监听路径损失、LLM成本损失和保护损失。

### 4.1 实验配置说明

- **完整损失函数（Full Loss）**: 使用所有损失项（`policy_gradient_weight=1.0, transition_loss_weight=0.3, listener_loss_weight=0.2, cost_loss_weight=0.1, lambda_protect=0.1`）
- **策略损失消融**: 移除策略损失（`policy_gradient_weight=0`）
- **状态转移损失消融**: 移除状态转移损失（`transition_loss_weight=0`）
- **监听路径损失消融**: 移除监听路径损失（`listener_loss_weight=0`）
- **LLM成本损失消融**: 移除LLM成本损失（`cost_loss_weight=0`）
- **保护损失消融**: 移除保护损失（`lambda_protect=0`）

### 4.2 损失函数消融实验表

| 实验ID | 实验名称 | 关键配置 | GSM8K准确率 (%) | MATH准确率 (%) | HumanEval准确率 (%) | 平均准确率 (%) |
|--------|---------|---------|----------------|---------------|-------------------|--------------|
| **Full-Loss** | **完整损失函数** | `policy_gradient_weight=1.0, transition_loss_weight=0.3, listener_loss_weight=0.2, cost_loss_weight=0.1, lambda_protect=0.1` | **-** | **-** | **-** | **-** |
| C1 | 策略损失消融 | `policy_gradient_weight=0` | - | - | - | - |
| C2 | 状态转移损失消融 | `transition_loss_weight=0` | - | - | - | - |
| C3 | 监听路径损失消融 | `listener_loss_weight=0` | - | - | - | - |
| C4 | LLM成本损失消融 | `cost_loss_weight=0` | - | - | - | - |
| C5 | 保护损失消融 | `lambda_protect=0` | - | - | - | - |
| C6 | 策略 + 转移损失消融 | `policy_gradient_weight=0, transition_loss_weight=0` | - | - | - | - |
| C7 | 策略 + 监听损失消融 | `policy_gradient_weight=0, listener_loss_weight=0` | - | - | - | - |
| C8 | 策略 + 成本损失消融 | `policy_gradient_weight=0, cost_loss_weight=0` | - | - | - | - |
| C9 | 转移 + 监听损失消融 | `transition_loss_weight=0, listener_loss_weight=0` | - | - | - | - |
| C10 | 全部辅助损失消融 | `transition_loss_weight=0, listener_loss_weight=0, cost_loss_weight=0, lambda_protect=0` | - | - | - | - |

**说明**:
- 策略损失是任务导向的主要损失，移除它应该导致准确率显著下降（预期下降15-25%）
- 状态转移损失和监听路径损失是辅助损失，移除它们应该导致相应指标下降，但任务准确率影响较小（预期下降2-5%）
- LLM成本损失控制成本，移除它应该导致成本增加，但任务准确率可能提升（预期提升1-3%）
- 保护损失在正常情况下影响较小，但在攻击场景下影响较大

**预期结果**:
- 策略损失消融应该导致最大的性能下降
- 辅助损失（转移、监听、成本）消融应该导致较小的性能下降
- 多个损失项同时消融应该导致更大的性能下降
- 完整损失函数应该实现最佳的平衡性能

---

## 5. 实验执行说明

### 5.1 实验配置

**数据集**: 
- GSM8K（数学推理）
- MATH（竞赛数学）
- HumanEval（代码生成）

**训练配置**:
- 训练轮数: 50 epochs
- 批次大小: 16
- 学习率: 0.001
- 随机种子: 42（固定，确保可复现性）

**运行次数**: 每个实验配置运行3次，报告平均值和标准差

**评估指标**: 任务准确率（Accuracy）

### 5.2 结果报告格式

- **准确率格式**: `平均值 ± 标准差`（例如：`85.3 ± 1.2`）
- **相对性能**: 相对于完整系统的性能下降百分比
- **统计显著性**: 使用t检验，报告p值（*表示p<0.05，**表示p<0.01，***表示p<0.001）

### 5.3 实验执行顺序

1. **阶段1**: 运行完整系统，建立性能基准
2. **阶段2**: 执行核心组件消融实验（表2.2）
3. **阶段3**: 执行防御保护消融实验（表3.2和3.3）
4. **阶段4**: 执行损失函数消融实验（表4.2）
5. **阶段5**: 汇总结果，进行统计分析，生成报告

---

**文档版本**: V2.0  
**最后更新**: 2024-11-13  
**状态**: 设计阶段（待实现）
