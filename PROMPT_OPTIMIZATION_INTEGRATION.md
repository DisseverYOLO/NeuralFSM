# 提示词优化功能说明文档
# Prompt Optimization Integration Documentation

## 📋 概述

本文档说明提示词优化功能在训练阶段的实现和集成方式。

---

## 🎯 优化目标

**目标**: 在训练阶段进行提示词层面的优化，通过数据驱动的方式自动选择和改进提示词，提升模型性能。

---

## 🔄 训练阶段的提示词优化流程

### 阶段1: 训练前初始化

**位置**: `train_fsm_mas_v2.py` 的 `__init__` 方法

**优化内容**:
- ✅ 初始化提示词优化器（`PromptOptimizer`）
- ✅ 注册多个提示词模板版本（基础版、详细版、简洁版等）
- ✅ 加载Few-shot示例库
- ✅ 配置A/B测试参数

**代码示例**:
```python
if self.config.get('enable_prompt_optimization', False):
    from neural_fsm_mas.prompt_optimization import (
        create_prompt_optimizer,
        PromptOptimizationConfig
    )
    
    opt_config = PromptOptimizationConfig(
        enable_ab_test=True,
        ab_test_ratio=0.1,  # 10%使用测试模板
        auto_switch_best=True,
        switch_interval=100,  # 每100个样本检查一次
        few_shot_strategy="hybrid"
    )
    
    self.prompt_optimizer = create_prompt_optimizer(
        domain=domain,
        config=opt_config
    )
    
    # 注册默认模板
    self.prompt_optimizer.register_default_templates()
```

---

### 阶段2: 训练循环中动态优化

**位置**: `train_fsm_mas_v2.py` 的 `_train_epoch` 方法

**优化内容**:
- ✅ **动态获取优化后的提示词**: 根据当前问题选择最佳提示词模板和Few-shot示例
- ✅ **A/B测试**: 自动分配测试流量，对比不同模板效果
- ✅ **性能追踪**: 记录每个模板和示例的使用效果

**代码流程**:
```python
for question_data in batch:
    # 1. 获取优化后的提示词
    formatted_question = self.data_processor.format_question_for_agents(question_data, domain)
    
    if self.prompt_optimizer:
        optimized_prompt, prompt_metadata = self.prompt_optimizer.get_prompt(
            question=formatted_question,
            agent_role=agent_role
        )
        # 更新agent的system_prompt
        agent_node.reasoning_llm.system_prompt = optimized_prompt
    
    # 2. 执行推理
    final_answer, execution_log = await self._execute_fsm_with_learned_structure(...)
    
    # 3. 记录结果并更新性能
    is_correct = self._check_correctness(final_answer, ground_truth, domain)
    
    if self.prompt_optimizer:
        self.prompt_optimizer.record_result(
            template_id=prompt_metadata['template_id'],
            is_correct=is_correct,
            response_time=response_time,
            token_usage=token_count,
            cost=cost,
            few_shot_example_ids=prompt_metadata.get('few_shot_example_ids', [])
        )
```

---

### 阶段3: 自动优化和切换

**位置**: `PromptOptimizer.record_result` 方法内部

**优化内容**:
- ✅ **性能统计**: 更新每个模板的准确率、响应时间、成本等指标
- ✅ **定期检查**: 每N个样本检查一次性能（可配置）
- ✅ **自动切换**: 如果发现更好的模板，自动切换到最佳模板

**优化逻辑**:
```python
# 在record_result中
self.sample_count += 1

# 每switch_interval个样本检查一次
if (self.sample_count - self.last_switch_check >= self.config.switch_interval):
    current_template = self.template_manager.get_template()
    best_template = self.template_manager.get_best_template("accuracy")
    
    # 如果最佳模板性能提升超过5%，自动切换
    if best_template and best_template.success_rate > current_template.success_rate * 1.05:
        self.template_manager.switch_to_best_template("accuracy")
        print(f"🔄 切换到最佳模板: {best_template.template_name}")
```

---

### 阶段4: Few-shot示例动态选择

**位置**: `PromptOptimizer.get_prompt` 方法

**优化内容**:
- ✅ **相似度选择**: 根据问题相似度选择最相关的示例
- ✅ **性能选择**: 根据示例历史效果选择最佳示例
- ✅ **混合策略**: 结合相似度和性能的综合选择

**选择策略**:
```python
# 1. 相似度策略：选择与当前问题最相似的示例
few_shot_examples = selector.select_examples(
    question=question,
    question_embedding=embedding,
    strategy="similarity"
)

# 2. 性能策略：选择历史效果最好的示例
few_shot_examples = selector.select_examples(
    question=question,
    strategy="performance"
)

# 3. 混合策略：相似度60% + 性能40%
few_shot_examples = selector.select_examples(
    question=question,
    question_embedding=embedding,
    strategy="hybrid"
)
```

---

## 📊 优化效果追踪

### 性能指标

每个提示词模板追踪以下指标：

1. **准确率** (`accuracy`): 正确回答的比例
2. **成功率** (`success_rate`): 成功次数 / 总次数
3. **平均响应时间** (`avg_response_time`): 平均LLM响应时间（秒）
4. **平均Token使用** (`avg_token_usage`): 平均每次调用的Token数
5. **总成本** (`cost`): 累计LLM调用成本

### 性能报告

训练过程中可以随时查看性能摘要：

```python
summary = self.prompt_optimizer.get_performance_summary()
print(summary)
# {
#   'domain': 'gsm8k',
#   'sample_count': 1000,
#   'template_summary': {
#     'current_template': 'enhanced',
#     'templates': {
#       'basic': {'accuracy': 0.75, 'total_count': 900, ...},
#       'enhanced': {'accuracy': 0.82, 'total_count': 100, ...}
#     }
#   }
# }
```

---

## 🔧 集成点说明

### 1. 训练器初始化 (`train_fsm_mas_v2.py`)

**集成点**: `FSMMultiAgentSystemTrainerV2.__init__`

**功能**:
- 根据配置初始化提示词优化器
- 注册默认模板或加载自定义模板
- 准备Few-shot示例库

---

### 2. 训练循环 (`train_fsm_mas_v2.py`)

**集成点**: `FSMMultiAgentSystemTrainerV2._train_epoch`

**功能**:
- 为每个问题获取优化后的提示词
- 更新agent的system_prompt
- 执行推理
- 记录结果并更新性能

---

### 3. 验证循环 (`train_fsm_mas_v2.py`)

**集成点**: `FSMMultiAgentSystemTrainerV2._validate_epoch`

**功能**:
- 使用当前最佳模板进行验证
- 不进行A/B测试（使用稳定模板）

---

### 4. Agent执行 (`agent_topology/multi_agent_topology.py`)

**集成点**: `MultiAgentTopologyManager.execute_fsm_reasoning`

**功能**:
- Agent使用优化后的system_prompt执行推理
- 支持Few-shot示例注入

---

## 📈 优化策略

### 策略1: A/B测试

**目的**: 对比不同提示词版本的效果

**实现**:
- 90%的请求使用当前最佳模板
- 10%的请求随机使用其他模板（测试组）
- 自动记录和对比性能

**优势**: 
- 持续探索更好的提示词
- 数据驱动的决策
- 避免过早收敛到次优解

---

### 策略2: 自动切换

**目的**: 根据性能自动选择最佳模板

**实现**:
- 每N个样本检查一次性能
- 如果发现更好的模板（性能提升>5%），自动切换
- 确保始终使用最佳模板

**优势**:
- 自动化优化过程
- 无需人工干预
- 持续改进

---

### 策略3: Few-shot动态选择

**目的**: 为每个问题选择最相关的示例

**实现**:
- 基于问题相似度选择
- 基于示例历史效果选择
- 混合策略（相似度+性能）

**优势**:
- 个性化Few-shot示例
- 提升示例相关性
- 提高推理质量

---

## 🎯 优化效果

### 预期改进

1. **准确率提升**: 通过选择最佳提示词，预期提升2-5%
2. **成本优化**: 通过选择高效的Few-shot示例，减少Token使用
3. **响应时间**: 通过优化提示词，可能减少推理时间
4. **鲁棒性**: 通过A/B测试，发现更鲁棒的提示词

### 实际效果追踪

训练过程中会记录：
- 每个模板的使用次数和准确率
- Few-shot示例的效果
- 自动切换历史
- 性能趋势

---

## 📝 配置选项

### 训练器配置

```python
config = {
    # 启用提示词优化
    'enable_prompt_optimization': True,
    
    # A/B测试配置
    'prompt_ab_test': True,
    'prompt_ab_test_ratio': 0.1,  # 10%测试流量
    
    # 自动切换配置
    'prompt_auto_switch': True,
    'prompt_switch_interval': 100,  # 每100个样本检查
    
    # Few-shot配置
    'few_shot_strategy': 'hybrid',  # similarity, performance, hybrid
    'max_few_shot_examples': 3,
    
    # 性能指标
    'prompt_performance_metric': 'accuracy'  # accuracy, success_rate, cost
}
```

---

## ✅ 实现状态

### ✅ 已完成

1. **提示词模板管理器** - 支持多版本管理和性能追踪
2. **Few-shot选择器** - 支持动态选择策略
3. **提示词优化器** - 集成所有功能
4. **A/B测试** - 自动对比不同模板
5. **自动切换** - 根据性能自动选择最佳模板
6. **性能追踪** - 详细的指标记录

### 🔄 待集成

1. **训练器集成** - 在`train_fsm_mas_v2.py`中集成
2. **实验脚本集成** - 在实验脚本中添加配置选项
3. **Agent提示词更新** - 在执行前更新agent的system_prompt

---

## 🚀 下一步

1. 在训练器中集成提示词优化功能
2. 在实验脚本中添加配置选项
3. 测试和验证优化效果
4. 根据实际效果调整策略

---

**文档版本**: V1.0  
**最后更新**: 2024-11-13  
**状态**: ✅ 功能已完成，待集成到训练流程

