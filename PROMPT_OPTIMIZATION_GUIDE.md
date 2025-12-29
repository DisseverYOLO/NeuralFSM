# 提示词优化框架使用指南
# Prompt Optimization Framework Guide

## 📋 概述

提示词优化框架是一个集成到训练流程中的小功能，支持：

1. **多版本提示词管理** - 支持多个提示词模板版本
2. **A/B测试** - 自动测试不同提示词版本的效果
3. **性能追踪** - 记录每个提示词版本的准确率、响应时间、成本等
4. **Few-shot示例动态选择** - 根据问题相似度和示例效果选择最佳示例
5. **自动优化** - 根据性能自动切换到最佳提示词

---

## 🚀 快速开始

### 1. 在训练器中集成

```python
from neural_fsm_mas.prompt_optimization import (
    create_prompt_optimizer,
    PromptOptimizationConfig
)

# 创建优化器
config = PromptOptimizationConfig(
    enable_ab_test=True,        # 启用A/B测试
    ab_test_ratio=0.1,          # 10%使用测试模板
    auto_switch_best=True,      # 自动切换到最佳模板
    switch_interval=100,        # 每100个样本检查一次
    few_shot_strategy="hybrid", # Few-shot选择策略
    max_few_shot_examples=3,    # 最多3个示例
    performance_metric="accuracy" # 性能指标
)

prompt_optimizer = create_prompt_optimizer(
    domain='gsm8k',
    config=config
)

# 注册自定义提示词模板
prompt_optimizer.template_manager.register_template(
    template_id="enhanced_v1",
    template_name="Enhanced V1",
    system_prompt="You are an expert mathematician. Solve problems step by step..."
)

# 在训练循环中使用
for question_data in training_data:
    # 获取优化后的提示词
    prompt, metadata = prompt_optimizer.get_prompt(
        question=question_data['question'],
        agent_role="MathematicalReasoner"
    )
    
    # 使用提示词调用LLM
    result = llm.chat(prompt)
    
    # 记录结果
    is_correct = check_answer(result, question_data['answer'])
    prompt_optimizer.record_result(
        template_id=metadata['template_id'],
        is_correct=is_correct,
        response_time=response_time,
        token_usage=token_count,
        cost=cost,
        few_shot_example_ids=metadata.get('few_shot_example_ids', [])
    )
```

---

## 📊 功能详解

### 1. 提示词模板管理

#### 注册多个模板版本

```python
# 基础模板
prompt_optimizer.template_manager.register_template(
    template_id="basic",
    template_name="Basic Template",
    system_prompt="You are a helpful assistant."
)

# 增强模板
prompt_optimizer.template_manager.register_template(
    template_id="enhanced",
    template_name="Enhanced Template",
    system_prompt="You are an expert. Think step by step...",
    few_shot_examples=[
        {"question": "...", "answer": "..."}
    ]
)
```

#### 查看性能

```python
summary = prompt_optimizer.get_performance_summary()
print(summary)
# {
#   'domain': 'gsm8k',
#   'templates': {
#     'basic': {'accuracy': 0.75, 'total_count': 100, ...},
#     'enhanced': {'accuracy': 0.82, 'total_count': 100, ...}
#   }
# }
```

---

### 2. A/B测试

框架自动进行A/B测试：

- **当前模板**: 90%的请求使用
- **测试模板**: 10%的请求使用（随机选择其他模板）
- **性能对比**: 自动记录两个模板的性能
- **自动切换**: 如果测试模板性能更好，自动切换

---

### 3. Few-shot示例选择

#### 添加示例

```python
# 添加Few-shot示例
prompt_optimizer.few_shot_selector.add_example(
    example_id="ex1",
    question="What is 2+2?",
    answer="4",
    reasoning="Adding 2 and 2 gives 4.",
    embedding=question_embedding  # 可选：问题嵌入向量
)
```

#### 选择策略

- **similarity**: 根据问题相似度选择
- **performance**: 根据示例效果选择
- **hybrid**: 混合策略（相似度60% + 性能40%）

---

### 4. 自动优化

框架会在训练过程中：

1. **追踪性能**: 记录每个模板和示例的效果
2. **定期检查**: 每N个样本检查一次（可配置）
3. **自动切换**: 如果发现更好的模板，自动切换
4. **性能报告**: 提供详细的性能摘要

---

## 🔧 集成到训练器

### 修改 `train_fsm_mas_v2.py`

```python
# 在 __init__ 中初始化
from neural_fsm_mas.prompt_optimization import create_prompt_optimizer

class FSMMultiAgentSystemTrainerV2:
    def __init__(self, ...):
        # ... 现有代码 ...
        
        # ✨ 初始化提示词优化器
        if self.config.get('enable_prompt_optimization', False):
            from neural_fsm_mas.prompt_optimization import (
                create_prompt_optimizer,
                PromptOptimizationConfig
            )
            opt_config = PromptOptimizationConfig(
                enable_ab_test=self.config.get('prompt_ab_test', True),
                ab_test_ratio=self.config.get('prompt_ab_test_ratio', 0.1),
                auto_switch_best=self.config.get('prompt_auto_switch', True),
                switch_interval=self.config.get('prompt_switch_interval', 100),
                few_shot_strategy=self.config.get('few_shot_strategy', 'hybrid'),
                max_few_shot_examples=self.config.get('max_few_shot_examples', 3)
            )
            self.prompt_optimizer = create_prompt_optimizer(
                domain=domain,
                config=opt_config
            )
            # 注册默认模板
            self.prompt_optimizer.register_default_templates()
        else:
            self.prompt_optimizer = None

# 在训练循环中使用
async def _train_epoch(self, ...):
    for question_data in batch:
        # ... 现有代码 ...
        
        # ✨ 获取优化后的提示词
        if self.prompt_optimizer:
            prompt, prompt_metadata = self.prompt_optimizer.get_prompt(
                question=formatted_question,
                agent_role=agent_role
            )
            # 使用优化后的提示词更新agent的system_prompt
            # ...
        
        # ... 执行推理 ...
        
        # ✨ 记录结果
        if self.prompt_optimizer:
            is_correct = self._check_correctness(...)
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

## ⚙️ 配置参数

### PromptOptimizationConfig

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable_ab_test` | 启用A/B测试 | `True` |
| `ab_test_ratio` | 测试比例 | `0.1` |
| `auto_switch_best` | 自动切换到最佳模板 | `True` |
| `switch_interval` | 切换检查间隔（样本数） | `100` |
| `few_shot_strategy` | Few-shot选择策略 | `"hybrid"` |
| `max_few_shot_examples` | 最大Few-shot示例数 | `3` |
| `performance_metric` | 性能指标 | `"accuracy"` |

---

## 📈 性能追踪

### 查看性能摘要

```python
summary = prompt_optimizer.get_performance_summary()
print(json.dumps(summary, indent=2))
```

输出示例：
```json
{
  "domain": "gsm8k",
  "sample_count": 1000,
  "template_summary": {
    "current_template": "enhanced",
    "templates": {
      "basic": {
        "accuracy": 0.75,
        "total_count": 900,
        "cost": 2.34
      },
      "enhanced": {
        "accuracy": 0.82,
        "total_count": 100,
        "cost": 0.26
      }
    }
  }
}
```

---

## 💡 最佳实践

### 1. 初始模板设计

- 创建2-3个不同风格的模板（详细、简洁、结构化）
- 让系统自动测试并选择最佳模板

### 2. Few-shot示例

- 选择高质量、多样化的示例
- 让系统根据相似度和效果自动选择

### 3. A/B测试

- 开始时使用较小的测试比例（5-10%）
- 随着数据积累，可以增加测试比例

### 4. 性能监控

- 定期查看性能摘要
- 根据结果调整模板和配置

---

## 🔍 示例场景

### 场景1: 测试不同提示词风格

```python
# 注册多个模板
prompt_optimizer.template_manager.register_template(
    template_id="step_by_step",
    template_name="Step-by-Step",
    system_prompt="Solve problems step by step. Show your reasoning..."
)

prompt_optimizer.template_manager.register_template(
    template_id="direct",
    template_name="Direct Answer",
    system_prompt="Provide direct and concise answers..."
)

# 系统会自动A/B测试并选择最佳模板
```

### 场景2: 动态Few-shot选择

```python
# 添加多个示例
for example in training_examples[:10]:
    prompt_optimizer.few_shot_selector.add_example(
        example_id=f"ex_{example['id']}",
        question=example['question'],
        answer=example['answer'],
        reasoning=example.get('reasoning')
    )

# 系统会根据问题相似度和示例效果自动选择3个最佳示例
```

---

## ✅ 优势

1. **自动化**: 无需手动调整，系统自动优化
2. **数据驱动**: 基于实际训练数据选择最佳提示词
3. **灵活**: 支持多种策略和配置
4. **可追踪**: 详细的性能记录和分析
5. **轻量级**: 集成简单，不影响现有代码结构

---

**文档版本**: V1.0  
**最后更新**: 2024-11-13


