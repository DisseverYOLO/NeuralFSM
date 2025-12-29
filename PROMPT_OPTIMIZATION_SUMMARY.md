# 提示词优化框架总结（已合并）
# Prompt Optimization Framework Summary (Archived)

> ℹ️ 该文档内容已合并至 `PROMPT_OPTIMIZATION_FINAL_SUMMARY.md`，以避免重复。以下为历史内容备份。

<!-- ARCHIVED CONTENT -->

## ✅ 已完成的功能

### 1. 提示词模板管理器 (`prompt_template_manager.py`)

**功能**:
- ✅ 管理多个提示词模板版本
- ✅ 支持A/B测试
- ✅ 性能追踪（准确率、响应时间、成本等）
- ✅ 自动选择最佳模板
- ✅ 缓存持久化

**核心类**:
- `PromptTemplate`: 提示词模板数据类
- `PromptPerformance`: 性能指标数据类
- `PromptTemplateManager`: 模板管理器

---

### 2. Few-shot示例选择器 (`few_shot_selector.py`)

**功能**:
- ✅ 动态选择Few-shot示例
- ✅ 基于相似度选择
- ✅ 基于性能选择
- ✅ 混合策略（相似度+性能）
- ✅ 示例性能追踪

**核心类**:
- `FewShotExample`: Few-shot示例数据类
- `FewShotSelector`: 示例选择器

---

### 3. 提示词优化器 (`prompt_optimizer.py`)

**功能**:
- ✅ 集成模板管理器和Few-shot选择器
- ✅ 自动优化提示词
- ✅ A/B测试支持
- ✅ 性能追踪和报告
- ✅ 自动切换到最佳模板

**核心类**:
- `PromptOptimizationConfig`: 优化配置
- `PromptOptimizer`: 提示词优化器

---

## 🚀 使用方法

### 基本使用

```python
from neural_fsm_mas.prompt_optimization import (
    create_prompt_optimizer,
    PromptOptimizationConfig
)

# 创建优化器
config = PromptOptimizationConfig(
    enable_ab_test=True,
    auto_switch_best=True
)

optimizer = create_prompt_optimizer('gsm8k', config)

# 注册模板
optimizer.template_manager.register_template(
    template_id="v1",
    template_name="Version 1",
    system_prompt="You are an expert..."
)

# 获取优化后的提示词
prompt, metadata = optimizer.get_prompt(question="...")

# 记录结果
optimizer.record_result(
    template_id=metadata['template_id'],
    is_correct=True,
    response_time=1.2,
    token_usage=150,
    cost=0.001
)
```

---

## 📊 集成到训练器

### 在 `train_fsm_mas_v2.py` 中集成

1. **初始化**:
```python
if self.config.get('enable_prompt_optimization', False):
    self.prompt_optimizer = create_prompt_optimizer(
        domain=domain,
        config=PromptOptimizationConfig(...)
    )
```

2. **训练循环中使用**:
```python
# 获取优化后的提示词
prompt, metadata = self.prompt_optimizer.get_prompt(
    question=formatted_question
)

# 记录结果
self.prompt_optimizer.record_result(
    template_id=metadata['template_id'],
    is_correct=is_correct,
    ...
)
```

---

## 🎯 核心特性

### 1. 多版本提示词管理
- 支持注册多个提示词模板
- 自动追踪每个模板的性能
- 支持模板版本控制

### 2. A/B测试
- 自动分配测试流量
- 对比不同模板效果
- 自动切换到最佳模板

### 3. Few-shot动态选择
- 根据问题相似度选择
- 根据示例效果选择
- 混合策略优化

### 4. 性能追踪
- 准确率追踪
- 响应时间追踪
- 成本追踪
- 详细性能报告

### 5. 自动优化
- 定期检查性能
- 自动切换到最佳模板
- 持续优化提示词

---

## 📁 文件结构

```
neural_fsm_mas/prompt_optimization/
├── __init__.py                    # 模块导出
├── prompt_template_manager.py     # 模板管理器
├── few_shot_selector.py           # Few-shot选择器
└── prompt_optimizer.py           # 优化器（集成）
```

---

## ⚙️ 配置选项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `enable_ab_test` | 启用A/B测试 | `True` |
| `ab_test_ratio` | 测试比例 | `0.1` |
| `auto_switch_best` | 自动切换最佳模板 | `True` |
| `switch_interval` | 切换检查间隔 | `100` |
| `few_shot_strategy` | Few-shot策略 | `"hybrid"` |
| `max_few_shot_examples` | 最大示例数 | `3` |
| `performance_metric` | 性能指标 | `"accuracy"` |

---

## 💡 优势

1. **自动化**: 无需手动调整，系统自动优化
2. **数据驱动**: 基于实际训练数据选择最佳提示词
3. **灵活**: 支持多种策略和配置
4. **可追踪**: 详细的性能记录和分析
5. **轻量级**: 集成简单，不影响现有代码结构
6. **可扩展**: 易于添加新的优化策略

---

## 📝 下一步

1. **集成到训练器**: 在 `train_fsm_mas_v2.py` 中集成
2. **添加命令行参数**: 在 `run_experiment_1_fsm_complete.py` 中添加配置选项
3. **性能监控**: 添加定期性能报告输出
4. **模板库**: 为不同数据集预定义模板库

---

**文档版本**: V1.0  
**最后更新**: 2024-11-13  
**状态**: ✅ 已完成

-->
