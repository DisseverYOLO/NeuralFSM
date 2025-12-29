# FSM自动生成指南
# FSM Auto-Generation Guide

## 📋 概述

当运行 `run_experiment_1_fsm_complete.py` 时，系统现在支持**自动生成缺失的FSM**，包括：

1. **普通数据集**（GSM8K、HumanEval、HotpotQA、ALFWorld、MATH）
2. **MMLU数据集**（包括类别级FSM）

---

## 🚀 使用方法

### 1. 普通数据集（自动生成）

```bash
python run_experiment_1_fsm_complete.py \
    --domains gsm8k humaneval hotpotqa alfworld math \
    --generate_fsm_if_missing
```

**行为**:
- ✅ 如果缓存中存在FSM，直接使用
- ✅ 如果缓存不存在且 `--generate_fsm_if_missing` 启用，自动生成
- ✅ 生成的FSM会自动保存到缓存

---

### 2. MMLU数据集（类别级FSM自动生成）

```bash
python run_experiment_1_fsm_complete.py \
    --domains mmlu \
    --mmlu_use_category_fsm \
    --generate_fsm_if_missing
```

**行为**:
- ✅ **实验前检查**: 扫描MMLU数据，识别所有类别
- ✅ **批量生成**: 为缺失的类别自动生成FSM
- ✅ **训练时生成**: 如果训练时遇到未预生成的类别，也会自动生成
- ✅ **缓存保存**: 所有生成的FSM都会保存到 `fsm_cache/mmlu/{category}/`

---

## 🔍 工作流程

### 普通数据集流程

```
1. 检查缓存
   ├─ 存在 → 使用缓存 ✅
   └─ 不存在
      ├─ generate_fsm_if_missing=True → 自动生成 → 保存缓存 ✅
      └─ generate_fsm_if_missing=False → 使用默认FSM ⚠️
```

### MMLU类别级FSM流程

```
1. 实验前（run_experiment_1_fsm_complete.py）
   ├─ 扫描MMLU数据，获取所有类别
   ├─ 检查每个类别的缓存
   │  ├─ 存在 → 跳过 ✅
   │  └─ 不存在 → 生成并保存 ✅
   └─ 显示统计信息

2. 训练时（train_fsm_mas_v2.py）
   ├─ 遇到新类别
   ├─ 检查缓存
   │  ├─ 存在 → 加载 ✅
   │  └─ 不存在
   │     ├─ generate_fsm_if_missing=True → 自动生成 → 保存缓存 ✅
   │     └─ generate_fsm_if_missing=False → 使用默认FSM ⚠️
   └─ 继续训练
```

---

## ⚙️ 配置参数

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--generate_fsm_if_missing` | 如果缓存不存在，自动生成FSM | `True` |
| `--mmlu_use_category_fsm` | MMLU使用类别级FSM | `True` |
| `--fsm_cache_dir` | FSM缓存目录 | `./fsm_cache` |

### 配置示例

```python
config = {
    'generate_fsm_if_missing': True,  # 启用自动生成
    'mmlu_use_category_fsm': True,     # MMLU使用类别级FSM
    'fsm_cache_dir': './fsm_cache'    # 缓存目录
}
```

---

## 📊 输出示例

### 普通数据集

```
📋 检查 GSM8K 的FSM...
  ✅ 找到缓存FSM
    - 状态数: 5
    - 智能体数: 4
```

或

```
📋 检查 HOTPOTQA 的FSM...
  🔨 生成新FSM...
    ✅ FSM生成完成 (成本: $0.0234)
```

### MMLU类别级FSM

```
📋 检查 MMLU 的FSM...
  ℹ️  MMLU使用类别级FSM
  🔍 检查MMLU类别FSM缓存...
  📊 发现 57 个类别需要FSM
      🔨 生成类别 'abstract_algebra' 的FSM...
        ✅ 完成 (成本: $0.0234)
      🔨 生成类别 'astronomy' 的FSM...
        ✅ 完成 (成本: $0.0234)
      ...
  📊 统计: 10 个已缓存, 47 个新生成
```

---

## 💡 最佳实践

### 1. 首次运行

**推荐**: 启用自动生成，让系统自动创建所有需要的FSM

```bash
python run_experiment_1_fsm_complete.py \
    --domains mmlu \
    --mmlu_use_category_fsm \
    --generate_fsm_if_missing
```

### 2. 后续运行

**推荐**: 使用已缓存的FSM，加快启动速度

```bash
python run_experiment_1_fsm_complete.py \
    --domains mmlu \
    --mmlu_use_category_fsm \
    --generate_fsm_if_missing  # 仍然启用，以防有新类别
```

### 3. 仅使用缓存（不生成）

如果确定所有FSM都已生成，可以禁用自动生成：

```bash
python run_experiment_1_fsm_complete.py \
    --domains mmlu \
    --mmlu_use_category_fsm \
    --no-generate_fsm_if_missing  # 禁用自动生成
```

---

## 🔧 手动生成MMLU类别FSM

如果需要手动预生成所有MMLU类别FSM：

```bash
python generate_mmlu_category_fsms.py \
    --output_dir ./fsm_cache/mmlu \
    --use_azure False
```

这会为所有57个MMLU类别生成FSM，并保存到缓存。

---

## ⚠️ 注意事项

1. **LLM成本**: 自动生成FSM会调用LLM，产生成本。每个FSM大约 $0.02-0.05。

2. **生成时间**: 
   - 普通数据集: 约30-60秒/个
   - MMLU类别: 约30-60秒/类别
   - 57个MMLU类别: 约30-60分钟

3. **缓存位置**: 
   - 普通数据集: `fsm_cache/{dataset}/`
   - MMLU类别: `fsm_cache/mmlu/{category}/`

4. **断点续传**: 系统会自动跳过已缓存的FSM，支持断点续传。

---

## ✅ 验证

运行测试脚本验证所有数据集：

```bash
python test_all_datasets.py
```

这会检查：
- ✅ 数据加载
- ✅ FSM生成模板
- ✅ 领域提示集合

---

**文档版本**: V1.0  
**最后更新**: 2024-11-13

