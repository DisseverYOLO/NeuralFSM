# 🧪 NeuralFSM 实验完整检查清单

## 📋 总览

本文档提供完整的实验检查清单,确保所有关键组件和脚本都已就位

---

## ✅ 核心模块实现状态

### 1. 保护机制核心模块 (6个)

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| 图中心性分析 | `neural_fsm_mas/defense_mechanisms/simplified_centrality.py` | 341 | ✅ 完成 |
| 异常检测 | `neural_fsm_mas/defense_mechanisms/simplified_anomaly.py` | 332 | ✅ 完成 |
| 信任计算 | `neural_fsm_mas/defense_mechanisms/trust_calculator.py` | 75 | ✅ 完成 |
| 消息权重 | `neural_fsm_mas/defense_mechanisms/message_weight_calculator.py` | 128 | ✅ 完成 |
| 保护损失 | `neural_fsm_mas/defense_mechanisms/protection_loss.py` | 224 | ✅ 完成 |
| 集成TGN | `neural_fsm_mas/defense_mechanisms/protected_tgn.py` | 386 | ✅ 完成 |
| 导出模块 | `neural_fsm_mas/defense_mechanisms/__init__.py` | 77 | ✅ 完成 |

**小计**: 1563行核心代码 ✅

---

### 2. 训练和集成模块 (3个)

| 模块 | 文件 | 状态 |
|------|------|------|
| 原始训练器 | `neural_fsm_mas/train_neural_mas.py` | ✅ 已存在 |
| 保护训练器 | `neural_fsm_mas/train_neural_mas_protected.py` | ✅ 已创建 |
| FSM生成器 | `neural_fsm_mas/fsm_integration/fsm_mas_generator.py` | ⚠️ 需修改 |

**状态**: 2/3 完成,1个需修改

---

### 3. 实验运行脚本 (5个)

| 脚本 | 文件 | 用途 | 状态 |
|------|------|------|------|
| 实验1 | `run_experiment_1_baseline.py` | Baseline (无保护) | ✅ 已创建 |
| 实验2 | `run_experiment_2_protected.py` | 带保护机制 | ✅ 已创建 |
| 鲁棒性测试 | `run_robustness_test.py` | 异常注入实验 | ✅ 已创建 |
| 批量运行 | `run_all_experiments.sh` | 运行所有实验 | ✅ 已创建 |
| 可视化 | `visualize_results.py` | 结果可视化 | ⚠️ 待创建 |

**状态**: 4/5 完成

---

### 4. 文档 (7个)

| 文档 | 文件 | 内容 | 状态 |
|------|------|------|------|
| 简化设计 | `DEFENSE_SIMPLIFIED_DESIGN.md` | 详细算法设计 | ✅ 完成 |
| 实现指南 | `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` | 实现路线图 | ✅ 完成 |
| 实现完成 | `IMPLEMENTATION_COMPLETE.md` | 模块清单 | ✅ 完成 |
| 集成指南 | `INTEGRATION_GUIDE.md` | 集成步骤 | ✅ 完成 |
| 最终报告 | `PROTECTION_MECHANISM_FINAL_REPORT.md` | 技术报告 | ✅ 完成 |
| 实验清单 | `EXPERIMENT_CHECKLIST.md` | 本文档 | ✅ 完成 |
| 对比方案 | `COMPARISON_ORIGINAL_VS_SIMPLIFIED.md` | 待创建 | ⚠️ 待创建 |

**状态**: 6/7 完成

---

## 🔧 待完成的关键任务

### 优先级1: 必须完成 ⚠️

- [ ] **修改 `fsm_mas_generator.py`**
  - 在TGN初始化后添加ProtectedTGN包装
  - 位置: 约200-250行
  - 参考: `INTEGRATION_GUIDE.md` 步骤1

- [ ] **创建 `visualize_results.py`**
  - 可视化准确率对比
  - 可视化鲁棒性测试结果
  - 生成保护优先级热力图
  - 展示学习到的权重

---

### 优先级2: 推荐完成 📊

- [ ] **创建对比文档** `COMPARISON_ORIGINAL_VS_SIMPLIFIED.md`
  - 原始方案 vs 简化方案
  - 性能对比
  - 使用场景建议

- [ ] **创建配置文件模板** `config_template.yaml`
  - 保护机制参数
  - 训练参数
  - 实验参数

---

## 📝 实验运行指南

### 实验1: Baseline (无保护机制)

```bash
python run_experiment_1_baseline.py \
    --mmlu_data_path ./datasets/mmlu \
    --output_dir ./results/experiment1_baseline \
    --num_epochs 50 \
    --batch_size 16
```

**预期输出**:
- 训练日志
- 每个领域的准确率
- 平均测试准确率
- 结果摘要 JSON

**预期时间**: 约2-4小时 (取决于硬件)

---

### 实验2: 带保护机制

```bash
python run_experiment_2_protected.py \
    --mmlu_data_path ./datasets/mmlu \
    --output_dir ./results/experiment2_protected \
    --baseline_results ./results/experiment1_baseline/experiment1_summary.json \
    --num_epochs 50 \
    --lambda_protect 0.1 \
    --visualize
```

**预期输出**:
- 训练日志 (包含保护损失)
- 每个领域的准确率
- 学习到的权重
- 与Baseline对比
- 保护优先级可视化图

**预期时间**: 约2.5-5小时 (+15% vs Baseline)

---

### 鲁棒性测试

```bash
# 频率攻击
python run_robustness_test.py \
    --attack_type frequency \
    --attack_ratio 0.1 \
    --num_epochs 10

# 语义攻击
python run_robustness_test.py \
    --attack_type semantic \
    --attack_ratio 0.1 \
    --num_epochs 10

# 混合攻击
python run_robustness_test.py \
    --attack_type mixed \
    --attack_ratio 0.1 \
    --num_epochs 10
```

**预期输出**:
- Baseline在攻击下的准确率
- 保护机制在攻击下的准确率
- 鲁棒性提升量化指标

**预期时间**: 每个约30分钟

---

### 批量运行所有实验

```bash
# Linux/Mac
chmod +x run_all_experiments.sh
./run_all_experiments.sh

# Windows (PowerShell)
bash run_all_experiments.sh
```

**预期时间**: 约6-10小时

---

## 📊 预期实验结果

### 准确率对比 (MMLU)

| 方法 | 预期准确率 | 训练时间 |
|------|-----------|---------|
| Baseline | 74-76% | 基准 |
| Protected | 74.5-76.5% | +12-15% |
| 差异 | +0.3-0.5% | 可接受 |

### 鲁棒性对比 (10%异常注入)

| 攻击类型 | Baseline下降 | Protected下降 | 鲁棒性提升 |
|---------|-------------|---------------|-----------|
| 频率 | -5~-7% | -2~-3% | +3~4% |
| 语义 | -6~-8% | -2~-4% | +4~4% |
| 混合 | -8~-10% | -3~-5% | +5~5% |

---

## ⚠️ 关键注意事项

### 1. 数据准备

确保MMLU数据集位于正确路径:
```bash
datasets/mmlu/
├── test/
├── dev/
└── val/
```

### 2. 环境依赖

确保已安装所有依赖:
```bash
pip install -r requirements.txt
```

关键依赖:
- torch >= 1.12
- torch_geometric
- networkx
- matplotlib
- sentence-transformers

### 3. GPU内存

- **Baseline**: 约2-3GB
- **Protected**: 约2.5-3.5GB (+15%)

如果内存不足,减小:
- `batch_size`: 16 → 8
- `memory_dim`: 128 → 64
- `weight_hidden_dim`: 64 → 32

### 4. LLM API

确保配置了LLM API密钥:
```yaml
# config.yaml
OPENAI_API_KEY: "your-key-here"
OPENAI_API_BASE: "https://api.openai.com/v1"
```

---

## 🐛 常见问题

### Q1: 导入错误 `ModuleNotFoundError: No module named 'torch'`

**解决**:
```bash
pip install torch torch_geometric
```

### Q2: `AttributeError: 'NeuralTemporalGraph' object has no attribute 'compute_loss'`

**原因**: 模型不是ProtectedTGN

**解决**: 确保配置中设置 `use_protection: True`

### Q3: 训练非常慢

**可能原因**:
1. 没有GPU
2. batch_size太小
3. num_epochs太多

**解决**:
```python
# 快速测试配置
config = {
    'num_epochs': 10,  # 减少epoch
    'batch_size': 32,  # 增加batch_size
    'num_rounds': 2    # 减少交互轮数
}
```

### Q4: 内存溢出

**解决**:
1. 减小batch_size
2. 减小feature_dim
3. 减小memory_dim
4. 使用更少的agents

---

## ✅ 实验完成检查清单

### 实验前检查
- [ ] 所有依赖已安装
- [ ] MMLU数据集已下载
- [ ] LLM API已配置
- [ ] GPU可用 (推荐)
- [ ] 磁盘空间充足 (>10GB)

### 实验1检查
- [ ] Baseline训练完成
- [ ] 生成 `experiment1_summary.json`
- [ ] 所有领域都有结果
- [ ] 平均准确率合理 (>70%)

### 实验2检查
- [ ] 保护机制训练完成
- [ ] 生成 `experiment2_summary.json`
- [ ] 与Baseline进行了对比
- [ ] 学习到的权重已记录
- [ ] 保护优先级已可视化

### 鲁棒性测试检查
- [ ] 三种攻击都已测试
- [ ] Baseline和Protected都已测试
- [ ] 生成鲁棒性摘要
- [ ] 鲁棒性提升已量化

### 最终检查
- [ ] 所有结果文件已生成
- [ ] 可视化图表已生成
- [ ] 结果符合预期
- [ ] 实验可重现

---

## 📈 下一步

完成所有实验后:

1. **分析结果**
   - 对比准确率
   - 分析鲁棒性
   - 检查学习到的权重

2. **撰写论文**
   - 方法章节
   - 实验章节
   - 结果分析

3. **代码开源**
   - 整理代码
   - 添加README
   - 发布到GitHub

---

## 🎯 成功标准

实验被认为成功,如果:

1. ✅ **准确率**: Protected ≈ Baseline (差异 < 1%)
2. ✅ **鲁棒性**: Protected显著优于Baseline (+3% ~ +6%)
3. ✅ **性能**: 训练时间开销 < 20%
4. ✅ **可解释性**: 学习到的权重合理
5. ✅ **可复现**: 实验结果稳定

---

**最后更新**: 2025-10-31  
**状态**: 核心模块完成,集成进行中  
**完成度**: 85% (17/20 任务完成)

