# 数据集下载与攻击测试指南

## 📋 目录

1. [数据集下载](#1-数据集下载)
2. [攻击方式详解](#2-攻击方式详解)
3. [运行鲁棒性测试](#3-运行鲁棒性测试)
4. [常见问题](#4-常见问题)

---

## 1. 数据集下载

### 🎯 **MMLU数据集**

项目使用 **MMLU (Massive Multitask Language Understanding)** 数据集进行实验。

#### **方法1: 使用Hugging Face (推荐)** ⭐

```bash
# 1. 安装依赖
pip install datasets

# 2. 切换到数据集目录
cd datasets/mmlu

# 3. 运行下载脚本
python download.py --method huggingface
```

**下载后**:
- 数据自动缓存到 `~/.cache/huggingface/datasets/`
- 无需额外配置
- 支持断点续传

**验证下载**:
```bash
python -c "from datasets import load_dataset; ds = load_dataset('cais/mmlu', 'all'); print(f'✅ 下载成功: {len(ds[\"test\"])} 测试样本')"
```

---

#### **方法2: 手动下载**

```bash
# 1. 切换到数据集目录
cd datasets/mmlu

# 2. 运行下载脚本
python download.py --method manual

# 3. 等待下载和解压完成
```

**下载后**:
- 数据位于 `datasets/mmlu/data/`
- 包含57个学科子文件夹
- 每个学科有 `train.csv`, `dev.csv`, `test.csv`

**验证文件结构**:
```bash
# Windows
dir datasets\mmlu\data

# Linux/Mac
ls datasets/mmlu/data/
```

应该看到类似输出:
```
abstract_algebra/
anatomy/
astronomy/
...
(共57个文件夹)
```

---

### 📊 **数据集详情**

| 属性 | 值 |
|------|-----|
| **总题目数** | ~15,908 |
| **学科领域** | 57个 |
| **题目类型** | 4选1选择题 |
| **分割** | train/dev/test |
| **领域覆盖** | STEM、人文、社会科学 |

---

## 2. 攻击方式详解

项目实现了 **3种攻击类型**，用于测试防御机制的鲁棒性。

### 🎯 **攻击1: 频率攻击 (Frequency Attack)**

**定义**: 恶意智能体发送消息的频率异常增加

**实现**:
```python
from neural_fsm_mas.defense_mechanisms import AnomalyInjector

injector = AnomalyInjector(attack_ratio=0.1)  # 10%节点被攻击
anomalous_nodes = injector.select_anomalous_nodes(num_nodes=5)

# 正常消息数: 3条
# 攻击后: 30条 (10倍放大)
anomalous_count = injector.inject_frequency_anomaly(
    node_id=0, 
    normal_count=3, 
    multiplier=10.0
)
```

**效果**:
- ⚡ **正常节点**: 每轮发送3条消息
- 🔥 **恶意节点**: 每轮发送30条消息 (模拟DDoS)

**防御机制如何应对**:
1. 检测消息频率异常 → 计算 `α_freq`
2. 降低该节点的信任分数 → `trust = 1 - α_freq`
3. 减小来自该节点的消息权重 → `w_{i→j} ↓`
4. 过滤低信任消息 → 拓扑裁剪

---

### 🧠 **攻击2: 语义攻击 (Semantic Attack)**

**定义**: 向智能体特征嵌入注入噪声，篡改语义信息

**实现**:
```python
# 原始特征: [num_agents, 384]
agent_features = torch.randn(5, 384)

# 注入高斯噪声 (标准差=5.0)
attacked_features = injector.inject_semantic_anomaly_batch(
    agent_features=agent_features,
    noise_scale=5.0,
    noise_type='gaussian'  # 或 'uniform', 'adversarial'
)

# 效果: 恶意节点的特征被强噪声污染
print(f"原始L2范数: {agent_features[0].norm():.2f}")
print(f"攻击后L2范数: {attacked_features[0].norm():.2f}")  # 显著增大
```

**效果**:
- 🔵 **正常节点**: 特征嵌入保持稳定
- 🔴 **恶意节点**: 特征嵌入被高斯噪声污染

**防御机制如何应对**:
1. 计算语义异常分数 → 与历史均值比较
2. 检测到异常嵌入 → `α_semantic ↑`
3. 降低信任分数 → `trust ↓`
4. 衰减输出影响 → `output_mix = trust * output + (1-trust) * original`

---

### ⚡ **攻击3: 混合攻击 (Mixed Attack)**

**定义**: 同时进行频率攻击和语义攻击

**实现**:
```python
normal_counts = [3] * 5  # 每个节点正常发3条消息

attacked_counts, attacked_features = injector.inject_mixed_anomaly_batch(
    agent_features=agent_features,
    normal_counts=normal_counts,
    freq_multiplier=10.0,   # 频率放大10倍
    noise_scale=5.0,        # 噪声强度5.0
    noise_type='gaussian'
)

# 恶意节点同时:
# 1. 发送30条消息 (vs 正常3条)
# 2. 特征被噪声污染
```

**效果**:
- 💥 **最强攻击**: 频率 + 语义双重攻击
- 🎯 **测试目标**: 验证防御机制能否同时应对多种威胁

**防御机制如何应对**:
1. 同时检测频率和语义异常
2. 综合异常分数: `α = λ_freq * α_freq + λ_semantic * α_semantic`
3. 多层次防御:
   - 拓扑裁剪 (过滤低信任边)
   - 消息权重衰减
   - 输出混合保护

---

### 📊 **攻击对比表**

| 攻击类型 | 频率异常 | 语义异常 | 检测难度 | 破坏力 |
|---------|---------|---------|---------|--------|
| **频率攻击** | ✅ | ❌ | ⭐⭐ | ⭐⭐⭐ |
| **语义攻击** | ❌ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **混合攻击** | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 3. 运行鲁棒性测试

### 🧪 **完整测试流程**

```bash
# 1. 确保已下载数据集
python datasets/mmlu/download.py --method huggingface

# 2. 运行鲁棒性测试
python run_robustness_test.py \
    --mmlu_data_path ./datasets/mmlu \
    --attack_type mixed \
    --attack_ratio 0.1 \
    --num_epochs 10

# 参数说明:
# --attack_type: 攻击类型 (frequency | semantic | mixed)
# --attack_ratio: 攻击节点比例 (0.1 = 10%)
# --num_epochs: 训练轮数 (测试用，可设小值)
```

---

### 📊 **测试输出示例**

```bash
================================================================================
🧪 鲁棒性测试: 异常注入实验
================================================================================
📁 MMLU数据路径: ./datasets/mmlu
📁 输出目录: ./results/robustness_test
⚔️  攻击类型: mixed
⚔️  攻击比例: 10%
================================================================================

================================================================================
📊 测试1: Baseline (无保护机制) 在攻击下的表现
================================================================================
🎯 选择了 1 个异常节点: [2]
📊 训练前测试准确率 (无攻击)...
  无攻击准确率: 0.7500
🚀 开始训练并注入 mixed 异常...
  ⚠️  注入频率异常: 节点2 (3 → 30 条消息)
  ⚠️  注入语义异常: 节点2 (噪声强度=5.0)
📊 评估攻击后的准确率...
  攻击后准确率: 0.5500 (下降: 0.2000)

================================================================================
📊 测试2: 保护机制 在攻击下的表现
================================================================================
🛡️  使用保护机制
🎯 选择了 1 个异常节点: [2]
📊 训练前测试准确率 (无攻击)...
  无攻击准确率: 0.7600
🚀 开始训练并注入 mixed 异常...
  🛡️  检测到异常: 节点2 (trust=0.15)
  🛡️  应用消息衰减: w_{2→*} *= 0.15
📊 评估攻击后的准确率...
  攻击后准确率: 0.6900 (下降: 0.0700)

================================================================================
📊 鲁棒性测试结果汇总
================================================================================

📊 无攻击情况:
  Baseline: 0.7500
  Protected: 0.7600

⚔️  Mixed 攻击后 (10% 节点):
  Baseline: 0.5500 (下降 0.2000)
  Protected: 0.6900 (下降 0.0700)

🛡️  鲁棒性提升: 0.1300
   (保护机制相比Baseline,准确率下降更少 0.1300)

💾 结果已保存到: ./results/robustness_test/robustness_summary_mixed.json
```

---

### 📈 **结果可视化**

```bash
# 生成对比图表
python visualize_results.py \
    --baseline_results ./results/experiment1_baseline/experiment1_summary.json \
    --protected_results ./results/experiment2_protected/experiment2_summary.json \
    --robustness_results ./results/robustness_test/robustness_summary_mixed.json \
    --output_dir ./results/visualizations
```

**生成图表**:
1. `accuracy_comparison.png` - 准确率对比
2. `robustness_comparison.png` - 鲁棒性对比
3. `attack_impact.png` - 攻击影响分析

---

## 4. 常见问题

### ❓ **Q1: 数据集下载失败怎么办？**

**A**: 尝试使用国内镜像:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python download.py --method huggingface
```

---

### ❓ **Q2: 攻击注入后为什么准确率没下降？**

**A**: 可能的原因:
1. **攻击强度不够**: 增加 `--attack_ratio` (如0.2)
2. **噪声太小**: 在代码中增加 `noise_scale` (如10.0)
3. **防御太强**: 正常现象，说明防御有效！

---

### ❓ **Q3: 如何自定义攻击参数？**

**A**: 修改 `run_robustness_test.py`:
```python
# 找到这行 (约第156行):
attacked_features = injector.inject_semantic_anomaly_batch(
    agent_features=original_features,
    noise_scale=5.0,  # 改为 10.0 增强攻击
    noise_type='gaussian'  # 或改为 'adversarial'
)
```

---

### ❓ **Q4: 如何测试单个攻击类型？**

**A**: 指定 `--attack_type`:
```bash
# 只测试频率攻击
python run_robustness_test.py --attack_type frequency

# 只测试语义攻击
python run_robustness_test.py --attack_type semantic
```

---

### ❓ **Q5: 攻击注入器的原理是什么？**

**A**: 核心思想:
1. **选择恶意节点**: 随机选择 `attack_ratio` 比例的节点
2. **修改节点行为**:
   - **频率攻击**: 放大消息计数 (记录到历史)
   - **语义攻击**: 向特征添加噪声 (实际修改嵌入)
3. **防御检测**: 防御机制通过分析历史和当前状态检测异常

---

## ✅ **完整实验流程**

```bash
# Step 1: 下载数据集
cd datasets/mmlu
python download.py --method huggingface
cd ../..

# Step 2: 运行Baseline实验
python run_experiment_1_baseline.py \
    --mmlu_data_path ./datasets/mmlu \
    --num_epochs 50

# Step 3: 运行保护机制实验
python run_experiment_2_protected.py \
    --mmlu_data_path ./datasets/mmlu \
    --num_epochs 50 \
    --visualize

# Step 4: 运行鲁棒性测试 (3种攻击)
for attack in frequency semantic mixed; do
    python run_robustness_test.py \
        --mmlu_data_path ./datasets/mmlu \
        --attack_type $attack \
        --attack_ratio 0.1 \
        --num_epochs 10
done

# Step 5: 可视化所有结果
python visualize_results.py \
    --baseline_results ./results/experiment1_baseline/experiment1_summary.json \
    --protected_results ./results/experiment2_protected/experiment2_summary.json \
    --output_dir ./results/visualizations
```

---

## 📚 **相关文档**

- [数据集README](datasets/README.md) - 数据集详细说明
- [PROTECTION_MECHANISM_FINAL_REPORT.md](PROTECTION_MECHANISM_FINAL_REPORT.md) - 防御机制详解
- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - 快速开始指南
- [MESSAGE_WEIGHT_TRAINING_EXPLAINED.md](MESSAGE_WEIGHT_TRAINING_EXPLAINED.md) - 消息权重训练详解

---

**最后更新**: 2025-11-05

