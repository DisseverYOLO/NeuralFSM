# LLM模型配置更新
# LLM Model Configuration Update

## ✅ **更新内容**

所有实验脚本的默认语言模型已从 `gpt-4` 更新为 `gpt-4o-mini`。

---

## 📝 **更新的文件列表**

### **1. 主实验脚本**
- ✅ `run_experiment_1_baseline.py` - 实验1（Baseline）
- ✅ `run_experiment_2_protected.py` - 实验2（保护机制）
- ✅ `run_robustness_test.py` - 鲁棒性测试

### **2. 训练脚本**
- ✅ `neural_fsm_mas/train_neural_mas.py` - 主训练器

### **3. 示例脚本**
- ✅ `neural_fsm_mas/examples/run_mmlu_training.py`

### **4. 智能体实现**
- ✅ `neural_fsm_mas/reasoning_agents/mathematical_reasoning_agent.py`
- ✅ `neural_fsm_mas/reasoning_agents/analytical_reasoning_agent.py`
- ✅ `neural_fsm_mas/reasoning_agents/decision_making_agent.py`
- ✅ `neural_fsm_mas/reasoning_agents/code_generation_agent.py`
- ✅ `neural_fsm_mas/reasoning_agents/adversarial_reasoning_agent.py`

### **5. 配置文件**
- ✅ `neural_fsm_mas/config/training_config.json`

### **6. 文档**
- ✅ `UNIFIED_DATASET_GUIDE.md`
- ✅ `DATASET_USAGE_GUIDE.md`

---

## 🚀 **使用方法**

### **使用默认模型 (gpt-4o-mini)**

所有脚本现在默认使用 `gpt-4o-mini`，无需指定：

```bash
# 实验1
python run_experiment_1_baseline.py --domains mmlu --num_epochs 50

# 实验2
python run_experiment_2_protected.py --domains mmlu --num_epochs 50

# 鲁棒性测试
python run_robustness_test.py --attack_type mixed --attack_ratio 0.1
```

### **使用其他模型 (可选参数)**

如果需要使用其他模型，可以通过 `--llm_name` 参数指定：

```bash
# 使用 gpt-4
python run_experiment_1_baseline.py \
    --domains mmlu \
    --llm_name gpt-4 \
    --num_epochs 50

# 使用 gpt-3.5-turbo
python run_experiment_1_baseline.py \
    --domains gsm8k \
    --llm_name gpt-3.5-turbo \
    --num_epochs 30

# 使用 gpt-4-turbo
python run_experiment_2_protected.py \
    --domains humaneval \
    --llm_name gpt-4-turbo \
    --num_epochs 20
```

---

## 🎯 **支持的模型**

所有OpenAI兼容的模型都可以使用：

### **OpenAI官方模型**
- ✅ `gpt-4o-mini` (默认，推荐用于实验)
- ✅ `gpt-4o`
- ✅ `gpt-4-turbo`
- ✅ `gpt-4`
- ✅ `gpt-3.5-turbo`

### **Azure OpenAI模型**
如果使用Azure OpenAI，需要在配置文件中设置：

```json
{
  "llm": {
    "llm_name": "your-deployment-name",
    "use_azure": true,
    "azure_endpoint": "https://your-resource.openai.azure.com/",
    "api_version": "2024-02-15-preview"
  }
}
```

---

## 💰 **模型选择建议**

### **成本 vs 性能**

| 模型 | 相对成本 | 性能 | 推荐场景 |
|-----|---------|------|---------|
| `gpt-4o-mini` | 💰 最低 | ⭐⭐⭐ 良好 | **实验开发、快速迭代** ⭐ 推荐 |
| `gpt-3.5-turbo` | 💰 低 | ⭐⭐ 中等 | 简单任务、基准测试 |
| `gpt-4-turbo` | 💰💰💰 中等 | ⭐⭐⭐⭐ 优秀 | 复杂推理、最终评估 |
| `gpt-4` | 💰💰💰💰 高 | ⭐⭐⭐⭐⭐ 最佳 | 论文实验、发布结果 |
| `gpt-4o` | 💰💰💰 中等 | ⭐⭐⭐⭐⭐ 最佳 | 平衡性能和成本 |

### **推荐配置**

```bash
# 开发/调试阶段 (使用默认)
python run_experiment_1_baseline.py --domains mmlu --num_epochs 10

# 正式实验 (可选择更强模型)
python run_experiment_1_baseline.py \
    --domains mmlu gsm8k humaneval \
    --llm_name gpt-4o \
    --num_epochs 50
```

---

## 🔧 **配置文件说明**

### **training_config.json**

```json
{
  "llm": {
    "llm_name": "gpt-4o-mini",  // 默认模型
    "use_azure": false,
    "temperature": 0.7,         // 采样温度
    "max_tokens": 2000          // 最大token数
  }
}
```

可以通过修改此文件来更改全局默认模型。

---

## 📊 **代码中的使用**

### **Python脚本中指定模型**

```python
from neural_fsm_mas.train_neural_mas import NeuralMASTrainer

config = {
    'num_epochs': 50,
    'batch_size': 16,
    'llm_name': 'gpt-4o-mini',  # 在这里指定模型
    'domains': ['mmlu']
}

trainer = NeuralMASTrainer(
    config=config,
    dataset_root="./datasets",
    output_dir="./results"
)

await trainer.train_all_domains()
```

### **创建智能体时指定模型**

```python
from neural_fsm_mas.reasoning_agents.mathematical_reasoning_agent import MathematicalReasoningAgent

# 使用默认模型 (gpt-4o-mini)
agent = MathematicalReasoningAgent(
    node_id="agent_1",
    agent_role="Math Solver",
    domain="mathematics"
)

# 指定其他模型
agent = MathematicalReasoningAgent(
    node_id="agent_2",
    agent_role="Math Solver",
    domain="mathematics",
    llm_name="gpt-4"  # 使用gpt-4
)
```

---

## ⚙️ **环境变量**

确保设置OpenAI API密钥：

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# Windows (CMD)
set OPENAI_API_KEY=your-api-key-here
```

---

## ✅ **验证配置**

运行以下命令验证模型配置：

```bash
# 快速测试 (使用默认 gpt-4o-mini)
python run_experiment_1_baseline.py \
    --domains mmlu \
    --num_epochs 1 \
    --batch_size 4

# 测试其他模型
python run_experiment_1_baseline.py \
    --domains mmlu \
    --num_epochs 1 \
    --batch_size 4 \
    --llm_name gpt-4
```

---

## 📝 **总结**

### **关键变化**
- ✅ 默认模型: `gpt-4` → `gpt-4o-mini`
- ✅ 所有脚本支持 `--llm_name` 参数
- ✅ 所有智能体支持自定义模型
- ✅ 配置文件已更新

### **为什么选择 gpt-4o-mini?**
1. **成本效益**: 比gpt-4便宜约90%
2. **性能足够**: 对于大多数实验任务表现良好
3. **快速迭代**: 响应速度快，适合开发调试
4. **易于升级**: 需要时可随时切换到更强模型

### **灵活性**
- ✅ 默认使用经济型模型 (gpt-4o-mini)
- ✅ 随时可切换到更强模型 (--llm_name参数)
- ✅ 不同实验可使用不同模型
- ✅ 单个智能体可使用不同模型

---

**建议**: 在开发和调试阶段使用 `gpt-4o-mini`，在最终评估和论文实验时使用 `gpt-4` 或 `gpt-4o`。

