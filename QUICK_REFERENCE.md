# 快速参考卡片 | Quick Reference Card

## 🚀 立即开始

### **1行命令运行实验**

```bash
python run_experiment_random_sampled_fsm.py --dataset gsm8k --epochs 50 --delta 0.1
```

---

## 📦 核心导入

```python
from neural_fsm_mas import (
    # FSM验证和执行
    FSMValidator, FSMExecutor, validate_fsm, create_fsm_executor,
    # 成本追踪
    LLMCostTracker, create_cost_tracker, print_pricing_table,
    # 损失函数
    CostRegularizedLoss, estimate_baseline_cost,
)
```

---

## 💡 5分钟上手

### **验证FSM**

```python
results = validate_fsm(fsm, auto_fix=True, verbose=True)
```

### **追踪成本**

```python
tracker = create_cost_tracker('gpt-4o-mini')
tracker.start_episode()
cost = tracker.track_call(input_tokens=500, output_tokens=200)
episode_cost = tracker.end_episode()
```

### **执行FSM**

```python
executor = create_fsm_executor(fsm, max_visits_per_state=3)
while executor.can_continue():
    state = executor.get_current_state()
    output = agent.run(state)
    if "<DONE>" in output:
        next_state = executor.select_next_state(output)
        executor.execute_transition(next_state)
```

### **计算损失**

```python
loss_fn = CostRegularizedLoss(alpha=1.0, beta=0.3, gamma=0.2, delta=0.1)
total_loss, loss_dict = loss_fn(
    policy_loss, transition_loss, listener_loss, 
    torch.tensor([episode_cost])
)
```

---

## 🔧 常用命令

### **查看LLM价格**

```bash
python -c "from neural_fsm_mas import print_pricing_table; print_pricing_table()"
```

### **测试模块**

```bash
# 测试FSM验证器
python -m neural_fsm_mas.agent_topology.fsm_validator

# 测试成本追踪
python -m neural_fsm_mas.utils.llm_cost_tracker

# 测试损失函数
python -m neural_fsm_mas.losses.cost_loss
```

### **运行实验**

```bash
# GSM8K
python run_experiment_random_sampled_fsm.py --dataset gsm8k --epochs 50

# MMLU
python run_experiment_random_sampled_fsm.py --dataset mmlu --epochs 50

# HumanEval
python run_experiment_random_sampled_fsm.py --dataset humaneval --epochs 50
```

---

## 📊 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `alpha` | 1.0 | 策略梯度损失权重 |
| `beta` | 0.3 | 状态转移损失权重 |
| `gamma` | 0.2 | 监听路径损失权重 |
| `delta` | 0.1 | **成本损失权重** ✨ |
| `max_visits_per_state` | 3 | 每状态最大访问次数 |
| `max_total_steps` | 20 | 总步数限制 |
| `batch_size` | 32 | 批大小 |

---

## 🐛 调试技巧

### **检查FSM有效性**

```python
from neural_fsm_mas import validate_fsm
results = validate_fsm(fsm, auto_fix=False, verbose=True)
print(results['errors'])
print(results['warnings'])
```

### **查看执行轨迹**

```python
executor.print_execution_trace()
```

### **分析成本**

```python
tracker.print_statistics()
tracker.get_cost_by_agent()
tracker.get_cost_by_state()
```

---

## 📚 文档快速链接

- **设计**: `RANDOM_SAMPLED_FSM_DESIGN.md`
- **集成指南**: `FINAL_INTEGRATION_GUIDE.md`
- **完成报告**: `ALL_TASKS_COMPLETE_SUMMARY.md`
- **实验脚本**: `run_experiment_random_sampled_fsm.py`

---

## ✅ 验证安装

```bash
python -c "from neural_fsm_mas import FSMValidator, LLMCostTracker; print('✅')"
```

---

## 🎯 典型工作流

```python
# 1. 生成FSM
fsm = generate_fsm(dataset='gsm8k')

# 2. 验证FSM
validate_fsm(fsm, auto_fix=True)

# 3. 创建组件
executor = create_fsm_executor(fsm, max_visits_per_state=3)
tracker = create_cost_tracker('gpt-4o-mini')
loss_fn = CostRegularizedLoss(alpha=1.0, beta=0.3, gamma=0.2, delta=0.1)

# 4. 训练循环
for epoch in range(epochs):
    for problem in dataset:
        tracker.start_episode()
        # ... 执行FSM ...
        episode_cost = tracker.end_episode()
        total_loss, _ = loss_fn(pg_loss, trans_loss, list_loss, torch.tensor([episode_cost]))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

**快速开始实验！** 🚀

