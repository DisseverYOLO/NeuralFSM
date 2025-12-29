# Random-Sampled FSM vs Fixed FSM Design
# 随机采样FSM vs 固定FSM设计文档

## 📋 核心设计理念对比

### **MetaAgent项目 (固定FSM)**

```
设计时: LLM生成完整FSM
    ↓
执行时: 严格按照预定义的转移条件执行
    ↓
问题: FSM是固定的,无法优化和剪枝
```

### **本项目 (随机采样FSM + TGN优化)**

```
设计时: LLM生成状态和所有可能的转移
    ↓
采样时: 随机采样状态连接,生成多种FSM变体
    ↓
执行时: 每次采样新连接后,更新转移条件
    ↓
优化时: TGN学习最优路径,剔除低效状态
```

---

## 🔑 关键问题解答

### **Q1: 状态完成判断由谁负责？**

**答案**: **执行该状态的Agent自己判断** ✅

**理由**:
1. **效率**: 避免额外LLM调用
2. **上下文**: 执行Agent最了解任务完成情况
3. **结构化输出**: Agent按指令输出完成标记

**实现方式**:
```python
# Agent收到的指令包含完成要求
instruction = """
Analyze the problem and extract key variables.
When completed, output: <ANALYSIS_COMPLETE>
"""

# Agent执行并自行判断是否完成
output = agent.run(instruction, context)

# 系统检查完成标记
if "<ANALYSIS_COMPLETE>" in output:
    state_completed = True
else:
    state_completed = False
```

---

### **Q2: 如何确保所有状态能连接到最终提交状态？**

**答案**: **FSM生成后执行可达性检查** ✅

**实现步骤**:
1. 生成FSM后,构建状态转移图
2. 从初始状态开始BFS/DFS遍历
3. 检查是否能到达至少一个最终状态
4. 如果不可达,自动添加"救援转移"

**代码示例**:
```python
def check_reachability(fsm: Dict) -> bool:
    """检查FSM可达性"""
    # 构建邻接表
    graph = {state['state_id']: [] for state in fsm['states']}
    for trans in fsm['transitions']:
        graph[trans['from_state']].append(trans['to_state'])
    
    # 找到初始状态和最终状态
    initial_states = [s['state_id'] for s in fsm['states'] if s['is_initial']]
    final_states = [s['state_id'] for s in fsm['states'] if s['is_final']]
    
    # BFS从初始状态开始
    visited = set()
    queue = initial_states.copy()
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        if current in final_states:
            return True  # 可达！
        
        for next_state in graph.get(current, []):
            if next_state not in visited:
                queue.append(next_state)
    
    return False  # 不可达！

def add_rescue_transitions(fsm: Dict):
    """如果不可达,添加救援转移"""
    if not check_reachability(fsm):
        # 找到离最终状态最近的状态
        final_states = [s for s in fsm['states'] if s['is_final']]
        non_final_states = [s for s in fsm['states'] if not s['is_final']]
        
        # 为每个非最终状态添加到最终状态的转移
        for state in non_final_states:
            fsm['transitions'].append({
                "from_state": state['state_id'],
                "to_state": final_states[0]['state_id'],
                "condition": f"If all attempts failed or max steps reached, finalize answer from state {state['state_name']}",
                "priority": 99  # 最低优先级
            })
```

---

### **Q3: 如何支持状态回退并避免无限循环？**

**答案**: **允许双向转移 + 最大访问次数限制** ✅

**设计原则**:
1. ✅ 允许状态A→B和B→A的双向转移
2. ✅ 记录每个状态的访问次数
3. ✅ 当状态访问次数超过阈值时,强制转移到下一阶段

**实现方式**:
```python
class FSMExecutor:
    def __init__(self, fsm: Dict, max_visits_per_state: int = 3):
        self.fsm = fsm
        self.max_visits = max_visits_per_state
        self.visit_counts = {s['state_id']: 0 for s in fsm['states']}
        self.current_state_id = None
    
    def can_transition_to(self, from_state: str, to_state: str) -> bool:
        """检查是否可以转移"""
        # 检查是否会导致无限循环
        if self.visit_counts[to_state] >= self.max_visits:
            print(f"⚠️  State {to_state} visited {self.visit_counts[to_state]} times, skipping")
            return False
        
        return True
    
    def execute_transition(self, to_state: str):
        """执行状态转移"""
        self.visit_counts[to_state] += 1
        self.current_state_id = to_state
        print(f"✅ Transition to {to_state} (visit #{self.visit_counts[to_state]})")
```

---

### **Q4: 如何为采样的状态连接生成转移条件？**

**答案**: **每次采样新连接后,LLM生成该转移的条件** ✅

**流程**:
```
1. 采样: 决定State A → State B
   ↓
2. 查询: State A和State B的描述是什么？
   ↓
3. LLM生成: 从A转移到B的条件
   ↓
4. 记录: 将条件写入转移规则
```

**代码示例**:
```python
def generate_transition_condition(state_from: Dict, state_to: Dict, llm: LLM) -> str:
    """为新采样的转移生成条件"""
    prompt = f"""
    Given two states in a problem-solving FSM:
    
    FROM State: {state_from['state_name']}
    Description: {state_from['instruction']}
    
    TO State: {state_to['state_name']}
    Description: {state_to['instruction']}
    
    Generate a clear, testable transition condition that determines when the system 
    should transition from the FROM state to the TO state.
    
    The condition should:
    1. Be based on observable outputs (e.g., "output contains <TAG>")
    2. Be specific and actionable
    3. Handle both success and failure cases
    
    Output ONLY the condition, no explanation:
    """
    
    condition = llm.chat(prompt).strip()
    return condition

# 使用示例
sampled_transition = sample_random_transition(fsm)  # A → B
condition = generate_transition_condition(
    state_from=get_state(sampled_transition['from']),
    state_to=get_state(sampled_transition['to']),
    llm=llm
)

# 更新转移规则
sampled_transition['condition'] = condition
fsm['transitions'].append(sampled_transition)
```

---

### **Q5: 最终状态是否需要专门的提交Agent？**

**答案**: **不需要,复用已有Agent** ✅

**理由**:
1. 节省Agent数量（降低复杂度）
2. 已有Agent可以在最终状态完成提交
3. 只需在instruction中指定提交格式

**实现方式**:
```json
{
  "state_id": "final",
  "state_name": "Final_Answer_Submission",
  "agent_id": "3",  // 复用Agent 3（例如验证者）
  "instruction": "Review the solution and submit the final answer using <|submit|> <answer> format.",
  "is_initial": false,
  "is_final": true,
  "listeners": []
}
```

**对比**:
```
❌ 原方案: 创建专门的"Submitter" Agent
   - Agent数量+1
   - 增加系统复杂度
   
✅ 新方案: 复用已有Agent（如验证者、协调者）
   - Agent数量不变
   - 最后一个状态由现有Agent执行提交
```

---

## 🎯 LLM调用成本优化

### **核心思想**

```
传统损失 = 任务准确率损失
          
新损失 = 任务准确率损失 + λ_cost × LLM调用成本损失
```

### **成本计算**

```python
# 单次调用成本
cost_per_call = (input_tokens × price_input + output_tokens × price_output) / 1000

# 一个episode的总成本
total_cost = Σ(所有Agent调用的成本)

# 成本损失（归一化）
L_cost = total_cost / baseline_cost  # baseline_cost为参考成本
```

### **LLM API价格清单**

| 模型 | Input ($/1K tokens) | Output ($/1K tokens) |
|------|---------------------|----------------------|
| gpt-4o-mini | $0.00015 | $0.0006 |
| gpt-4o | $0.0025 | $0.010 |
| gpt-4-turbo | $0.010 | $0.030 |
| gpt-3.5-turbo | $0.0005 | $0.0015 |

---

## 📊 四目标组合损失函数

### **数学公式**

```
L_total = α·L_policy + β·L_trans + γ·L_listener + δ·L_cost

其中:
- L_policy: 策略梯度损失（任务准确率）
- L_trans: 状态转移损失（学习最优转移序列）
- L_listener: 监听路径损失（学习最优通信路径）
- L_cost: LLM调用成本损失（鼓励减少调用次数）✨ 新增

- α = 1.0 (主要目标: 准确率)
- β = 0.3 (次要目标: 状态序列优化)
- γ = 0.2 (次要目标: 通信优化)
- δ = 0.1 (约束目标: 成本控制) ✨ 新增
```

### **成本损失的作用**

```
δ = 0: 不考虑成本,可能产生冗余状态和调用

δ = 0.1: 轻微约束,鼓励减少无效调用
         - TGN学习跳过低价值状态
         - 减少重复访问同一状态

δ = 0.5: 强约束,积极减少调用
         - 可能影响准确率
         - 适合资源受限场景
```

---

## 🔄 随机采样FSM执行流程

### **完整流程图**

```
┌────────────────────────────────────────────────┐
│ Phase 1: FSM生成（LLM一次性生成）              │
└────────────────────────────────────────────────┘
  ↓
  生成所有状态和所有可能的转移
  ↓
┌────────────────────────────────────────────────┐
│ Phase 2: 随机采样（每个episode）               │
└────────────────────────────────────────────────┘
  ↓
  从所有可能的转移中随机采样一部分
  ↓
  为新采样的转移生成具体条件（LLM调用）
  ↓
┌────────────────────────────────────────────────┐
│ Phase 3: 执行（按采样的FSM）                   │
└────────────────────────────────────────────────┘
  ↓
  Step 1: 获取当前状态
  ↓
  Step 2: 执行Agent（记录调用成本）
  ↓
  Step 3: Agent自己判断是否完成
  ↓
  if 完成:
      Step 4: 检查转移条件,选择下一状态
             - 匹配转移条件
             - 检查访问次数限制
             - TGN预测最优转移
      Step 5: 消息传递（按监听矩阵）
      Step 6: 执行转移
  else:
      继续当前状态或标记错误
  ↓
  Step 7: 检查是否到达最终状态
  ↓
┌────────────────────────────────────────────────┐
│ Phase 4: TGN优化（多个episodes后）             │
└────────────────────────────────────────────────┘
  ↓
  学习最优状态转移概率
  学习最优监听路径
  学习最小成本路径 ✨ 新增
  ↓
  剔除低效状态（转移概率低 + 成本高）
```

---

## 📐 状态转移条件的生成和匹配

### **生成时（采样阶段）**

```python
# 为每个采样的转移生成条件
for transition in sampled_transitions:
    if 'condition' not in transition:
        # 动态生成条件
        condition = generate_transition_condition(
            state_from=get_state(transition['from_state']),
            state_to=get_state(transition['to_state']),
            llm=llm
        )
        transition['condition'] = condition
        transition['priority'] = assign_priority(transition)
```

### **执行时（状态转移判断）**

```python
def select_next_state(current_state: str, agent_output: str, 
                     fsm: Dict, tgn: FSMTemporalGraph) -> str:
    """根据转移条件和TGN预测选择下一状态"""
    
    # 1. 找到所有可能的转移
    possible_transitions = [
        t for t in fsm['transitions'] 
        if t['from_state'] == current_state
    ]
    
    # 2. 检查转移条件
    valid_transitions = []
    for trans in possible_transitions:
        if check_condition(trans['condition'], agent_output):
            valid_transitions.append(trans)
    
    # 3. 如果有多个有效转移,使用TGN预测
    if len(valid_transitions) > 1:
        state_probs = tgn.predict_next_state(current_state_features)
        # 选择TGN预测概率最高的
        next_state = select_by_tgn_prob(valid_transitions, state_probs)
    elif len(valid_transitions) == 1:
        next_state = valid_transitions[0]['to_state']
    else:
        # 没有有效转移,使用默认策略
        next_state = default_transition(current_state, fsm)
    
    return next_state

def check_condition(condition: str, agent_output: str) -> bool:
    """检查转移条件是否满足"""
    # 解析条件并检查
    if "contains" in condition.lower():
        # 提取标记
        marker = extract_marker(condition)
        return marker in agent_output
    
    elif "field" in condition.lower():
        # 检查字段存在
        return check_field_existence(condition, agent_output)
    
    elif "success" in condition.lower():
        # 检查成功标记
        return check_success_marker(agent_output)
    
    # ... 更多条件类型
```

---

## 🆚 与MetaAgent的关键区别

| 维度 | MetaAgent (固定FSM) | 本项目 (随机采样FSM) |
|------|---------------------|---------------------|
| **FSM生成** | 一次性固定 | 生成所有可能转移 |
| **执行方式** | 严格按预定义路径 | 每次采样不同路径 |
| **转移条件** | 生成时固定 | 采样时动态生成 |
| **优化能力** | ❌ 无法优化 | ✅ TGN学习最优路径 |
| **状态剪枝** | ❌ 无法剪枝 | ✅ 自动剔除低效状态 |
| **回退机制** | 固定的回退路径 | 动态学习回退策略 |
| **成本优化** | ❌ 不考虑 | ✅ 成本纳入损失函数 |
| **最终状态** | 专门Submitter | 复用已有Agent |

---

## 🎓 核心创新点

1. **随机采样FSM**: 探索多种状态序列,而非固定一条
2. **动态转移条件**: 根据采样的连接生成具体条件
3. **TGN端到端优化**: 同时优化转移、通信、成本
4. **自动状态剪枝**: 基于转移概率和成本剔除冗余
5. **成本约束学习**: 将LLM调用成本纳入优化目标

---

## 📝 实现checklist

- [ ] 1. 创建LLM API价格清单模块
- [ ] 2. 实现成本追踪和计算
- [ ] 3. 设计成本损失函数
- [ ] 4. 更新四目标组合损失
- [ ] 5. 实现状态可达性检查
- [ ] 6. 实现状态访问次数限制
- [ ] 7. 实现动态转移条件生成
- [ ] 8. 修改最终状态复用Agent
- [ ] 9. 更新FSM生成器（支持随机采样）
- [ ] 10. 更新文档和流程图

---

**这套设计使得FSM不再是固定的,而是可学习、可优化的,同时保证了实用性（成本约束）和可靠性（可达性保证）！**

