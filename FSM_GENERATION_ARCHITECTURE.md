# FSM生成架构说明
# FSM Generation Architecture

## 📋 核心组件关系

### 1. `Enhanced_FSM_Gen.py` - **主要FSM生成器** ✨

**作用**: 使用LLM动态生成完整的FSM系统

**生成内容**:
- ✅ **智能体描述** (agents): 包含详细的`system_prompt`、`role`、`state_completion_criteria`
- ✅ **FSM状态** (states): 包含`state_id`、`state_name`、`agent_id`、`instruction`、`completion_condition`
- ✅ **状态转移规则** (transitions): 包含`from_state`、`to_state`、`condition`、`priority`
- ✅ **监听关系** (listeners): 定义哪些智能体接收某个状态的输出

**使用场景**:
- 实验开始时生成FSM（如果缓存不存在）
- 为MMLU的57个类别分别生成FSM
- 生成的数据保存到`fsm_cache`中

**输出格式**:
```json
{
  "dataset": "gsm8k",
  "agents": [
    {
      "agent_id": "0",
      "name": "Problem Analyzer",
      "role": "Analyze word problems",
      "system_prompt": "You are a problem analysis expert...",
      "state_completion_criteria": "Output contains <ANALYSIS_DONE> tag"
    }
  ],
  "fsm": {
    "states": [
      {
        "state_id": "0",
        "state_name": "Problem_Analysis",
        "agent_id": "0",
        "instruction": "Analyze the problem...",
        "completion_condition": "Output contains <ANALYSIS_DONE>",
        "is_initial": true,
        "is_final": false,
        "listeners": ["1", "2"]
      }
    ],
    "transitions": [
      {
        "from_state": "0",
        "to_state": "1",
        "condition": "State 0 completion_condition is met",
        "priority": 1
      }
    ]
  }
}
```

---

### 2. `prompt_manager.py` - **预定义模板库** 📚

**作用**: 提供静态的智能体角色模板和连接关系

**提供内容**:
- ✅ **预定义角色名称**: 如"Knowledge Expert", "Problem Analyzer"等
- ✅ **角色描述**: 每个角色的简要描述
- ✅ **角色连接关系**: 定义哪些角色可以通信
- ✅ **回答提示模板**: 为每个角色提供标准化的提示格式

**使用场景**:
- **Fallback机制**: 当FSM缓存不存在且不自动生成时使用
- **快速原型**: 用于快速测试和开发
- **参考模板**: 为`Enhanced_FSM_Gen.py`提供设计参考

**限制**:
- ❌ **不包含**详细的FSM状态定义
- ❌ **不包含**状态转移规则
- ❌ **不包含**完成条件
- ❌ **不包含**监听关系

---

## 🔄 集成流程

### 正确的使用流程

```
1. 实验启动 (run_experiment_1_fsm_complete.py)
   │
   ├─> 检查FSM缓存 (fsm_cache_manager)
   │   │
   │   ├─> 缓存存在 → 加载缓存的FSM配置
   │   │              (包含agents、states、transitions)
   │   │
   │   └─> 缓存不存在 → 使用Enhanced_FSM_Gen.py生成
   │                    → 保存到缓存
   │
   └─> 训练开始 (train_fsm_mas_v2.py)
       │
       ├─> create_domain_fsm_system()
       │   │
       │   ├─> 优先: 从缓存加载FSM配置
       │   │         → 使用缓存的agents和states
       │   │
       │   └─> Fallback: 使用prompt_manager.py的预定义角色
       │                 → 生成简单FSM
       │
       └─> 训练循环
           │
           └─> 使用FSM执行推理
               → 使用缓存的completion_condition判断状态完成
               → 使用缓存的transitions进行状态转移
```

---

## ⚠️ 当前问题

### 问题1: `_create_fsm_from_cache`未正确恢复FSM状态

**当前代码**:
```python
def _create_fsm_from_cache(self, cached: Dict, domain: str):
    # ...
    # TODO: 实现从fsm_config恢复FSM状态
    # 目前先使用默认生成方法
    self._generate_fsm_states(fsm_manager, agent_roles, agent_ids, domain)
```

**问题**: 没有从缓存的`fsm_config`中恢复状态，而是重新生成了简单状态

**修复**: 需要实现从`fsm_config`恢复完整FSM状态

---

### 问题2: `_get_agent_roles_for_domain`优先使用prompt_manager

**当前代码**:
```python
def _get_agent_roles_for_domain(self, domain: str):
    # 优先使用预定义的领域提示
    prompt_set = DomainPromptManager.get_manager(domain)
    agent_roles = prompt_set.get_available_roles()
```

**问题**: 应该优先使用缓存的FSM中的智能体，而不是prompt_manager

**修复**: 先检查缓存，再fallback到prompt_manager

---

## ✅ 修复方案

### 修复1: 实现从缓存恢复FSM状态

```python
def _create_fsm_from_cache(self, cached: Dict, domain: str):
    fsm_config = cached['fsm_config']
    agents = cached['agents']
    
    # 从缓存的FSM配置恢复状态
    for state in fsm_config.get('states', []):
        fsm_manager.add_state(
            state_id=int(state['state_id']),
            state_name=state['state_name'],
            responsible_agent_id=state['agent_id'],
            is_initial=state.get('is_initial', False),
            is_final=state.get('is_final', False),
            description=state.get('instruction', ''),
            completion_condition=state.get('completion_condition', '')
        )
        
        # 恢复监听关系
        for listener_id in state.get('listeners', []):
            fsm_manager.add_listener(
                state_id=int(state['state_id']),
                listener_agent_id=listener_id
            )
    
    # 恢复转移规则（存储到fsm_manager中）
    fsm_manager.transitions = fsm_config.get('transitions', [])
```

### 修复2: 优先使用缓存的FSM

```python
def _get_agent_roles_for_domain(self, domain: str):
    # 优先使用缓存的FSM中的智能体
    if hasattr(self, 'fsm_cache_manager') and self.fsm_cache_manager.has_cache(domain):
        cached = self.fsm_cache_manager.load_fsm(domain)
        agents = cached.get('agents', [])
        if agents:
            agent_roles = [agent.get('name', f"Agent{i}") for i, agent in enumerate(agents)]
            return agent_roles
    
    # Fallback: 使用prompt_manager
    try:
        prompt_set = DomainPromptManager.get_manager(domain)
        agent_roles = prompt_set.get_available_roles()
        return agent_roles
    except:
        # 最后fallback: 使用默认角色
        return default_roles
```

---

## 📊 总结

| 组件 | 作用 | 优先级 | 使用场景 |
|------|------|--------|----------|
| `Enhanced_FSM_Gen.py` | **主要生成器** | ⭐⭐⭐ | 生成完整FSM系统，保存到缓存 |
| `prompt_manager.py` | **模板库** | ⭐ | Fallback，快速原型，参考模板 |
| `fsm_cache_manager` | **缓存管理** | ⭐⭐⭐ | 存储和加载生成的FSM |

**关键原则**:
1. ✅ **优先使用**`Enhanced_FSM_Gen.py`生成的FSM（从缓存加载）
2. ✅ **Fallback到**`prompt_manager.py`（仅当缓存不存在且不自动生成时）
3. ✅ **确保**训练代码正确恢复缓存的FSM状态和转移规则

---

**文档版本**: V1.0  
**最后更新**: 2024-11-13

