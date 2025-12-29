# NeuralFSM: 基于时序图网络的自适应有限状态机多智能体系统
# NeuralFSM: Adaptive Finite State Machine Multi-Agent System via Temporal Graph Networks

**论文文档 - 第二部分：有限状态机定义与内部机制**

---

## 2. 有限状态机定义与内部机制

### 2.1 FSM的设计理念

有限状态机（FSM）是NeuralFSM系统的核心组织架构。FSM将复杂的问题求解过程分解为一系列状态，每个状态对应一个特定的子任务，由专门的智能体负责执行。这种设计使得系统能够清晰地组织问题求解流程，同时保持灵活性。

FSM的设计遵循几个核心原则。首先，状态与智能体的一对一映射原则。每个状态由唯一的智能体负责执行，这确保了职责清晰，避免了混乱。其次，状态完成条件原则。每个状态定义明确的完成条件，用于判断是否可以转移到下一状态。完成条件可以是结构化的标记、关键词匹配或LLM判断。第三，监听关系原则。定义哪些智能体监听哪些状态的输出，这决定了信息的传递路径。第四，转移规则原则。定义状态转移的条件和优先级，这决定了状态转移的逻辑。

### 2.2 有限状态机的形式化定义

在NeuralFSM系统中，有限状态机（FSM）被形式化定义为六元组：

```
FSM = (S, A, δ, L, C, s₀)
```

其中：
- **S = {s₁, s₂, ..., sₙ}**：有限状态集合，每个状态代表问题求解的一个阶段
- **A = {a₁, a₂, ..., aₘ}**：智能体集合，每个智能体负责执行特定类型的任务
- **δ: S × Context → S**：状态转移函数，根据当前状态和上下文特征，决定下一状态
- **L: S → P(A)**：监听关系函数，将状态映射到监听该状态输出的智能体集合
- **C: S → Condition**：完成条件函数，为每个状态定义完成条件
- **s₀ ∈ S**：初始状态

系统的核心约束是每个状态由一个智能体负责执行。这个约束确保了每个状态都有明确的执行者，避免了责任不清的问题。形式化表示为：

```
∀s ∈ S, ∃!a ∈ A: responsible(s) = a
```

其中 `responsible: S → A` 是状态到智能体的映射函数。

### 2.3 状态转移机制

状态转移函数δ根据当前状态和上下文特征，决定下一状态。状态转移函数的形式化定义为：

```
δ(sᵢ, c) = sⱼ
```

其中：
- `sᵢ ∈ S`：当前状态
- `c ∈ Context`：上下文特征向量，包含问题描述、历史状态信息、智能体输出等
- `sⱼ ∈ S`：下一状态

在实际实现中，状态转移不是确定性的，而是由T。**关键创新在于，TGN接收问题嵌入作为输入，实现任务自适应的状态转移预测：**

```
P(sⱼ | sᵢ, c, q) = TGN_transition(sᵢ, c, q_emb)
```

其中：
- `TGN_transition` 是TGN的状态转移预测模块
- `q_emb ∈ ℝᵈ` 是问题文本的嵌入向量（通过Sentence Transformer编码得到，d=384维）
- 输出一个概率分布 `P(·|sᵢ, c, q_emb) ∈ Δ|S|`，其中 `Δ|S|` 是 |S| 维概率单纯形

给定当前状态 `sᵢ`、上下文特征 `c` 和问题嵌入 `q_emb`，TGN计算转移到每个可能状态的概率：

```
P(sⱼ | sᵢ, c, q_emb) = softmax(TGN_transition(sᵢ, c, q_emb))ⱼ
```

其中 `TGN_transition(sᵢ, c, q_emb)` 输出一个 |S| 维的logits向量，经过softmax归一化后得到概率分布。**问题嵌入的引入使得TGN能够根据每个具体问题的特性，预测最适合该问题的状态转移序列，实现了真正的任务自适应。**

系统支持多种转移策略：

**策略1：贪婪选择（Greedy Selection）**
```
s_{t+1} = argmax_{sⱼ ∈ S} P(sⱼ | sᵢ, c)
```
选择概率最高的状态作为下一状态。

**策略2：概率采样（Probabilistic Sampling）** ✨ **推荐策略（任务自适应）**
```
s_{t+1} ~ P(· | sᵢ, c, q_emb)
```
根据概率分布采样下一状态，增加探索性。**这是系统的默认策略，通过概率采样实现任务自适应的状态转移。采样得到的状态ID用于从总FSM中匹配对应的状态和转移条件。**

**策略3：ε-贪婪（ε-Greedy）**
```
s_{t+1} = {
    argmax_{sⱼ ∈ S} P(sⱼ | sᵢ, c)  with probability 1-ε
    random sample from S            with probability ε
}
```
以概率 ε 进行随机探索，以概率 1-ε 选择最优状态。

**策略4：条件匹配（Condition Matching）**
```
s_{t+1} = first sⱼ ∈ S such that check_condition(sⱼ, c) = True
```
根据转移规则中定义的条件进行匹配，选择第一个满足条件的转移。

### 2.4 监听关系机制

监听关系函数L定义了状态输出到智能体的信息传递。监听关系函数的形式化定义为：

```
L(sᵢ) = {aⱼ₁, aⱼ₂, ..., aⱼₖ} ⊆ A
```

表示状态 `sᵢ` 的输出会被智能体 `aⱼ₁, aⱼ₂, ..., aⱼₖ` 监听（接收）。

在实际实现中，监听关系由TGN预测的权重矩阵表示。**关键创新在于，TGN接收问题嵌入作为输入，实现任务自适应的监听关系预测：**

```
L(sᵢ, aⱼ, q_emb) = w(sᵢ, aⱼ, q_emb) ∈ [0, 1]
```

其中：
- `w(sᵢ, aⱼ, q_emb)` 是状态 `sᵢ` 向智能体 `aⱼ` 发送消息的权重（基于问题嵌入 `q_emb` 计算）
- 权重越高，表示该监听关系越重要

**系统使用概率采样选择监听关系：**

```
sampled_listeners ~ P(· | sᵢ, q_emb)
```

其中 `P(· | sᵢ, q_emb)` 是基于监听权重 `w(sᵢ, ·, q_emb)` 的概率分布。采样得到的智能体索引用于从总FSM中匹配对应的智能体。

当状态sᵢ完成时，其输出消息会被传递给所有采样得到的监听智能体。**系统不再依赖固定阈值，而是完全基于学习到的概率分布进行采样，实现了任务自适应的通信路径选择。**如果一个智能体监听多个状态，它接收到的消息会被聚合，聚合策略可以是加权平均、拼接或注意力机制。

### 2.5 完成条件机制

完成条件函数C为每个状态定义了完成判断标准。完成条件可以是自然语言描述的条件，例如"状态包含最终答案"、"所有子问题已解决"、"达到最大迭代次数"等。系统在执行过程中检查这些条件，判断状态是否完成。

完成条件可以是显式标记、答案格式、逻辑判断或固定步数。显式标记检查输出中是否包含特定标记（如[COMPLETE]或[DONE]）；答案格式检查输出是否符合最终答案的格式要求；逻辑判断通过LLM判断输出是否完整；固定步数检查是否达到该状态的最大执行步数。

为了防止系统陷入无限循环，系统实现了循环避免机制。系统为每个状态设置最大访问次数，当状态的访问次数超过阈值时，系统禁止再次访问该状态。同时，系统设置全局最大步数，当执行步数超过阈值时，系统强制终止执行。

#### 4.2.3 转移条件匹配

每个转移可以定义条件，系统在执行转移前检查条件是否满足：

```
check_condition(sᵢ → sⱼ, c) = {
    True   if condition(sᵢ → sⱼ) matches c
    False  otherwise
}
```

条件可以是：
- **结构化标记**：检查输出中是否包含特定标记（如 `[FINAL_ANSWER]`）
- **关键词匹配**：检查输出中是否包含特定关键词
- **失败条件**：检查是否满足失败条件（如 `[ERROR]` 或 `[FAILED]`）

### 4.3 状态完成机制

#### 4.3.1 完成条件检查

系统在每个状态执行后检查完成条件：

```
is_complete(sᵢ, output) = check_completion_condition(C(sᵢ), output)
```

其中 `output` 是当前状态智能体的输出。

#### 4.3.2 完成条件类型

完成条件可以是：

1. **显式标记**：输出中包含 `[COMPLETE]` 或 `[DONE]` 标记
2. **答案格式**：输出符合最终答案的格式要求
3. **逻辑判断**：通过LLM判断输出是否完整
4. **固定步数**：达到该状态的最大执行步数

#### 4.3.3 循环避免机制

为了防止系统陷入无限循环，系统实现了循环避免机制：

```
visit_count(sᵢ) = number of times sᵢ has been visited
max_visits = 3  # 每个状态最多访问3次

if visit_count(sᵢ) >= max_visits:
    force_transition(sᵢ)  # 强制转移到其他状态
```

### 4.4 监听关系机制

#### 4.4.1 监听权重计算

给定状态 `sᵢ` 和智能体 `aⱼ`，TGN计算监听权重：

```
w(sᵢ, aⱼ) = TGN_listener(sᵢ, aⱼ, c)
```

其中 `TGN_listener` 是TGN的监听关系预测模块，输出一个标量权重值。

#### 4.4.2 消息传递

当状态 `sᵢ` 完成时，其输出消息会被传递给所有监听该状态的智能体：

```
message(sᵢ → aⱼ) = {
    output(sᵢ) × w(sᵢ, aⱼ)  if w(sᵢ, aⱼ) > threshold
    None                      otherwise
}
```

其中 `threshold` 是一个阈值（例如0.1），只有权重超过阈值的监听关系才会传递消息。

#### 4.4.3 消息聚合

如果一个智能体 `aⱼ` 监听多个状态，它接收到的消息会被聚合：

```
aggregated_message(aⱼ) = aggregate({message(sᵢ → aⱼ) : sᵢ ∈ S, w(sᵢ, aⱼ) > threshold})
```

聚合策略可以是：
- **加权平均**：`Σᵢ w(sᵢ, aⱼ) × message(sᵢ → aⱼ) / Σᵢ w(sᵢ, aⱼ)`
- **拼接**：将所有消息拼接在一起
- **注意力机制**：使用注意力权重聚合消息

### 4.5 FSM执行流程

#### 4.5.1 执行算法

FSM的执行流程可以用以下算法描述：

```
Algorithm: FSM Execution
Input: question q, FSM (S, A, δ, L, C, s₀)
Output: final answer

1. Initialize:
   current_state = s₀
   context = {question: q}
   visit_count = {s: 0 for s in S}
   max_total_steps = 20

2. For step = 1 to max_total_steps:
   a. Check if current_state is complete:
      if is_complete(current_state, context):
          if current_state is final_state:
              return extract_answer(context)
          else:
              continue
   
   b. Check visit limit:
      if visit_count[current_state] >= max_visits:
          force_transition(current_state)
   
   c. Execute agent at current_state:
      agent = responsible(current_state)
      output = agent.execute(context)
      context.update({current_state: output})
   
   d. Check completion condition:
      if check_completion_condition(C(current_state), output):
          mark_complete(current_state)
   
   e. Predict next state:
      transition_probs = TGN_transition(current_state, context)
      next_state = select_strategy(transition_probs)
   
   f. Update state:
      visit_count[current_state] += 1
      current_state = next_state
   
   g. Propagate messages:
      for each agent a in L(current_state):
          message = output × w(current_state, a)
          context.update({a: message})

3. Return timeout_answer(context)
```

#### 4.5.2 执行示例

考虑一个简单的数学问题求解FSM：

**状态定义**：
- `s₁`: 问题分析（Agent: Analyzer）
- `s₂`: 数学建模（Agent: Modeler）
- `s₃`: 计算求解（Agent: Solver）
- `s₄`: 答案验证（Agent: Verifier）

**执行流程**：
```
Step 1: s₁ (Analyzer)
  - Input: "Solve 2x + 3 = 7"
  - Output: "Linear equation, variable x, constant terms: 2x and 3"
  - Complete: Yes (contains analysis)
  - Next: s₂ (probability: 0.9)

Step 2: s₂ (Modeler)
  - Input: Previous analysis + question
  - Output: "Equation: 2x + 3 = 7, isolate x: 2x = 4"
  - Complete: Yes (contains model)
  - Next: s₃ (probability: 0.95)

Step 3: s₃ (Solver)
  - Input: Previous model + question
  - Output: "x = 2"
  - Complete: Yes (contains answer)
  - Next: s₄ (probability: 0.8)

Step 4: s₄ (Verifier)
  - Input: Previous answer + question
  - Output: "Verification: 2(2) + 3 = 7 ✓"
  - Complete: Yes (contains verification)
  - Final: Return "x = 2"
```

### 2.6 FSM执行流程

FSM的执行是一个动态的过程，系统根据当前状态、完成条件和TGN的预测，动态决定状态转移。执行流程如下：首先，系统初始化FSM，设置初始状态。初始状态通常是问题理解或任务分析状态，负责理解任务需求。系统重置所有状态，清除历史记录，设置访问计数为0。

然后，系统进入执行循环。在每次循环中，系统首先获取当前状态，检查当前状态的访问次数，如果超过最大访问次数，系统终止执行，防止无限循环。接着，系统获取当前状态的负责智能体，智能体根据当前状态和任务输入，执行相应的子任务。在执行过程中，提示词优化框架可能动态优化智能体的提示词，提升推理质量。智能体执行完成后，输出结果和完成标记。

然后，系统检查完成条件，判断当前状态是否完成。如果满足，系统认为当前状态已完成，可以转移到下一状态。如果不满足，系统可能继续在当前状态执行，或根据策略选择其他操作。接着，系统使用TGN预测下一状态。TGN根据当前状态的嵌入、上下文特征和历史信息，预测所有可能下一状态的概率分布。系统根据概率分布和转移规则，选择下一状态。

然后，系统执行状态转移，更新FSM状态管理器，将当前状态设置为选定的下一状态，记录转移历史，增加访问计数。系统更新监听关系，通知监听新状态的智能体，传递相关信息。最后，系统检查终止条件，包括到达最终状态、任务完成、达到最大步数等。如果满足终止条件，系统终止执行，返回最终结果。否则，系统继续执行循环。

### 2.7 FSM的自动生成与任务自适应匹配

FSM的生成是一个自动化的过程，由LLM驱动的生成器完成。生成器根据数据集的特性和任务需求，自动生成**总FSM**的结构和规则。总FSM包含所有可能的状态、转移规则（含转移条件）、智能体和监听关系。生成过程分为四个步骤：第一步是智能体生成，生成器分析数据集的特点，生成适合的智能体角色描述；第二步是状态生成，生成器根据智能体角色，生成对应的状态描述；第三步是监听关系生成，生成器分析任务需求，确定哪些智能体需要哪些状态的输出；第四步是转移规则生成，生成器生成状态转移的条件和优先级。

对于MMLU数据集，由于包含57个不同的学科类别，系统为每个类别生成独立的总FSM。每个类别的总FSM针对该学科的特点进行优化，例如数学类强调计算和证明步骤，历史类强调时间线和因果关系，科学类强调实验和观察。

系统实现了FSM缓存机制，避免重复生成。系统首先检查是否已经为该类别生成了总FSM，如果已经生成，系统从缓存中加载总FSM；如果没有生成，系统使用LLM驱动的生成器为该类别生成总FSM，并保存到缓存中。这大大减少了LLM调用成本，提升了系统效率。

**任务自适应的FSM匹配机制：**

训练完成后，对于每个新问题，系统执行以下步骤：

1. **问题嵌入生成**：使用Sentence Transformer将问题文本编码为向量 `q_emb ∈ ℝ³⁸⁴`

2. **概率采样**：
   - 状态转移采样：`s_{t+1} ~ P(· | s_t, c, q_emb)`，从总FSM中匹配对应的状态和转移条件
   - 监听关系采样：`listeners ~ P(· | s_t, q_emb)`，从总FSM中匹配对应的智能体

3. **动态MAS构建**：
   - 根据采样结果，从总FSM中提取对应的状态描述、转移条件描述和智能体描述
   - 构建针对该问题的定制化MAS结构

这种设计使得系统能够为每个问题生成最适合的FSM子结构，实现了真正的任务自适应。

---

**第二部分结束**

*下一部分将详细阐述时序图网络（TGN）的运行机制和数学公式。*


