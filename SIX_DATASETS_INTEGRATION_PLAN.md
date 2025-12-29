# 六数据集完整集成方案
# Six Datasets Complete Integration Plan

## 📊 数据集概览

### 现有数据集 (3个)
1. **GSM8K** - Grade School Math 数学题解答
2. **MMLU** - Massive Multitask Language Understanding (57个子类别)
3. **HumanEval** - Python代码生成

### 新增数据集 (3个)
4. **HotpotQA** - 多跳问答推理
5. **ALFWorld** - 具身智能体交互任务
6. **MATH** - 高级数学竞赛题

---

## 🎯 核心设计原则

### FSM生成策略

#### 单数据集模式 (GSM8K, HumanEval, HotpotQA, ALFWorld, MATH)
- ✅ **一开始生成一套FSM** (状态、智能体、转移描述)
- ✅ 所有问题共享同一套FSM
- ✅ TGN训练时采样这套FSM
- ❌ 不为每个问题重新生成FSM

#### MMLU特殊模式 (57个类别)
- ✅ **为每个类别生成一套专属FSM**
  - abstract_algebra → FSM_abstract_algebra
  - anatomy → FSM_anatomy
  - ... (共57套)
- ✅ 训练/测试时根据问题所属类别加载对应FSM
- ✅ 不同类别的FSM可以有不同的状态和智能体
- ✅ TGN学习每个类别特有的状态转移模式

---

## 📋 详细任务清单

### Phase 1: 数据集分析与格式理解 ✅

#### 数据格式分析

**HotpotQA**:
```json
{
  "_id": "5ab430fa5542991751b4d6dd",
  "question": "What kind of company is this group...",
  "answer": "pan-Asian life insurance group",
  "supporting_facts": [[doc_title, sent_id], ...],
  "context": [[doc_title, [sentences]], ...],
  "type": "bridge",  // or "comparison"
  "level": "hard"    // or "medium", "easy"
}
```
**特点**:
- 多跳推理：需要跨文档推理
- 长上下文：多个文档片段
- 支撑事实：明确标注推理路径
- 难度分级：easy/medium/hard

**ALFWorld**:
```json
{
  "task": "alfworld",
  "id": 0,
  "goal": "look at bowl under the desklamp.",
  "subgoals": ["^(?=.* you see)(?=.*a bowl \\d+)", ...],
  "difficulty": "hard",
  "additional_info": {"description": "..."}
}
```
**特点**:
- 具身任务：需要与环境交互
- 多步骤：需要规划动作序列
- 子目标：明确的中间目标（正则表达式匹配）
- 反馈驱动：根据环境反馈调整动作

**MATH**:
```json
{
  "problem": "Convert the point $(0,3)$ in rectangular...",
  "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3....",
  "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
  "subject": "Precalculus",  // 数学子领域
  "level": 2,                // 难度级别 1-5
  "unique_id": "test/precalculus/807.json"
}
```
**特点**:
- 高级数学：竞赛级别数学题
- 详细解答：提供完整解题过程
- 子领域分类：代数、几何、微积分等
- 难度分级：1-5级

---

### Phase 2: 数据集加载器 (Dataset Loaders)

#### Task 2: HotpotQA数据集加载器
**文件**: `D:\NeuralFSM\datasets\hotpotqa_dataset.py`

**功能**:
- 加载hotpotqa.jsonl
- 解析多文档上下文
- 提取支撑事实
- 按难度分层采样

**接口**:
```python
class HotpotQADataset:
    def __init__(self, data_path: str)
    def load_data(self) -> List[Dict]
    def get_train_test_split(self, train_ratio=0.7)
    def filter_by_difficulty(self, level: str)
    def format_context(self, sample: Dict) -> str
```

#### Task 3: ALFWorld数据集加载器
**文件**: `D:\NeuralFSM\datasets\alfworld_dataset.py`

**功能**:
- 加载test.jsonl
- 解析任务目标和子目标
- 支持动作序列验证
- 环境状态管理

**接口**:
```python
class ALFWorldDataset:
    def __init__(self, data_path: str)
    def load_data(self) -> List[Dict]
    def get_train_test_split(self, train_ratio=0.7)
    def parse_subgoals(self, sample: Dict) -> List[str]
    def verify_action_sequence(self, actions: List[str], subgoals: List[str]) -> bool
```

#### Task 4: MATH数据集加载器
**文件**: `D:\NeuralFSM\datasets\math_dataset.py`

**功能**:
- 加载math.jsonl
- 按subject分类
- 按level过滤
- LaTeX公式处理

**接口**:
```python
class MATHDataset:
    def __init__(self, data_path: str)
    def load_data(self) -> List[Dict]
    def get_train_test_split(self, train_ratio=0.7)
    def filter_by_subject(self, subject: str)
    def filter_by_level(self, min_level: int, max_level: int)
    def get_subjects(self) -> List[str]
```

---

### Phase 3: Enhanced_FSM_Gen扩展

#### Task 5: HotpotQA FSM模板
**函数**: `_get_hotpotqa_template()`

**状态设计** (建议6-8个状态):
1. **Question Understanding** - 理解问题和推理需求
2. **Context Retrieval** - 检索相关文档
3. **Multi-hop Reasoning** - 多跳推理连接
4. **Supporting Facts Extraction** - 提取支撑事实
5. **Answer Synthesis** - 综合答案
6. **Consistency Verification** - 验证答案一致性
7. **Final Answer** - 最终答案提交

**智能体设计**:
- Question Analyzer (问题分析)
- Document Retriever (文档检索)
- Reasoning Engine (推理引擎)
- Fact Extractor (事实提取)
- Answer Synthesizer (答案综合)
- Verifier (验证器)

#### Task 6: ALFWorld FSM模板
**函数**: `_get_alfworld_template()`

**状态设计** (建议5-7个状态):
1. **Task Understanding** - 理解任务目标
2. **Environment Exploration** - 环境探索
3. **Action Planning** - 动作规划
4. **Action Execution** - 动作执行
5. **Subgoal Verification** - 子目标验证
6. **Error Recovery** - 错误恢复
7. **Goal Achievement** - 目标达成

**智能体设计**:
- Task Parser (任务解析)
- Explorer (探索者)
- Action Planner (动作规划器)
- Executor (执行器)
- State Monitor (状态监控)
- Recovery Agent (恢复智能体)

#### Task 7: MATH FSM模板
**函数**: `_get_math_template()`

**状态设计** (建议7-9个状态):
1. **Problem Analysis** - 问题分析
2. **Concept Identification** - 概念识别
3. **Strategy Formulation** - 策略制定
4. **Step-by-step Solving** - 分步求解
5. **Calculation Verification** - 计算验证
6. **Alternative Approach** - 备选方法
7. **Solution Review** - 解答审查
8. **Final Answer Formatting** - 最终答案格式化

**智能体设计**:
- Problem Analyzer (问题分析器)
- Math Concept Expert (数学概念专家)
- Strategy Designer (策略设计师)
- Step Solver (分步求解器)
- Calculator (计算器)
- Verifier (验证器)
- Formatter (格式化器)

---

### Phase 4: Domain Prompts扩展

#### Task 8: 为新数据集创建DomainPromptSet
**文件**: `D:\NeuralFSM\neural_fsm_mas\domain_prompts\prompt_manager.py`

**新增类**:
1. `HotpotQADomainPromptSet(DomainPromptSet)`
2. `ALFWorldDomainPromptSet(DomainPromptSet)`
3. `MATHDomainPromptSet(DomainPromptSet)`

**更新**: `DomainPromptManager.get_manager()`
```python
@staticmethod
def get_manager(domain: str) -> DomainPromptSet:
    managers = {
        'mmlu': MMLUDomainPromptSet,
        'gsm8k': GSM8KDomainPromptSet,
        'humaneval': HumanEvalDomainPromptSet,
        'hotpotqa': HotpotQADomainPromptSet,    # ✨ NEW
        'alfworld': ALFWorldDomainPromptSet,    # ✨ NEW
        'math': MATHDomainPromptSet,            # ✨ NEW
    }
    return managers.get(domain, DomainPromptSet)(domain)
```

---

### Phase 5: MMLU类别级FSM系统

#### Task 9 & 10: MMLU类别级FSM生成

**文件**: `D:\NeuralFSM\generate_mmlu_category_fsms.py`

**功能**:
1. 扫描MMLU的57个类别
2. 为每个类别生成专属FSM
3. 保存到 `fsm_cache/mmlu/{category_name}/`
4. 记录生成元数据（时间、LLM成本等）

**目录结构**:
```
fsm_cache/
├── mmlu/
│   ├── abstract_algebra/
│   │   ├── fsm_config.json      # FSM配置
│   │   ├── agents.json          # 智能体定义
│   │   └── metadata.json        # 元数据
│   ├── anatomy/
│   │   ├── fsm_config.json
│   │   ├── agents.json
│   │   └── metadata.json
│   ├── ... (共57个类别)
├── gsm8k/
│   ├── fsm_config.json
│   └── agents.json
├── hotpotqa/
│   ├── fsm_config.json
│   └── agents.json
├── alfworld/
│   ├── fsm_config.json
│   └── agents.json
├── math/
│   ├── fsm_config.json
│   └── agents.json
└── humaneval/
    ├── fsm_config.json
    └── agents.json
```

**脚本接口**:
```python
def generate_mmlu_category_fsms(
    output_dir: str = "./fsm_cache/mmlu",
    categories: Optional[List[str]] = None,  # None = all 57
    use_azure: bool = False,
    save_metadata: bool = True
) -> Dict[str, Any]:
    """
    为MMLU的每个类别生成专属FSM
    
    Returns:
        {
            "total_categories": 57,
            "generated": 57,
            "failed": 0,
            "total_cost": 12.34,
            "categories": {
                "abstract_algebra": {"cost": 0.21, "states": 5, "agents": 4},
                ...
            }
        }
    """
```

**MMLU类别列表** (57个):
```python
MMLU_CATEGORIES = [
    # STEM (18)
    'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_physics',
    'computer_security', 'conceptual_physics', 'electrical_engineering',
    'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_mathematics', 
    'high_school_physics', 'high_school_statistics', 'machine_learning',
    
    # Humanities (13)
    'formal_logic', 'high_school_european_history', 'high_school_us_history',
    'high_school_world_history', 'international_law', 'jurisprudence',
    'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
    'prehistory', 'professional_law', 'world_religions',
    
    # Social Sciences (12)
    'econometrics', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology',
    'human_sexuality', 'professional_psychology', 'public_relations', 
    'security_studies', 'sociology', 'us_foreign_policy',
    
    # Other (14)
    'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
    'global_facts', 'human_aging', 'management', 'marketing', 
    'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting',
    'professional_medicine', 'virology'
]
```

---

### Phase 6: FSM缓存管理系统

#### Task 16: FSM缓存管理器
**文件**: `D:\NeuralFSM\neural_fsm_mas\fsm_cache_manager.py`

**功能**:
- FSM缓存的保存和加载
- 版本管理
- 缓存验证
- 自动重新生成（如缓存失效）

**接口**:
```python
class FSMCacheManager:
    def __init__(self, cache_dir: str = "./fsm_cache")
    
    def save_fsm(self, dataset: str, fsm_config: Dict, 
                 category: Optional[str] = None)
    
    def load_fsm(self, dataset: str, category: Optional[str] = None) -> Dict
    
    def has_cache(self, dataset: str, category: Optional[str] = None) -> bool
    
    def clear_cache(self, dataset: str, category: Optional[str] = None)
    
    def get_cache_info(self, dataset: str, category: Optional[str] = None) -> Dict
    
    def list_all_caches(self) -> Dict[str, List[str]]
    
    def validate_cache(self, dataset: str, category: Optional[str] = None) -> bool
```

---

### Phase 7: 实验脚本

#### Task 11: run_hotpotqa.py
```python
"""
HotpotQA实验脚本
支持：
1. 单FSM模式（整个数据集共享）
2. TGN学习状态转移
3. 多跳推理路径优化
"""
```

#### Task 12: run_alfworld.py
```python
"""
ALFWorld实验脚本
支持：
1. 单FSM模式（整个数据集共享）
2. 动作序列学习
3. 环境交互反馈
"""
```

#### Task 13: run_math.py
```python
"""
MATH实验脚本
支持：
1. 单FSM模式（整个数据集共享）
2. 复杂数学推理
3. 按subject和level过滤
"""
```

#### Task 14: 更新run_mmlu.py
```python
"""
MMLU实验脚本（更新）
新增功能：
1. 自动识别问题所属类别
2. 动态加载对应类别的FSM
3. 类别级性能统计
4. 跨类别泛化评估
"""

# 核心逻辑
def load_category_fsm(category: str) -> Dict:
    cache_manager = FSMCacheManager()
    if cache_manager.has_cache('mmlu', category):
        return cache_manager.load_fsm('mmlu', category)
    else:
        # 动态生成
        fsm = generate_category_fsm(category)
        cache_manager.save_fsm('mmlu', fsm, category)
        return fsm

def run_mmlu_experiment(test_data: List[Dict]):
    results_by_category = {}
    
    for sample in test_data:
        category = sample['category']
        
        # 加载类别专属FSM
        fsm = load_category_fsm(category)
        
        # 运行实验
        result = evaluate_sample(sample, fsm)
        
        # 记录结果
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(result)
    
    return results_by_category
```

---

### Phase 8: 数据集注册系统

#### Task 15: 统一数据集注册表
**文件**: `D:\NeuralFSM\datasets\dataset_registry.py`

```python
class DatasetRegistry:
    """统一数据集注册和管理"""
    
    DATASETS = {
        'gsm8k': {
            'class': 'GSM8KDataset',
            'path': 'datasets/gsm8k/gsm8k.jsonl',
            'type': 'single_fsm',
            'has_categories': False,
        },
        'mmlu': {
            'class': 'MMLUDataset',
            'path': 'datasets/mmlu/data',
            'type': 'category_fsm',
            'has_categories': True,
            'num_categories': 57,
        },
        'humaneval': {
            'class': 'HumanEvalDataset',
            'path': 'datasets/humaneval/humaneval-py.jsonl',
            'type': 'single_fsm',
            'has_categories': False,
        },
        'hotpotqa': {
            'class': 'HotpotQADataset',
            'path': 'datasets/hotpotqa/hotpotqa.jsonl',
            'type': 'single_fsm',
            'has_categories': False,
        },
        'alfworld': {
            'class': 'ALFWorldDataset',
            'path': 'datasets/alfworld/test.jsonl',
            'type': 'single_fsm',
            'has_categories': False,
        },
        'math': {
            'class': 'MATHDataset',
            'path': 'datasets/math/math.jsonl',
            'type': 'single_fsm',
            'has_categories': False,
        },
    }
    
    @staticmethod
    def get_dataset(name: str):
        """获取数据集实例"""
        
    @staticmethod
    def list_datasets() -> List[str]:
        """列出所有数据集"""
        
    @staticmethod
    def get_dataset_info(name: str) -> Dict:
        """获取数据集信息"""
```

---

### Phase 9: 文档和测试

#### Task 19: 综合文档
**文件**: `D:\NeuralFSM\SIX_DATASETS_USAGE_GUIDE.md`

**内容**:
1. 六个数据集的详细介绍
2. FSM生成策略说明
3. 实验脚本使用方法
4. MMLU类别级FSM特殊处理
5. 性能基准和对比
6. 常见问题FAQ

#### Task 20: 集成测试
**文件**: `D:\NeuralFSM\test_all_datasets.py`

**测试项**:
- [ ] 所有数据集加载器
- [ ] FSM生成（6个数据集 + MMLU 57个类别）
- [ ] FSM缓存管理
- [ ] 实验脚本端到端
- [ ] MMLU类别识别和FSM加载
- [ ] 数据集注册表
- [ ] Domain Prompts

---

## 🎯 执行优先级

### 高优先级 (立即执行)
1. ✅ 数据格式分析
2. 数据集加载器 (Task 2, 3, 4)
3. Enhanced_FSM_Gen扩展 (Task 5, 6, 7)
4. FSM缓存管理器 (Task 16)

### 中优先级
5. Domain Prompts扩展 (Task 8)
6. MMLU类别级FSM生成 (Task 9, 10)
7. 实验脚本 (Task 11, 12, 13, 14)

### 低优先级 (后续完善)
8. 数据集注册表 (Task 15)
9. 文档 (Task 19)
10. 测试 (Task 20)

---

## 📊 预期成果

### FSM数量统计
- GSM8K: 1套FSM
- MMLU: 57套FSM (每个类别1套)
- HumanEval: 1套FSM
- HotpotQA: 1套FSM
- ALFWorld: 1套FSM
- MATH: 1套FSM
- **总计**: 62套FSM

### 代码文件统计
- 数据集加载器: 3个新文件
- Enhanced_FSM_Gen: 3个新模板函数
- Domain Prompts: 3个新类
- 实验脚本: 3个新文件 + 1个更新
- FSM缓存管理: 1个新文件
- MMLU类别生成: 1个新脚本
- 数据集注册表: 1个新文件
- 文档: 1个新文件
- 测试: 1个新文件
- **总计**: 约18个新文件/更新

---

## 🚀 开始执行

准备就绪，开始按顺序执行所有任务！

**文档版本**: V1.0  
**最后更新**: 2024-11-13  
**作者**: Neural FSM-MAS Team

