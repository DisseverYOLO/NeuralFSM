"""
统一数据处理器
Unified Data Processor for MMLU, GSM8K, and HumanEval

支持所有数据集的统一接口，用于训练TGN网络
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import random


class UnifiedDataProcessor:
    """
    统一数据处理器
    
    功能：
    1. 加载MMLU、GSM8K、HumanEval数据集
    2. 统一数据格式
    3. 创建训练/验证/测试分割
    4. 为多智能体系统准备数据
    """
    
    def __init__(self, dataset_root: str = "./datasets"):
        self.dataset_root = Path(dataset_root)
        self.processed_data = {}
        self.domain_splits = {}
        # ✨ GAIA: 运行期可由训练器设置（1/2/3），用于按难度过滤
        self.gaia_level: Optional[int] = None
        
        # MMLU学科分类
        self.mmlu_categories = self._define_mmlu_categories()
    
    def _define_mmlu_categories(self) -> Dict[str, List[str]]:
        """定义MMLU学科分类"""
        return {
            "STEM": [
                "abstract_algebra", "anatomy", "astronomy", "college_biology",
                "college_chemistry", "college_computer_science", "college_mathematics",
                "college_physics", "computer_security", "conceptual_physics",
                "electrical_engineering", "elementary_mathematics", "high_school_biology",
                "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
                "high_school_physics", "high_school_statistics", "machine_learning"
            ],
            "Humanities": [
                "formal_logic", "high_school_european_history", "high_school_us_history",
                "high_school_world_history", "philosophy", "prehistory", "world_religions"
            ],
            "Social_Sciences": [
                "econometrics", "high_school_geography", "high_school_government_and_politics",
                "high_school_macroeconomics", "high_school_microeconomics", "human_aging",
                "international_law", "jurisprudence", "professional_law", "sociology"
            ],
            "Other": [
                "business_ethics", "clinical_knowledge", "medical_genetics", "nutrition"
            ]
        }
    
    def load_dataset(self, domain: str, split: str = "test") -> List[Dict[str, Any]]:
        """
        加载指定领域的数据集
        
        Args:
            domain: 'mmlu', 'gsm8k', 'humaneval', 'hotpotqa', 'alfworld', 'math', 'mbpp', 'gpqa', 'gaia'
            split: 'train', 'test', 'val', 'dev'
        
        Returns:
            统一格式的数据列表
        """
        if domain == 'mmlu':
            return self._load_mmlu(split)
        elif domain == 'gsm8k':
            return self._load_gsm8k(split)
        elif domain == 'humaneval':
            return self._load_humaneval(split)
        elif domain == 'hotpotqa':
            return self._load_hotpotqa(split)
        elif domain == 'alfworld':
            return self._load_alfworld(split)
        elif domain == 'math':
            return self._load_math(split)
        elif domain == 'gpqa':
            return self._load_gpqa(split)
        elif domain == 'gaia':
            return self._load_gaia(split)
        else:
            raise ValueError(f"Unknown domain: {domain}. Supported: mmlu, gsm8k, humaneval, hotpotqa, alfworld, math, mbpp, gpqa, gaia")

    def _load_gpqa(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        加载 GPQA Diamond 数据集（选择题）

        预期文件:
          datasets/gpqa/gpqa_diamond.jsonl

        字段约定（jsonl每行）:
          - task_id: int
          - question: str（包含选项 A/B/C/D）
          - answer: str（'A'/'B'/'C'/'D'）
        """
        gpqa_file = self.dataset_root / "gpqa" / "gpqa_diamond.jsonl"
        if not gpqa_file.exists():
            raise FileNotFoundError(f"GPQA data not found: {gpqa_file}")

        all_data: List[Dict[str, Any]] = []
        with open(gpqa_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item_raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                q = item_raw.get("question", "")
                a = str(item_raw.get("answer", "")).strip().upper()
                # 统一为 A/B/C/D
                if a and a[0] in "ABCD":
                    a = a[0]

                item = {
                    "domain": "gpqa",
                    "task_id": item_raw.get("task_id"),
                    "question": q,
                    "answer": a,
                    "split": split,
                    "category": "GPQA",
                    "full_question": q,
                }
                all_data.append(item)

        return all_data

    def _load_gaia(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        加载 GAIA 数据集（Level 1/2/3）。

        GAIA 官方数据通常托管在 HuggingFace 数据集仓库，并且验证集包含公开答案。
        由于仓库可能需要登录/同意条款，本项目采用本地落盘方式读取：

        预期目录：
          datasets/gaia/

        支持两种本地格式（任选其一）：
        1) JSONL（推荐，便于训练管线复用）：
           - gaia_level1.jsonl / gaia_level2.jsonl / gaia_level3.jsonl
           每行字段建议包含：question, answer, level, file_path(可选)
        2) Parquet（来自HF仓库验证集）：
           - metadata.level1.parquet / metadata.level2.parquet / metadata.level3.parquet

        运行期过滤：
          self.gaia_level = 1/2/3 时，只加载对应 level；否则合并 1-3。
        """
        gaia_dir = self.dataset_root / "gaia"
        if not gaia_dir.exists():
            raise FileNotFoundError(f"GAIA data dir not found: {gaia_dir}")

        levels = [self.gaia_level] if self.gaia_level in (1, 2, 3) else [1, 2, 3]

        all_data: List[Dict[str, Any]] = []

        def normalize_record(q: str, a: str, level: int, file_path: str = "") -> Dict[str, Any]:
            q = q or ""
            a = "" if a is None else str(a)
            file_path = file_path or ""
            return {
                "domain": "gaia",
                "question": q,
                "answer": a,
                "level": int(level),
                "file_path": file_path,
                "split": split,
                "category": f"GAIA-Level{int(level)}",
                "full_question": q,
            }

        # 优先读取 JSONL（无需额外依赖）
        jsonl_loaded = False
        for level in levels:
            jsonl_path = gaia_dir / f"gaia_level{level}.jsonl"
            if not jsonl_path.exists():
                continue
            jsonl_loaded = True
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item_raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q = item_raw.get("question") or item_raw.get("Question") or ""
                    a = item_raw.get("answer") or item_raw.get("Answer") or ""
                    fp = item_raw.get("file_path") or item_raw.get("file_name") or item_raw.get("file") or ""
                    all_data.append(normalize_record(q=q, a=a, level=level, file_path=fp))

        if jsonl_loaded:
            return all_data

        # 回退：读取 Parquet（需要 pyarrow）
        try:
            import pandas as _pd  # noqa: F401
        except Exception as e:
            raise ImportError(
                "GAIA parquet metadata found but pandas/pyarrow is required. "
                "Please convert parquet -> jsonl or install pyarrow."
            ) from e

        for level in levels:
            parquet_path = gaia_dir / f"metadata.level{level}.parquet"
            if not parquet_path.exists():
                continue
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read GAIA parquet: {parquet_path}. "
                                   f"Consider installing pyarrow or converting to jsonl.") from e

            # 常见列名（GAIA仓库字段可能为 Question/Final answer/file_path）
            q_col = "Question" if "Question" in df.columns else ("question" if "question" in df.columns else None)
            a_col = "Final answer" if "Final answer" in df.columns else (
                "answer" if "answer" in df.columns else ("Answer" if "Answer" in df.columns else None)
            )
            fp_col = "file_path" if "file_path" in df.columns else ("file_name" if "file_name" in df.columns else "")

            if q_col is None:
                continue

            for _, row in df.iterrows():
                q = row.get(q_col, "")
                a = row.get(a_col, "") if a_col else ""
                fp = row.get(fp_col, "") if fp_col else ""
                all_data.append(normalize_record(q=str(q), a=a, level=level, file_path=str(fp) if fp is not None else ""))

        if not all_data:
            raise FileNotFoundError(
                f"GAIA metadata not found under {gaia_dir}. "
                "Expected gaia_level{1,2,3}.jsonl or metadata.level{1,2,3}.parquet."
            )

        return all_data
    
    def _load_mmlu(self, split: str = "test", samples_per_subject: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载MMLU数据集
        
        Args:
            split: 数据集分割 ('dev', 'val', 'test', 'auxiliary_train')
            samples_per_subject: 每个学科采样的问题数量，None表示加载全部
        
        Returns:
            统一格式的数据列表
        """
        data_dir = self.dataset_root / "mmlu" / "data" / split
        
        if not data_dir.exists():
            raise FileNotFoundError(f"MMLU {split} data not found: {data_dir}")
        
        all_data = []
        
        # 按学科组织数据
        by_subject = {}
        
        for csv_file in data_dir.glob("*.csv"):
            subject = csv_file.stem.replace(f"_{split}", "")
            
            df = pd.read_csv(csv_file, header=None)
            
            subject_data = []
            for _, row in df.iterrows():
                # 统一格式
                item = {
                    'domain': 'mmlu',
                    'subject': subject,
                    'question': row[0],
                    'choices': [row[1], row[2], row[3], row[4]],
                    'answer': row[5],  # A/B/C/D
                    'split': split,
                    # 构建完整问题文本
                    'full_question': f"{row[0]}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}",
                    'category': self._get_mmlu_category(subject)
                }
                subject_data.append(item)
            
            by_subject[subject] = subject_data
        
        # ✨ 如果指定了每个学科的采样数量，进行随机采样
        if samples_per_subject is not None:
            for subject, items in by_subject.items():
                if len(items) > samples_per_subject:
                    sampled = random.sample(items, samples_per_subject)
                    all_data.extend(sampled)
                else:
                    all_data.extend(items)
                print(f"  📚 {subject}: {min(len(items), samples_per_subject)}/{len(items)} 问题")
        else:
            for items in by_subject.values():
                all_data.extend(items)
        
        return all_data
    
    def _get_mmlu_category(self, subject: str) -> str:
        """获取MMLU学科类别"""
        for category, subjects in self.mmlu_categories.items():
            if subject in subjects:
                return category
        return "Other"
    
    def _load_gsm8k(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        加载GSM8K数据集
        
        注意: GSM8K答案格式为 "推理过程\n#### 最终答案"
        需要保留完整答案（包含推理过程）用于训练，提取最终答案用于验证
        """
        gsm8k_file = self.dataset_root / "gsm8k" / "gsm8k.jsonl"
        
        if not gsm8k_file.exists():
            raise FileNotFoundError(f"GSM8K data not found: {gsm8k_file}")
        
        all_data = []
        
        with open(gsm8k_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行，避免 json.loads("") 报错
                if not line:
                    continue
                
                try:
                    item_raw = json.loads(line)
                except json.JSONDecodeError as e:
                    # 数据集中可能存在格式异常的行，这里跳过并给出提示，但不中断整个训练
                    print(f"⚠️  Skipping invalid GSM8K JSON line: {e}")
                    continue
                
                # 验证必需字段
                if 'question' not in item_raw or 'answer' not in item_raw:
                    continue  # 跳过不完整的数据
                
                # 提取最终答案（用于验证）
                answer_text = item_raw['answer']
                final_answer = answer_text.split('####')[-1].strip() if '####' in answer_text else answer_text
                
                # 统一格式
                item = {
                    'domain': 'gsm8k',
                    'question': item_raw['question'],
                    'answer': final_answer,  # 最终答案（用于验证）
                    'full_answer': answer_text,  # 完整答案（包含推理过程，用于训练）
                    'split': split,
                    'category': 'Mathematics'
                }
                all_data.append(item)
        
        return all_data
    
    def _load_humaneval(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        加载HumanEval数据集
        
        注意: HumanEval数据集通常用于测试，所有数据都包含测试代码
        分割时确保测试代码完整
        """
        humaneval_file = self.dataset_root / "humaneval" / "humaneval-py.jsonl"
        
        if not humaneval_file.exists():
            raise FileNotFoundError(f"HumanEval data not found: {humaneval_file}")
        
        all_data = []
        
        with open(humaneval_file, 'r', encoding='utf-8') as f:
            for line in f:
                item_raw = json.loads(line.strip())
                
                # 验证必需字段
                if 'prompt' not in item_raw or 'test' not in item_raw:
                    continue  # 跳过不完整的数据
                
                # 统一格式
                item = {
                    'domain': 'humaneval',
                    'task_name': item_raw['name'],
                    'question': item_raw['prompt'],  # 代码提示
                    'answer': item_raw.get('entry_point', ''),  # 函数名
                    'test_code': item_raw.get('test', ''),  # ✨ 测试代码（必需）
                    'entry_point': item_raw.get('entry_point', ''),  # ✨ 函数入口点
                    'language': item_raw.get('language', 'python'),
                    'split': split,
                    'category': 'Code Generation'
                }
                all_data.append(item)
        
        return all_data
    
    def _load_hotpotqa(self, split: str = "test") -> List[Dict[str, Any]]:
        """加载HotpotQA数据集"""
        # 导入数据集加载器（使用importlib避免与HuggingFace datasets包冲突）
        import importlib.util
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent.resolve()
        hotpotqa_dataset_path = project_root / "datasets" / "hotpotqa_dataset.py"
        
        if not hotpotqa_dataset_path.exists():
            raise FileNotFoundError(f"Cannot find hotpotqa_dataset.py at {hotpotqa_dataset_path}")
        
        spec = importlib.util.spec_from_file_location("hotpotqa_dataset", hotpotqa_dataset_path)
        hotpotqa_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hotpotqa_module)
        HotpotQADataset = hotpotqa_module.HotpotQADataset
        
        hotpotqa_file = self.dataset_root / "hotpotqa" / "hotpotqa.jsonl"
        
        if not hotpotqa_file.exists():
            raise FileNotFoundError(f"HotpotQA data not found: {hotpotqa_file}")
        
        # 使用数据集加载器
        dataset = HotpotQADataset(str(hotpotqa_file))
        dataset.load_data()
        
        # ✨ 如果请求 'all'，直接返回所有原始数据（用于外部划分）
        if split == 'all':
            raw_data = dataset.data
        else:
            # 获取分割（按难度分层采样）
            train_data, val_data, test_data = dataset.get_train_test_split(
                stratify_by_difficulty=True  # ✨ 按难度分层
            )
            
            split_map = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            raw_data = split_map.get(split, test_data)
        
        # 转换为统一格式
        all_data = []
        for item_raw in raw_data:
            # 格式化上下文
            context_text = dataset.format_context(item_raw, max_docs=10)
            
            item = {
                'domain': 'hotpotqa',
                'question': item_raw['question'],
                'answer': item_raw['answer'],
                'context': context_text,
                'supporting_facts': item_raw.get('supporting_facts', []),
                'type': item_raw.get('type', 'bridge'),  # bridge or comparison
                'level': item_raw.get('level', 'medium'),  # easy, medium, hard
                'split': split,
                'category': f"Multi-hop QA ({item_raw.get('type', 'bridge')})",
                'full_question': f"Context:\n{context_text}\n\nQuestion: {item_raw['question']}"
            }
            all_data.append(item)
        
        return all_data
    
    def _load_alfworld(self, split: str = "test") -> List[Dict[str, Any]]:
        """加载ALFWorld数据集"""
        # 导入数据集加载器（使用importlib避免与HuggingFace datasets包冲突）
        import importlib.util
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent.resolve()
        alfworld_dataset_path = project_root / "datasets" / "alfworld_dataset.py"
        
        if not alfworld_dataset_path.exists():
            raise FileNotFoundError(f"Cannot find alfworld_dataset.py at {alfworld_dataset_path}")
        
        spec = importlib.util.spec_from_file_location("alfworld_dataset", alfworld_dataset_path)
        alfworld_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alfworld_module)
        ALFWorldDataset = alfworld_module.ALFWorldDataset
        
        alfworld_file = self.dataset_root / "alfworld" / "test.jsonl"
        
        if not alfworld_file.exists():
            raise FileNotFoundError(f"ALFWorld data not found: {alfworld_file}")
        
        # 使用数据集加载器
        dataset = ALFWorldDataset(str(alfworld_file))
        dataset.load_data()
        
        # 获取分割（按难度分层采样）
        train_data, val_data, test_data = dataset.get_train_test_split(
            stratify_by_difficulty=True  # ✨ 按难度分层
        )
        
        split_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        raw_data = split_map.get(split, test_data)
        
        # 转换为统一格式
        all_data = []
        for item_raw in raw_data:
            item = {
                'domain': 'alfworld',
                'goal': item_raw['goal'],
                'subgoals': item_raw.get('subgoals', []),
                'difficulty': item_raw.get('difficulty', 'hard'),
                'task_description': dataset.get_task_description(item_raw),
                'split': split,
                'category': 'Embodied AI',
                'question': item_raw['goal'],  # 使用goal作为question
                'answer': item_raw['goal'],  # 目标本身就是答案
                'full_question': dataset.format_for_llm(item_raw, include_subgoals=False)
            }
            all_data.append(item)
        
        return all_data
    
    def _load_mbpp(self) -> List[Dict[str, Any]]:
        """
        加载MBPP数据集
        
        数据格式：
        {
            "text": "问题描述",
            "code": "参考代码",
            "test_list": ["assert ...", ...],
            "task_id": 123
        }
        """
        mbpp_file = self.dataset_root / "mbpp" / "mbpp.jsonl"
        
        if not mbpp_file.exists():
            raise FileNotFoundError(f"MBPP data not found: {mbpp_file}")
            
        raw_data = []
        with open(mbpp_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    raw_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
                    
        # 转换为统一格式
        all_data = []
        for item_raw in raw_data:
            # 构建 prompt
            # MBPP 需要根据 text 生成代码，test_list 用于验证
            prompt = f"""Task: {item_raw['text']}
            
Please write a Python function to solve this problem.
The function should pass the following tests:
{chr(10).join(item_raw['test_list'][:3])}

Your code:
"""
            item = {
                'domain': 'mbpp',
                'question': item_raw['text'],
                'text': item_raw['text'],
                'code': item_raw['code'],
                'test_list': item_raw['test_list'],
                'task_id': item_raw.get('task_id'),
                'category': 'Python Coding',
                'answer': item_raw['code'],  # 参考答案
                'full_question': prompt
            }
            all_data.append(item)
            
        return all_data

    def _load_math(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        加载MATH数据集
        
        注意：对于小数据集（如80条），直接返回所有数据，由create_domain_splits统一划分
        这样可以避免分层采样导致某些subject的验证/测试集过小或为空
        """
        # 导入数据集加载器（使用importlib避免与HuggingFace datasets包冲突）
        import importlib.util
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent.resolve()
        math_dataset_path = project_root / "datasets" / "math_dataset.py"
        
        if not math_dataset_path.exists():
            raise FileNotFoundError(f"Cannot find math_dataset.py at {math_dataset_path}")
        
        spec = importlib.util.spec_from_file_location("math_dataset", math_dataset_path)
        math_dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(math_dataset_module)
        MATHDataset = math_dataset_module.MATHDataset
        
        math_file = self.dataset_root / "math" / "math.jsonl"
        
        if not math_file.exists():
            raise FileNotFoundError(f"MATH data not found: {math_file}")
        
        # 使用数据集加载器
        dataset = MATHDataset(str(math_file))
        dataset.load_data()
        
        # ✨ 直接返回所有原始数据，不进行预划分
        # 由create_domain_splits统一进行简单随机划分（类似GSM8k）
        # 这样可以避免小数据集分层采样导致验证/测试集过小的问题
        raw_data = dataset.data
        
        # 转换为统一格式
        all_data = []
        for item_raw in raw_data:
            item = {
                'domain': 'math',
                'problem': item_raw['problem'],
                'solution': item_raw.get('solution', ''),
                'answer': item_raw['answer'],
                'subject': item_raw.get('subject', 'Unknown'),
                'level': item_raw.get('level', 0),
                'unique_id': item_raw.get('unique_id', ''),
                'split': split,
                'category': item_raw.get('subject', 'Mathematics'),
                'question': item_raw['problem'],  # 问题文本
                'full_question': dataset.format_for_llm(item_raw, include_solution=False)
            }
            all_data.append(item)
        
        return all_data
    
    def create_domain_splits(self, 
                            domain: str,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            random_seed: int = 42) -> Dict[str, List[Dict]]:
        """
        为指定领域创建训练/验证/测试分割
        
        Args:
            domain: 数据集名称
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        
        Returns:
            包含train/val/test的字典
        """
        random.seed(random_seed)
        
        # 加载完整数据
        if domain == 'mmlu':
            # ✨ MMLU特殊处理：使用dev作为训练集，test作为测试集，不使用验证集
            # 每个学科随机采样3个问题
            print("🔹 MMLU数据集加载配置：")
            print("  训练集：从 dev/ 中每个学科随机采样 3 个问题 (约171个)")
            print("  验证集：不使用（跳过验证阶段）")
            print("  测试集：从 test/ 中每个学科随机采样 3 个问题 (约171个)")
            
            splits = {
                'train': self._load_mmlu('dev', samples_per_subject=3),
                'val': [],  # 不使用验证集
                'test': self._load_mmlu('test', samples_per_subject=3)
            }
        elif domain == 'alfworld':
            # ✨ ALFWorld特殊处理：只加载一次，然后手动划分为训练集和测试集
            # 不使用验证集，70%训练，30%测试
            print("🔹 ALFWorld数据集加载配置：")
            print("  数据源：datasets/alfworld/test.jsonl")
            print("  划分方式：简单随机划分（70% 训练，30% 测试）")
            print("  ✅ 验证集：不使用（验证批次 = 0）")
            
            # ✅ 直接加载原始数据，不通过 _load_alfworld（避免重复打印统计）
            import importlib.util
            from pathlib import Path
            
            project_root = Path(__file__).parent.parent.parent.resolve()
            alfworld_dataset_path = project_root / "datasets" / "alfworld_dataset.py"
            
            if not alfworld_dataset_path.exists():
                raise FileNotFoundError(f"Cannot find alfworld_dataset.py at {alfworld_dataset_path}")
            
            spec = importlib.util.spec_from_file_location("alfworld_dataset", alfworld_dataset_path)
            alfworld_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(alfworld_module)
            ALFWorldDataset = alfworld_module.ALFWorldDataset
            
            alfworld_file = self.dataset_root / "alfworld" / "test.jsonl"
            dataset = ALFWorldDataset(str(alfworld_file))
            
            # ✅ 静默加载数据（不打印统计信息）
            dataset.data = []
            with open(alfworld_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        dataset.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            # 简单随机划分：70%训练，30%测试
            all_raw_data = dataset.data.copy()
            random.shuffle(all_raw_data)
            
            n = len(all_raw_data)
            train_end = int(n * 0.7)
            
            train_raw = all_raw_data[:train_end]
            test_raw = all_raw_data[train_end:]
            
            print(f"\n✅ ALFWorld数据划分完成:")
            print(f"  总数: {n} 个任务")
            print(f"  训练集: {len(train_raw)} 个任务 ({len(train_raw)/n*100:.1f}%)")
            print(f"  验证集: 0 个任务 (不使用验证阶段)")
            print(f"  测试集: {len(test_raw)} 个任务 ({len(test_raw)/n*100:.1f}%)")
            
            # 转换为统一格式
            def convert_to_unified(raw_samples):
                unified = []
                for item_raw in raw_samples:
                    item = {
                        'domain': 'alfworld',
                        'goal': item_raw['goal'],
                        'subgoals': item_raw.get('subgoals', []),
                        'difficulty': item_raw.get('difficulty', 'hard'),
                        'task_description': dataset.get_task_description(item_raw),
                        'category': 'Embodied AI',
                        'question': item_raw['goal'],
                        'answer': item_raw['goal'],
                        'full_question': dataset.format_for_llm(item_raw, include_subgoals=False)
                    }
                    unified.append(item)
                return unified
            
            splits = {
                'train': convert_to_unified(train_raw),
                'val': [],  # 不使用验证集
                'test': convert_to_unified(test_raw)
            }
        elif domain == 'hotpotqa':
            # ✨ HotpotQA：使用简单随机划分（类似GSM8k/MATH），避免小数据集分层采样的潜在问题
            print("🔹 HotpotQA数据集加载配置：")
            print("  数据源：datasets/hotpotqa/hotpotqa.jsonl")
            print("  划分方式：简单随机划分（70% / 15% / 15%）")
            print("  说明：不使用按难度分层采样，确保小数据集划分稳定")
            
            all_data = self.load_dataset(domain, split='all')  # 加载全部数据
            random.shuffle(all_data)
            
            n = len(all_data)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits = {
                'train': all_data[:train_end],
                'val': all_data[train_end:val_end],
                'test': all_data[val_end:]
            }
        elif domain == 'math':
            # ✨ MATH：使用简单随机划分（类似GSM8k），避免小数据集分层采样导致验证/测试集过小
            # 对于80条数据，分层采样会导致某些subject的验证/测试集只有0-1条
            print("🔹 MATH数据集加载配置：")
            print("  数据源：datasets/math/math.jsonl")
            print("  划分方式：简单随机划分（70% / 15% / 15%）")
            print("  说明：不使用分层采样，避免小数据集验证/测试集过小")
            
            all_data = self.load_dataset(domain, split='train')
            random.shuffle(all_data)
            
            n = len(all_data)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits = {
                'train': all_data[:train_end],
                'val': all_data[train_end:val_end],
                'test': all_data[val_end:]
            }
        elif domain == 'mbpp':
            # ✨ MBPP特殊处理：手动加载并划分为训练集和测试集（无验证集）
            print("🔹 MBPP数据集加载配置：")
            print("  数据源：datasets/mbpp/mbpp.jsonl")
            print("  划分方式：简单随机划分（70% 训练，30% 测试）")
            print("  ✅ 验证集：不使用（验证批次 = 0）")
            
            all_data = self._load_mbpp()
            random.shuffle(all_data)
            
            n = len(all_data)
            train_end = int(n * 0.7)
            
            train_raw = all_data[:train_end]
            test_raw = all_data[train_end:]
            
            print(f"\n✅ MBPP数据划分完成:")
            print(f"  总数: {n} 个任务")
            print(f"  训练集: {len(train_raw)} 个任务 (70.0%)")
            print(f"  验证集: 0 个任务 (不使用验证阶段)")
            print(f"  测试集: {len(test_raw)} 个任务 (30.0%)")
            
            splits = {
                'train': train_raw,
                'val': [],  # 不使用验证集
                'test': test_raw
            }
        else:
            # GSM8K、HumanEval等：统一按 train/val/test 三集划分
            all_data = self.load_dataset(domain, split='train')
            random.shuffle(all_data)
            
            n = len(all_data)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits = {
                'train': all_data[:train_end],
                'val': all_data[train_end:val_end],
                'test': all_data[val_end:]
            }
        
        self.domain_splits[domain] = splits
        return splits
    
    def prepare_multi_agent_training_data(self,
                                         domain: str,
                                         split: str = "train",
                                         batch_size: int = 16) -> List[List[Dict]]:
        """
        为多智能体训练准备批次数据
        
        Args:
            domain: 数据集名称
            split: 数据分割
            batch_size: 批次大小
        
        Returns:
            批次数据列表
        """
        # 获取数据
        if domain in self.domain_splits and split in self.domain_splits[domain]:
            data = self.domain_splits[domain][split]
        else:
            data = self.load_dataset(domain, split)
        
        # 创建批次
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def get_domain_statistics(self, domain: str = None) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Args:
            domain: 数据集名称，如果为None则返回所有数据集统计
        
        Returns:
            统计信息字典
        """
        if domain:
            domains = [domain]
        else:
            domains = ['mmlu', 'gsm8k', 'humaneval', 'hotpotqa', 'alfworld', 'math']
        
        stats = {}
        
        for d in domains:
            try:
                if d in self.domain_splits:
                    splits_data = self.domain_splits[d]
                    stats[d] = {
                        'train': len(splits_data.get('train', [])),
                        'val': len(splits_data.get('val', [])),
                        'test': len(splits_data.get('test', []))
                    }
                else:
                    # 尝试加载以获取统计
                    test_data = self.load_dataset(d, 'test')
                    stats[d] = {'test': len(test_data)}
            except FileNotFoundError:
                stats[d] = {'error': 'Dataset not found'}
        
        return stats
    
    def get_task_description(self, domain: str) -> str:
        """
        获取领域任务描述（用于生成FSM和智能体）
        
        Args:
            domain: 数据集名称
        
        Returns:
            任务描述文本
        """
        descriptions = {
            'gpqa': """
Task: GPQA (Graduate-Level Multiple-Choice Question Answering)

You need to solve high-difficulty, graduate-level science multiple-choice questions.
Each question has 4 options (A/B/C/D). The prompt may include intermediate labels like a)/b)/c)/d) and a mapping to A/B/C/D.

Your goal:
1. Parse the question and identify what is being asked.
2. Activate relevant scientific concepts/formulas.
3. Evaluate each option and eliminate distractors systematically.
4. Cross-check for unit/logic consistency.
5. Output ONLY the final option letter: A/B/C/D (no extra text).

Available tools: problem_parsing, knowledge_retrieval, option_elimination, consistency_check, verification
""",
            'gaia': """
Task: GAIA (General AI Assistant Benchmark) - Level-based Questions

You need to answer real-world questions that may require multi-step reasoning and tool usage.
Questions are grouped into three difficulty levels (Level 1/2/3).

Your goal:
1. Understand the question and any referenced resources (e.g., attached files, if provided).
2. Reason step-by-step to reach a precise final answer.
3. Output ONLY the final answer (concise; avoid unnecessary explanation unless requested).

Available tools: reasoning, planning, information_extraction, verification
""",
            'mmlu': """
Task: Comprehensive Multi-domain Multiple Choice Question Answering (57 Academic Subjects)

You need to analyze and answer multiple-choice questions across 57 diverse academic subjects spanning:

**STEM (19 subjects):**
- Mathematics: abstract_algebra, college_mathematics, elementary_mathematics, high_school_mathematics, high_school_statistics
- Physics: college_physics, conceptual_physics, high_school_physics
- Chemistry: college_chemistry, high_school_chemistry
- Computer Science: college_computer_science, computer_security, high_school_computer_science, machine_learning
- Biology & Medicine: anatomy, college_biology, high_school_biology, clinical_knowledge, medical_genetics
- Engineering: electrical_engineering
- Astronomy: astronomy

**Humanities (10 subjects):**
- History: high_school_european_history, high_school_us_history, high_school_world_history, prehistory
- Philosophy: formal_logic, philosophy, moral_disputes, moral_scenarios, logical_fallacies
- Religion: world_religions
- Human Studies: human_sexuality

**Social Sciences (18 subjects):**
- Economics: econometrics, high_school_macroeconomics, high_school_microeconomics
- Political Science: high_school_government_and_politics, international_law, us_foreign_policy, security_studies
- Law: jurisprudence, professional_law
- Psychology: professional_psychology, high_school_psychology
- Sociology: sociology, human_aging
- Geography: high_school_geography
- Accounting: professional_accounting
- Business: business_ethics, marketing, management, public_relations

**Other Domains (10 subjects):**
- Medicine: college_medicine, nutrition, virology
- Miscellaneous: global_facts, miscellaneous

Each question has 4 choices (A, B, C, D). Your goal is to select the correct answer through:
1. **Domain Recognition**: Quickly identify which of the 57 subjects the question belongs to
2. **Knowledge Retrieval**: Recall relevant facts, concepts, and principles from that specific domain
3. **Systematic Analysis**: Evaluate each choice against domain knowledge
4. **Cross-domain Integration**: Some questions may require knowledge from multiple subjects
5. **Logical Reasoning**: Apply domain-specific reasoning patterns (mathematical, scientific, philosophical, legal, etc.)
6. **Confident Decision**: Select the most accurate answer based on comprehensive analysis

Key Capabilities Required:
- Broad knowledge coverage across all 57 academic subjects
- Domain-specific reasoning (quantitative for STEM, qualitative for humanities, analytical for social sciences)
- Ability to switch between different knowledge domains quickly
- Critical thinking and elimination strategies for uncertain questions

Available tools: knowledge_retrieval, domain_identification, logical_reasoning, systematic_analysis, critical_thinking, multi_domain_expertise
""",
            
            'gsm8k': """
Task: Grade School Math Word Problem Solving

You need to solve mathematical word problems that involve:
- Arithmetic operations (addition, subtraction, multiplication, division)
- Multi-step reasoning and calculations
- Understanding real-world contexts and scenarios
- Extracting relevant numerical information from text

Your goal is to provide the correct numerical answer through:
1. Reading and understanding the problem statement
2. Identifying the key information and unknowns
3. Breaking down the problem into logical steps
4. Performing accurate calculations
5. Verifying the solution

Available tools: calculation, logical_reasoning, problem_decomposition, verification, math_solver
""",
            
            'humaneval': """
Task: Python Function Implementation

You need to write Python functions that satisfy given specifications:
- Function signature and docstring are provided
- Must implement the correct logic to pass all test cases
- Code should be clean, efficient, and follow best practices
- Handle edge cases and potential errors

Your goal is to generate correct Python code through:
1. Understanding the function requirements and specifications
2. Designing the algorithm and data structures
3. Writing clean and efficient implementation
4. Testing with example cases
5. Ensuring correctness and handling edge cases

Available tools: code_design, code_writing, algorithm_analysis, testing, debugging
""",
            
            'mbpp': """
Task: Basic Python Programming Problems

You need to write Python functions to solve basic programming problems:
- Problems are described in natural language (e.g., "Write a function to check if a number is prime")
- Usually involves fundamental algorithms, data structures, and logic
- Must pass a set of specific assertions/test cases
- Code should be concise and correct

Your goal is to generate correct Python code through:
1. Understanding the problem description
2. Identifying the core logic required
3. Implementing the solution in Python
4. Verifying against common cases
5. Ensuring standard library usage where appropriate

Available tools: code_writing, algorithm_implementation, logical_thinking, python_basics
""",
            
            'hotpotqa': """
Task: Multi-hop Question Answering

You need to answer questions that require reasoning across multiple documents:
- Questions may be "bridge" type (A→B→C) or "comparison" type (A vs B)
- Need to identify and connect supporting facts from different documents
- Long context with multiple document paragraphs
- Difficulty levels: easy, medium, hard

Your goal is to provide accurate answers through:
1. Understanding the question type and information needs
2. Retrieving relevant documents from context
3. Extracting key facts from documents
4. Connecting facts across documents (multi-hop reasoning)
5. Synthesizing answer with supporting evidence
6. Verifying answer consistency

Available tools: document_retrieval, fact_extraction, multi_hop_reasoning, answer_synthesis, verification
""",
            
            'alfworld': """
Task: Embodied AI Interactive Task in Simulated Household Environment

You are an embodied AI agent that must complete household tasks through text-based interaction with a simulated environment.

**Task Categories:**
1. **Simple Pick and Place** (easy): Find an object and place it somewhere (e.g., "put pencil in shelf")
2. **Pick, Clean, and Place** (hard): Find object → pick it → clean it (using sink/sinkbasin) → place it
3. **Pick, Heat, and Place** (hard): Find object → pick it → heat it (using microwave) → place it
4. **Pick, Cool, and Place** (hard): Find object → pick it → cool it (using fridge) → place it
5. **Examine Object with Light** (hard): Find object → pick it → find desklamp → use desklamp to examine
6. **Pick Two Objects and Place** (hard): Find first object → pick and place → find second same object → pick and place

**Available Actions (Fixed Template):**
- Navigation: go to [receptacle/object] [number]
- Observation: examine [object] [number], look (see current location)
- Manipulation: take [object] [number] from [receptacle] [number], put [object] [number] in/on [receptacle] [number]
- Object State: open/close [receptacle] [number], toggle [object] [number] (turn on/off)
- Special: clean [object] [number] with [sinkbasin] [number], heat [object] [number] with [microwave] [number], cool [object] [number] with [fridge] [number]

**Environment Feedback:**
- After each action, environment returns text feedback describing what you see and action results
- Use feedback to understand current state and plan next action
- Subgoals are marked as achieved when specific patterns appear in feedback

**Problem-Solving Process:**
1. **Goal Understanding**: Parse the high-level goal (e.g., "put a clean plate in countertop")
2. **Subgoal Decomposition**: Break down into subgoals (find plate → pick plate → clean plate → find countertop → place plate)
3. **Environment Exploration**: Navigate and observe to locate required objects and receptacles
4. **Action Planning**: Determine action sequences to achieve each subgoal
5. **Action Execution**: Execute actions one by one, receiving environment feedback
6. **State Monitoring**: Track progress through subgoals based on feedback patterns
7. **Error Recovery**: When actions fail or lead to wrong state, replan and try alternative actions
8. **Task Completion**: Verify all subgoals achieved and final goal completed

**Key Considerations:**
- Objects have numbers (e.g., plate 1, plate 2) - must specify the number
- Receptacles can be open/closed - may need to open before accessing contents
- Some objects are inside other objects (e.g., plate in cabinet) - must open container first
- Each subgoal completion can be verified by regex pattern matching on environment feedback
- Tasks require 5-20 sequential actions depending on complexity

Available tools: goal_parsing, subgoal_decomposition, environment_exploration, action_planning, action_execution, feedback_analysis, state_tracking, error_recovery
""",
            
            'math': """
Task: Advanced Mathematics Competition Problem Solving

You need to solve high-level competition mathematics problems:
- Subjects: Algebra, Geometry, Number Theory, Combinatorics, Precalculus, etc.
- Difficulty levels: 1-5 (competition level)
- Requires deep mathematical reasoning
- Solutions involve multiple steps and concepts
- Answers often in LaTeX format

Your goal is to solve problems through:
1. Analyzing problem type, subject, and key concepts
2. Identifying relevant mathematical concepts and theorems
3. Choosing appropriate solution approach
4. Executing step-by-step solution with detailed reasoning
5. Performing calculations and symbolic manipulations
6. Verifying solution correctness
7. Formatting answer in LaTeX

Available tools: problem_analysis, concept_identification, strategy_design, step_solving, calculation, verification, latex_formatting
"""
        }
        
        return descriptions.get(domain, f"Task: Solve problems in {domain} domain")
    
    def format_for_llm(self, item: Dict[str, Any]) -> str:
        """
        将数据项格式化为LLM输入
        
        Args:
            item: 数据项
        
        Returns:
            格式化的文本
        """
        domain = item['domain']
        
        if domain == 'mmlu':
            return f"{item['full_question']}\n\nPlease select the correct answer (A/B/C/D)."
        elif domain == 'gpqa':
            # GPQA是高难度选择题：强制模型只输出选项字母，避免输出长解释污染评估
            q = item.get('full_question', item['question'])
            return (
                f"{q}\n\n"
                "Instructions:\n"
                "- This is a multiple-choice question. Choose the correct option among A/B/C/D.\n"
                "- If the prompt contains both a)/b)/c)/d) and a mapping to A/B/C/D, FOLLOW the final A/B/C/D mapping.\n"
                "- Output ONLY a single letter: A or B or C or D. No other text (no 'Answer:', no explanation).\n"
            )
        
        elif domain == 'gsm8k':
            return f"Math Problem: {item['question']}\n\nPlease solve this problem and provide the final numerical answer."
        
        elif domain == 'gaia':
            fp = item.get('file_path') or ""
            fp_hint = f"\n\nAttached file path: {fp}" if fp else ""
            return (
                f"GAIA Question (Level {item.get('level', '')}): {item.get('question', '')}"
                f"{fp_hint}\n\n"
                "Please answer with ONLY the final answer."
            )
        
        elif domain == 'humaneval':
            return f"Programming Task:\n{item['question']}\n\nPlease implement the function according to the specification."
        
        elif domain == 'mbpp':
            # ✨ MBPP特殊处理：提取函数签名并包含在prompt中
            import re
            code = item.get('code', '')
            # 提取函数名和参数（支持def func_name(args): 格式）
            func_match = re.search(r'def\s+(\w+)\s*\([^)]*\)', code)
            if func_match:
                func_signature = func_match.group(0)
                return f"{item['text']}\n\nImplement the function with this signature:\n{func_signature}\n\nYour implementation should pass these tests:\n{chr(10).join(item.get('test_list', [])[:3])}"
            else:
                # 回退到使用full_question
                return item.get('full_question', item['question'])
        
        elif domain == 'hotpotqa':
            return item.get('full_question', item['question'])
        
        elif domain == 'alfworld':
            return item.get('full_question', item['question'])
        
        elif domain == 'math':
            return item.get('full_question', item['question'])
        
        return item['question']
    
    def format_question_for_agents(self, question_data: Dict[str, Any], domain: str) -> str:
        """
        为智能体格式化问题（用于训练）
        
        Args:
            question_data: 问题数据
            domain: 数据集名称
        
        Returns:
            格式化的问题文本
        """
        # 使用format_for_llm方法
        return self.format_for_llm(question_data)


# 兼容旧代码：保留MMLUDataProcessor作为别名
class MMLUDataProcessor(UnifiedDataProcessor):
    """MMLU数据处理器（兼容旧代码）"""
    
    def __init__(self, data_root_path: str):
        # 假设传入的是mmlu/data路径
        mmlu_data_path = Path(data_root_path)
        if mmlu_data_path.name == "data":
            dataset_root = mmlu_data_path.parent.parent
        elif mmlu_data_path.name == "mmlu":
            dataset_root = mmlu_data_path.parent
        else:
            dataset_root = mmlu_data_path
        
        super().__init__(str(dataset_root))
        self.data_root_path = Path(data_root_path)
    
    def load_mmlu_data(self, split: str = "test") -> Dict[str, List[Dict]]:
        """加载MMLU数据（兼容旧接口）"""
        data = self._load_mmlu(split)
        
        # 按学科组织
        by_subject = defaultdict(list)
        for item in data:
            by_subject[item['subject']].append(item)
        
        return dict(by_subject)


if __name__ == "__main__":
    # 测试数据处理器
    processor = UnifiedDataProcessor("./datasets")
    
    # 测试加载各个数据集
    for domain in ['mmlu', 'gsm8k', 'humaneval']:
        try:
            print(f"\n{'='*60}")
            print(f"Testing {domain.upper()}")
            print('='*60)
            
            # 加载数据
            data = processor.load_dataset(domain, 'test')
            print(f"✅ Loaded {len(data)} samples")
            
            # 显示第一个样本
            if data:
                print(f"\nSample:")
                sample = data[0]
                formatted = processor.format_for_llm(sample)
                print(formatted[:200] + "..." if len(formatted) > 200 else formatted)
            
            # 获取任务描述
            task_desc = processor.get_task_description(domain)
            print(f"\nTask Description: {task_desc[:150]}...")
            
        except Exception as e:
            print(f"❌ Error loading {domain}: {e}")
    
    print("\n" + "="*60)
    print("✅ Unified Data Processor Test Complete")
    print("="*60)

