
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import random


class UnifiedDataProcessor:
    
    def __init__(self, dataset_root: str = "./datasets"):
        self.dataset_root = Path(dataset_root)
        self.processed_data = {}
        self.domain_splits = {}
        self.gaia_level: Optional[int] = None
        
        self.mmlu_categories = self._define_mmlu_categories()
    
    def _define_mmlu_categories(self) -> Dict[str, List[str]]:
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
        data_dir = self.dataset_root / "mmlu" / "data" / split
        
        if not data_dir.exists():
            raise FileNotFoundError(f"MMLU {split} data not found: {data_dir}")
        
        all_data = []
        
        by_subject = {}
        
        for csv_file in data_dir.glob("*.csv"):
            subject = csv_file.stem.replace(f"_{split}", "")
            
            df = pd.read_csv(csv_file, header=None)
            
            subject_data = []
            for _, row in df.iterrows():
                item = {
                    'domain': 'mmlu',
                    'subject': subject,
                    'question': row[0],
                    'choices': [row[1], row[2], row[3], row[4]],
                    'answer': row[5],  # A/B/C/D
                    'split': split,
                    'full_question': f"{row[0]}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}",
                    'category': self._get_mmlu_category(subject)
                }
                subject_data.append(item)
            
            by_subject[subject] = subject_data
        
        if samples_per_subject is not None:
            for subject, items in by_subject.items():
                if len(items) > samples_per_subject:
                    sampled = random.sample(items, samples_per_subject)
                    all_data.extend(sampled)
                else:
                    all_data.extend(items)
                print(f"  📚 {subject}: {min(len(items), samples_per_subject)}/{len(items)} translated")
        else:
            for items in by_subject.values():
                all_data.extend(items)
        
        return all_data
    
    def _get_mmlu_category(self, subject: str) -> str:
        for category, subjects in self.mmlu_categories.items():
            if subject in subjects:
                return category
        return "Other"
    
    def _load_gsm8k(self, split: str = "train") -> List[Dict[str, Any]]:
        gsm8k_file = self.dataset_root / "gsm8k" / "gsm8k.jsonl"
        
        if not gsm8k_file.exists():
            raise FileNotFoundError(f"GSM8K data not found: {gsm8k_file}")
        
        all_data = []
        
        with open(gsm8k_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item_raw = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid GSM8K JSON line: {e}")
                    continue
                
                if 'question' not in item_raw or 'answer' not in item_raw:
                    continue
                
                answer_text = item_raw['answer']
                final_answer = answer_text.split('####')[-1].strip() if '####' in answer_text else answer_text
                
                item = {
                    'domain': 'gsm8k',
                    'question': item_raw['question'],
                    'answer': final_answer,
                    'full_answer': answer_text,
                    'split': split,
                    'category': 'Mathematics'
                }
                all_data.append(item)
        
        return all_data
    
    def _load_humaneval(self, split: str = "test") -> List[Dict[str, Any]]:
        humaneval_file = self.dataset_root / "humaneval" / "humaneval-py.jsonl"
        
        if not humaneval_file.exists():
            raise FileNotFoundError(f"HumanEval data not found: {humaneval_file}")
        
        all_data = []
        
        with open(humaneval_file, 'r', encoding='utf-8') as f:
            for line in f:
                item_raw = json.loads(line.strip())
                
                if 'prompt' not in item_raw or 'test' not in item_raw:
                    continue
                
                item = {
                    'domain': 'humaneval',
                    'task_name': item_raw['name'],
                    'question': item_raw['prompt'],
                    'answer': item_raw.get('entry_point', ''),
                    'test_code': item_raw.get('test', ''),
                    'entry_point': item_raw.get('entry_point', ''),
                    'language': item_raw.get('language', 'python'),
                    'split': split,
                    'category': 'Code Generation'
                }
                all_data.append(item)
        
        return all_data
    
    def _load_hotpotqa(self, split: str = "test") -> List[Dict[str, Any]]:
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
        
        dataset = HotpotQADataset(str(hotpotqa_file))
        dataset.load_data()
        
        if split == 'all':
            raw_data = dataset.data
        else:
            train_data, val_data, test_data = dataset.get_train_test_split(
                stratify_by_difficulty=True
            )
            
            split_map = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            raw_data = split_map.get(split, test_data)
        
        all_data = []
        for item_raw in raw_data:
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
        
        dataset = ALFWorldDataset(str(alfworld_file))
        dataset.load_data()
        
        train_data, val_data, test_data = dataset.get_train_test_split(
            stratify_by_difficulty=True
        )
        
        split_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        raw_data = split_map.get(split, test_data)
        
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
                'question': item_raw['goal'],
                'answer': item_raw['goal'],
                'full_question': dataset.format_for_llm(item_raw, include_subgoals=False)
            }
            all_data.append(item)
        
        return all_data
    
    def _load_mbpp(self) -> List[Dict[str, Any]]:
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
                    
        all_data = []
        for item_raw in raw_data:
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
                'answer': item_raw['code'],
                'full_question': prompt
            }
            all_data.append(item)
            
        return all_data

    def _load_math(self, split: str = "train") -> List[Dict[str, Any]]:
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
        
        dataset = MATHDataset(str(math_file))
        dataset.load_data()
        
        raw_data = dataset.data
        
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
                'question': item_raw['problem'],
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
        random.seed(random_seed)
        
        if domain == 'mmlu':
            print("🔹 MMLUtranslated：")
            print("  translated：translated dev/ translated 3 translated (translated171translated)")
            print("  translated：translated（translated）")
            print("  translated：translated test/ translated 3 translated (translated171translated)")
            
            splits = {
                'train': self._load_mmlu('dev', samples_per_subject=3),
                'val': [],
                'test': self._load_mmlu('test', samples_per_subject=3)
            }
        elif domain == 'alfworld':
            print("🔹 ALFWorldtranslated：")
            print("  translated：datasets/alfworld/test.jsonl")
            print("  translated：translated（70% translated，30% translated）")
            print("  ✅ translated：translated（translated = 0）")
            
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
            
            dataset.data = []
            with open(alfworld_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        dataset.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            all_raw_data = dataset.data.copy()
            random.shuffle(all_raw_data)
            
            n = len(all_raw_data)
            train_end = int(n * 0.7)
            
            train_raw = all_raw_data[:train_end]
            test_raw = all_raw_data[train_end:]
            
            print(f"\n✅ ALFWorldtranslated:")
            print(f"  translated: {n} translated")
            print(f"  translated: {len(train_raw)} translated ({len(train_raw)/n*100:.1f}%)")
            print(f"  translated: 0 translated (translated)")
            print(f"  translated: {len(test_raw)} translated ({len(test_raw)/n*100:.1f}%)")
            
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
                'val': [],
                'test': convert_to_unified(test_raw)
            }
        elif domain == 'hotpotqa':
            print("🔹 HotpotQAtranslated：")
            print("  translated：datasets/hotpotqa/hotpotqa.jsonl")
            print("  translated：translated（70% / 15% / 15%）")
            print("  translated：translated，translated")
            
            all_data = self.load_dataset(domain, split='all')
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
            print("🔹 MATHtranslated：")
            print("  translated：datasets/math/math.jsonl")
            print("  translated：translated（70% / 15% / 15%）")
            print("  translated：translated，translated/translated")
            
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
            print("🔹 MBPPtranslated：")
            print("  translated：datasets/mbpp/mbpp.jsonl")
            print("  translated：translated（70% translated，30% translated）")
            print("  ✅ translated：translated（translated = 0）")
            
            all_data = self._load_mbpp()
            random.shuffle(all_data)
            
            n = len(all_data)
            train_end = int(n * 0.7)
            
            train_raw = all_data[:train_end]
            test_raw = all_data[train_end:]
            
            print(f"\n✅ MBPPtranslated:")
            print(f"  translated: {n} translated")
            print(f"  translated: {len(train_raw)} translated (70.0%)")
            print(f"  translated: 0 translated (translated)")
            print(f"  translated: {len(test_raw)} translated (30.0%)")
            
            splits = {
                'train': train_raw,
                'val': [],
                'test': test_raw
            }
        else:
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
        if domain in self.domain_splits and split in self.domain_splits[domain]:
            data = self.domain_splits[domain][split]
        else:
            data = self.load_dataset(domain, split)
        
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def get_domain_statistics(self, domain: str = None) -> Dict[str, Any]:
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
                    test_data = self.load_dataset(d, 'test')
                    stats[d] = {'test': len(test_data)}
            except FileNotFoundError:
                stats[d] = {'error': 'Dataset not found'}
        
        return stats
    
    def get_task_description(self, domain: str) -> str:
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
        domain = item['domain']
        
        if domain == 'mmlu':
            return f"{item['full_question']}\n\nPlease select the correct answer (A/B/C/D)."
        elif domain == 'gpqa':
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
            import re
            code = item.get('code', '')
            func_match = re.search(r'def\s+(\w+)\s*\([^)]*\)', code)
            if func_match:
                func_signature = func_match.group(0)
                return f"{item['text']}\n\nImplement the function with this signature:\n{func_signature}\n\nYour implementation should pass these tests:\n{chr(10).join(item.get('test_list', [])[:3])}"
            else:
                return item.get('full_question', item['question'])
        
        elif domain == 'hotpotqa':
            return item.get('full_question', item['question'])
        
        elif domain == 'alfworld':
            return item.get('full_question', item['question'])
        
        elif domain == 'math':
            return item.get('full_question', item['question'])
        
        return item['question']
    
    def format_question_for_agents(self, question_data: Dict[str, Any], domain: str) -> str:
        return self.format_for_llm(question_data)


class MMLUDataProcessor(UnifiedDataProcessor):
    
    def __init__(self, data_root_path: str):
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
        data = self._load_mmlu(split)
        
        by_subject = defaultdict(list)
        for item in data:
            by_subject[item['subject']].append(item)
        
        return dict(by_subject)


if __name__ == "__main__":
    processor = UnifiedDataProcessor("./datasets")
    
    for domain in ['mmlu', 'gsm8k', 'humaneval']:
        try:
            print(f"\n{'='*60}")
            print(f"Testing {domain.upper()}")
            print('='*60)
            
            data = processor.load_dataset(domain, 'test')
            print(f"✅ Loaded {len(data)} samples")
            
            if data:
                print(f"\nSample:")
                sample = data[0]
                formatted = processor.format_for_llm(sample)
                print(formatted[:200] + "..." if len(formatted) > 200 else formatted)
            
            task_desc = processor.get_task_description(domain)
            print(f"\nTask Description: {task_desc[:150]}...")
            
        except Exception as e:
            print(f"❌ Error loading {domain}: {e}")
    
    print("\n" + "="*60)
    print("✅ Unified Data Processor Test Complete")
    print("="*60)

