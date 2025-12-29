"""
FSM Cache Manager
FSM缓存管理系统

功能:
1. 保存和加载预生成的FSM配置
2. 支持版本管理和缓存验证
3. MMLU类别级缓存支持
4. 自动生成和更新缓存

目录结构:
fsm_cache/
├── gsm8k/
│   ├── fsm_config.json
│   ├── agents.json
│   └── metadata.json
├── mmlu/
│   ├── abstract_algebra/
│   │   ├── fsm_config.json
│   │   ├── agents.json
│   │   └── metadata.json
│   ├── anatomy/
│   │   └── ...
│   └── ... (57 categories)
├── humaneval/
├── hotpotqa/
├── alfworld/
└── math/
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import shutil


class FSMCacheManager:
    """
    FSM缓存管理器
    
    功能:
    - 保存/加载FSM配置
    - 缓存验证
    - MMLU类别级管理
    - 元数据跟踪
    """
    
    def __init__(self, cache_dir: str = "./fsm_cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存根目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的数据集
        self.supported_datasets = [
            'gsm8k', 'mmlu', 'humaneval', 
            'hotpotqa', 'alfworld', 'math'
        ]
        
        # MMLU特殊处理：有57个类别
        self.mmlu_categories = None  # 延迟加载
    
    def get_cache_path(self, dataset: str, category: Optional[str] = None) -> Path:
        """
        获取缓存路径
        
        Args:
            dataset: 数据集名称
            category: MMLU类别（仅MMLU需要）
        
        Returns:
            缓存目录路径
        """
        if dataset.lower() == 'mmlu' and category:
            return self.cache_dir / 'mmlu' / category
        else:
            return self.cache_dir / dataset.lower()
    
    def has_cache(self, dataset: str, category: Optional[str] = None) -> bool:
        """
        检查缓存是否存在
        
        Args:
            dataset: 数据集名称
            category: MMLU类别（可选）
        
        Returns:
            是否存在缓存
        """
        cache_path = self.get_cache_path(dataset, category)
        
        # 检查必需文件
        required_files = ['fsm_config.json', 'agents.json', 'metadata.json']
        return all((cache_path / f).exists() for f in required_files)
    
    def save_fsm(self, 
                 dataset: str,
                 fsm_config: Dict[str, Any],
                 agents: List[Dict[str, Any]],
                 category: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        保存FSM到缓存
        
        Args:
            dataset: 数据集名称
            fsm_config: FSM配置字典
            agents: 智能体列表
            category: MMLU类别（可选）
            metadata: 额外元数据
        """
        cache_path = self.get_cache_path(dataset, category)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # 保存FSM配置
        with open(cache_path / 'fsm_config.json', 'w', encoding='utf-8') as f:
            json.dump(fsm_config, f, indent=2, ensure_ascii=False)
        
        # 保存智能体
        with open(cache_path / 'agents.json', 'w', encoding='utf-8') as f:
            json.dump(agents, f, indent=2, ensure_ascii=False)
        
        # 生成并保存元数据
        full_metadata = {
            'dataset': dataset,
            'category': category,
            'created_at': datetime.now().isoformat(),
            'num_states': len(fsm_config.get('states', [])),
            'num_agents': len(agents),
            'num_transitions': sum(len(s.get('transitions', [])) for s in fsm_config.get('states', [])),
            'version': '1.0',
            'config_hash': self._compute_hash(fsm_config),
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        with open(cache_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        cache_id = f"{dataset}/{category}" if category else dataset
        print(f"✅ Saved FSM cache for {cache_id} at {cache_path}")
    
    def load_fsm(self, dataset: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        从缓存加载FSM
        
        Args:
            dataset: 数据集名称
            category: MMLU类别（可选）
        
        Returns:
            包含fsm_config, agents, metadata的字典
        
        Raises:
            FileNotFoundError: 如果缓存不存在
        """
        if not self.has_cache(dataset, category):
            cache_id = f"{dataset}/{category}" if category else dataset
            raise FileNotFoundError(f"No cache found for {cache_id}")
        
        cache_path = self.get_cache_path(dataset, category)
        
        # 加载文件
        with open(cache_path / 'fsm_config.json', 'r', encoding='utf-8') as f:
            fsm_config = json.load(f)
        
        with open(cache_path / 'agents.json', 'r', encoding='utf-8') as f:
            agents = json.load(f)
        
        with open(cache_path / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        cache_id = f"{dataset}/{category}" if category else dataset
        print(f"📦 Loaded FSM cache for {cache_id}")
        
        return {
            'fsm_config': fsm_config,
            'agents': agents,
            'metadata': metadata
        }
    
    def clear_cache(self, dataset: str, category: Optional[str] = None):
        """
        清除缓存
        
        Args:
            dataset: 数据集名称
            category: MMLU类别（可选，如果为None则清除整个数据集）
        """
        cache_path = self.get_cache_path(dataset, category)
        
        if cache_path.exists():
            shutil.rmtree(cache_path)
            cache_id = f"{dataset}/{category}" if category else dataset
            print(f"🗑️  Cleared cache for {cache_id}")
        else:
            print(f"⚠️  No cache found to clear")
    
    def get_cache_info(self, dataset: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Args:
            dataset: 数据集名称
            category: MMLU类别（可选）
        
        Returns:
            缓存信息字典
        """
        if not self.has_cache(dataset, category):
            return {'exists': False}
        
        cache_path = self.get_cache_path(dataset, category)
        
        with open(cache_path / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metadata['exists'] = True
        metadata['cache_path'] = str(cache_path)
        
        return metadata
    
    def list_all_caches(self) -> Dict[str, Any]:
        """
        列出所有缓存
        
        Returns:
            {
                'gsm8k': {'exists': True, 'num_states': 8, ...},
                'mmlu': {
                    'abstract_algebra': {...},
                    'anatomy': {...},
                    ...
                },
                ...
            }
        """
        result = {}
        
        for dataset in self.supported_datasets:
            if dataset == 'mmlu':
                # MMLU特殊处理：列出所有类别
                mmlu_dir = self.cache_dir / 'mmlu'
                if mmlu_dir.exists():
                    result['mmlu'] = {}
                    for category_dir in mmlu_dir.iterdir():
                        if category_dir.is_dir():
                            category = category_dir.name
                            result['mmlu'][category] = self.get_cache_info('mmlu', category)
            else:
                # 其他数据集
                result[dataset] = self.get_cache_info(dataset)
        
        return result
    
    def validate_cache(self, dataset: str, category: Optional[str] = None) -> bool:
        """
        验证缓存完整性
        
        Args:
            dataset: 数据集名称
            category: MMLU类别（可选）
        
        Returns:
            是否有效
        """
        if not self.has_cache(dataset, category):
            return False
        
        try:
            # 尝试加载
            cached = self.load_fsm(dataset, category)
            
            # 验证必需字段
            fsm_config = cached['fsm_config']
            agents = cached['agents']
            
            # FSM配置验证
            if 'states' not in fsm_config:
                print(f"❌ Invalid cache: missing 'states' in fsm_config")
                return False
            
            # 智能体验证
            if not isinstance(agents, list) or len(agents) == 0:
                print(f"❌ Invalid cache: agents must be a non-empty list")
                return False
            
            # 状态转移验证
            for state in fsm_config['states']:
                if 'state_name' not in state:
                    print(f"❌ Invalid cache: state missing 'state_name'")
                    return False
            
            print(f"✅ Cache validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Cache validation failed: {e}")
            return False
    
    def _compute_hash(self, data: Any) -> str:
        """
        计算数据哈希值
        
        Args:
            data: 要哈希的数据
        
        Returns:
            SHA256哈希字符串
        """
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def get_mmlu_categories(self) -> List[str]:
        """
        获取MMLU的所有类别
        
        Returns:
            类别列表
        """
        if self.mmlu_categories is None:
            # 从MMLU数据集目录扫描
            mmlu_test_dir = Path("./datasets/mmlu/data/test")
            if mmlu_test_dir.exists():
                self.mmlu_categories = [
                    f.stem.replace('_test', '') 
                    for f in mmlu_test_dir.glob('*_test.csv')
                ]
            else:
                # 硬编码的MMLU类别列表
                self.mmlu_categories = [
                    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
                    'clinical_knowledge', 'college_biology', 'college_chemistry',
                    'college_computer_science', 'college_mathematics', 'college_medicine',
                    'college_physics', 'computer_security', 'conceptual_physics',
                    'econometrics', 'electrical_engineering', 'elementary_mathematics',
                    'formal_logic', 'global_facts', 'high_school_biology',
                    'high_school_chemistry', 'high_school_computer_science',
                    'high_school_european_history', 'high_school_geography',
                    'high_school_government_and_politics', 'high_school_macroeconomics',
                    'high_school_mathematics', 'high_school_microeconomics',
                    'high_school_physics', 'high_school_psychology',
                    'high_school_statistics', 'high_school_us_history',
                    'high_school_world_history', 'human_aging', 'human_sexuality',
                    'international_law', 'jurisprudence', 'logical_fallacies',
                    'machine_learning', 'management', 'marketing', 'medical_genetics',
                    'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
                    'philosophy', 'prehistory', 'professional_accounting',
                    'professional_law', 'professional_medicine', 'professional_psychology',
                    'public_relations', 'security_studies', 'sociology',
                    'us_foreign_policy', 'virology', 'world_religions'
                ]
        
        return self.mmlu_categories
    
    def __repr__(self) -> str:
        return f"FSMCacheManager(cache_dir={self.cache_dir})"


def create_cache_manager(cache_dir: str = "./fsm_cache") -> FSMCacheManager:
    """
    便捷函数: 创建缓存管理器
    
    Args:
        cache_dir: 缓存目录
    
    Returns:
        FSMCacheManager实例
    """
    return FSMCacheManager(cache_dir)


# 示例用法
if __name__ == "__main__":
    # 创建缓存管理器
    manager = create_cache_manager()
    
    # 示例：保存GSM8K的FSM
    example_fsm_config = {
        'states': [
            {'state_name': 'UnderstandProblem', 'description': '...', 'transitions': []},
            {'state_name': 'SolveProblem', 'description': '...', 'transitions': []},
        ]
    }
    example_agents = [
        {'name': 'Analyzer', 'role': '...'},
        {'name': 'Solver', 'role': '...'},
    ]
    
    # manager.save_fsm('gsm8k', example_fsm_config, example_agents)
    
    # 检查缓存
    print(f"\n📊 Cache Status:")
    print(f"  GSM8K: {'✅' if manager.has_cache('gsm8k') else '❌'}")
    print(f"  MMLU (abstract_algebra): {'✅' if manager.has_cache('mmlu', 'abstract_algebra') else '❌'}")
    
    # 列出所有缓存
    print(f"\n📂 All Caches:")
    all_caches = manager.list_all_caches()
    for dataset, info in all_caches.items():
        if dataset == 'mmlu' and isinstance(info, dict):
            print(f"  {dataset}:")
            for category, cat_info in info.items():
                if cat_info.get('exists'):
                    print(f"    - {category}: ✅")
        elif info.get('exists'):
            print(f"  {dataset}: ✅ ({info.get('num_states')} states)")
        else:
            print(f"  {dataset}: ❌")

