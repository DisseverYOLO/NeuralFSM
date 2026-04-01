"""
FSM Cache Manager
FSM cache management system

Features:
1. Save and load pre-generated FSM configurations.
2. Support version management and cache validation.
3. Support category-level caching for MMLU.
4. Automatically generate and update caches.

Directory structure:
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
    FSM cache manager
    
    Features:
    - Save and load FSM configurations
    - Cache validation
    - Category-level management for MMLU
    - Metadata tracking
    """
    
    def __init__(self, cache_dir: str = "./fsm_cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Root cache directory.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported datasets.
        self.supported_datasets = [
            'gsm8k', 'mmlu', 'humaneval', 
            'hotpotqa', 'alfworld', 'math'
        ]
        
        # Special handling for MMLU, which has 57 categories.
        self.mmlu_categories = None  # Lazy-loaded.
    
    def get_cache_path(self, dataset: str, category: Optional[str] = None) -> Path:
        """
        Get the cache path.
        
        Args:
            dataset: Dataset name.
            category: MMLU category, only needed for MMLU.
        
        Returns:
            Cache directory path.
        """
        if dataset.lower() == 'mmlu' and category:
            return self.cache_dir / 'mmlu' / category
        else:
            return self.cache_dir / dataset.lower()
    
    def has_cache(self, dataset: str, category: Optional[str] = None) -> bool:
        """
        Check whether a cache exists.
        
        Args:
            dataset: Dataset name.
            category: Optional MMLU category.
        
        Returns:
            Whether the cache exists.
        """
        cache_path = self.get_cache_path(dataset, category)
        
        # Check required files.
        required_files = ['fsm_config.json', 'agents.json', 'metadata.json']
        return all((cache_path / f).exists() for f in required_files)
    
    def save_fsm(self, 
                 dataset: str,
                 fsm_config: Dict[str, Any],
                 agents: List[Dict[str, Any]],
                 category: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Save an FSM to the cache.
        
        Args:
            dataset: Dataset name.
            fsm_config: FSM configuration dictionary.
            agents: Agent list.
            category: Optional MMLU category.
            metadata: Additional metadata.
        """
        cache_path = self.get_cache_path(dataset, category)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save the FSM configuration.
        with open(cache_path / 'fsm_config.json', 'w', encoding='utf-8') as f:
            json.dump(fsm_config, f, indent=2, ensure_ascii=False)
        
        # Save the agents.
        with open(cache_path / 'agents.json', 'w', encoding='utf-8') as f:
            json.dump(agents, f, indent=2, ensure_ascii=False)
        
        # Generate and save metadata.
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
        Load an FSM from the cache.
        
        Args:
            dataset: Dataset name.
            category: Optional MMLU category.
        
        Returns:
            Dictionary containing `fsm_config`, `agents`, and `metadata`.
        
        Raises:
            FileNotFoundError: Raised if the cache does not exist.
        """
        if not self.has_cache(dataset, category):
            cache_id = f"{dataset}/{category}" if category else dataset
            raise FileNotFoundError(f"No cache found for {cache_id}")
        
        cache_path = self.get_cache_path(dataset, category)
        
        # Load files.
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
        Clear a cache.
        
        Args:
            dataset: Dataset name.
            category: Optional MMLU category. If `None`, clear the entire dataset cache.
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
        Get cache information.
        
        Args:
            dataset: Dataset name.
            category: Optional MMLU category.
        
        Returns:
            Cache information dictionary.
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
        List all caches.
        
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
                # Special handling for MMLU: list all categories.
                mmlu_dir = self.cache_dir / 'mmlu'
                if mmlu_dir.exists():
                    result['mmlu'] = {}
                    for category_dir in mmlu_dir.iterdir():
                        if category_dir.is_dir():
                            category = category_dir.name
                            result['mmlu'][category] = self.get_cache_info('mmlu', category)
            else:
                # Other datasets.
                result[dataset] = self.get_cache_info(dataset)
        
        return result
    
    def validate_cache(self, dataset: str, category: Optional[str] = None) -> bool:
        """
        Validate cache integrity.
        
        Args:
            dataset: Dataset name.
            category: Optional MMLU category.
        
        Returns:
            Whether the cache is valid.
        """
        if not self.has_cache(dataset, category):
            return False
        
        try:
            # Try loading the cache.
            cached = self.load_fsm(dataset, category)
            
            # Validate required fields.
            fsm_config = cached['fsm_config']
            agents = cached['agents']
            
            # Validate the FSM configuration.
            if 'states' not in fsm_config:
                print(f"❌ Invalid cache: missing 'states' in fsm_config")
                return False
            
            # Validate the agents.
            if not isinstance(agents, list) or len(agents) == 0:
                print(f"❌ Invalid cache: agents must be a non-empty list")
                return False
            
            # Validate state transitions.
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
        Compute the data hash value.
        
        Args:
            data: Data to hash.
        
        Returns:
            SHA256 hash string.
        """
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def get_mmlu_categories(self) -> List[str]:
        """
        Get all MMLU categories.
        
        Returns:
            Category list.
        """
        if self.mmlu_categories is None:
            # Scan the MMLU dataset directory.
            mmlu_test_dir = Path("./datasets/mmlu/data/test")
            if mmlu_test_dir.exists():
                self.mmlu_categories = [
                    f.stem.replace('_test', '') 
                    for f in mmlu_test_dir.glob('*_test.csv')
                ]
            else:
                # Hard-coded MMLU category list.
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
    Convenience function for creating a cache manager.
    
    Args:
        cache_dir: Cache directory.
    
    Returns:
        FSMCacheManager instance.
    """
    return FSMCacheManager(cache_dir)


# Example usage
if __name__ == "__main__":
    # Create the cache manager.
    manager = create_cache_manager()
    
    # Example: save a GSM8K FSM.
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
    
    # Check caches.
    print(f"\n📊 Cache Status:")
    print(f"  GSM8K: {'✅' if manager.has_cache('gsm8k') else '❌'}")
    print(f"  MMLU (abstract_algebra): {'✅' if manager.has_cache('mmlu', 'abstract_algebra') else '❌'}")
    
    # List all caches.
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
