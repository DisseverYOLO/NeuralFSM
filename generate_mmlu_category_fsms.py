"""
MMLU类别级FSM生成脚本
MMLU Category-Level FSM Generation Script

功能:
1. 为MMLU的57个类别分别生成专属FSM
2. 每个类别生成一套状态、智能体和转移描述
3. 保存到fsm_cache/mmlu/{category_name}/
4. 支持断点续传（跳过已生成的类别）
5. 记录生成成本和元数据

使用方法:
python generate_mmlu_category_fsms.py --output_dir ./fsm_cache/mmlu --use_azure False
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator
from neural_fsm_mas.fsm_cache_manager import create_cache_manager


# MMLU的57个类别
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


def get_category_description(category: str) -> str:
    """
    获取类别的描述（用于FSM生成）
    
    Args:
        category: 类别名称
    
    Returns:
        类别描述
    """
    descriptions = {
        # STEM
        'abstract_algebra': 'Abstract algebra problems involving groups, rings, fields, and algebraic structures',
        'astronomy': 'Astronomy questions about celestial bodies, cosmology, and astrophysics',
        'college_biology': 'College-level biology covering genetics, evolution, ecology, and molecular biology',
        'college_chemistry': 'College-level chemistry including organic, inorganic, and physical chemistry',
        'college_computer_science': 'College-level computer science covering algorithms, data structures, and systems',
        'college_mathematics': 'College-level mathematics including calculus, linear algebra, and analysis',
        'college_physics': 'College-level physics covering mechanics, electromagnetism, and quantum physics',
        'machine_learning': 'Machine learning concepts, algorithms, and applications',
        
        # Humanities
        'philosophy': 'Philosophical questions about ethics, metaphysics, epistemology, and logic',
        'formal_logic': 'Formal logic problems involving propositional and predicate logic',
        'high_school_european_history': 'European history from ancient times to modern era',
        'high_school_us_history': 'United States history from colonial period to present',
        
        # Social Sciences
        'high_school_psychology': 'Psychology concepts including cognitive, developmental, and social psychology',
        'high_school_macroeconomics': 'Macroeconomics covering GDP, inflation, monetary policy, and fiscal policy',
        'high_school_microeconomics': 'Microeconomics covering supply, demand, market structures, and consumer behavior',
        
        # Other
        'clinical_knowledge': 'Clinical medicine knowledge including diagnosis, treatment, and patient care',
        'professional_medicine': 'Professional medical knowledge for healthcare practitioners',
    }
    
    return descriptions.get(category, f'{category.replace("_", " ").title()} questions requiring domain-specific knowledge')


async def generate_category_fsm(category: str,
                                fsm_generator: EnhancedFSMGenerator,
                                cache_manager,
                                output_dir: Path,
                                use_azure: bool = False) -> Dict[str, Any]:
    """
    为单个类别生成FSM
    
    Args:
        category: 类别名称
        fsm_generator: FSM生成器
        cache_manager: FSM缓存管理器
        output_dir: 输出目录
        use_azure: 是否使用Azure OpenAI
    
    Returns:
        生成结果字典
    """
    print(f"\n{'='*80}")
    print(f"📋 生成类别: {category}")
    print(f"{'='*80}")
    
    # 检查是否已存在
    if cache_manager.has_cache('mmlu', category):
        print(f"  ✅ 已存在缓存，跳过生成")
        cached = cache_manager.load_fsm('mmlu', category)
        return {
            'category': category,
            'status': 'cached',
            'cost': cached['metadata'].get('generation_cost', 0.0),
            'num_states': cached['metadata']['num_states'],
            'num_agents': cached['metadata']['num_agents'],
            'created_at': cached['metadata'].get('created_at', 'unknown')
        }
    
    try:
        # 获取类别描述
        category_desc = get_category_description(category)
        
        # 生成FSM（使用MMLU模板，但添加类别特定描述）
        print(f"  🔨 开始生成FSM...")
        start_time = time.time()
        
        mas_config, cost = fsm_generator.generate_complete_mas(
            dataset='mmlu',
            save_path=None,  # 不保存到文件，手动保存到缓存
            task_description=f"""
MMLU {category.replace('_', ' ').title()} Category

This is a specific category within the MMLU dataset focusing on:
{category_desc}

The questions in this category require:
- Deep domain-specific knowledge
- Specialized reasoning patterns
- Category-specific problem-solving approaches

Generate FSM states and agents optimized for this specific academic domain.
"""
        )
        
        generation_time = time.time() - start_time
        
        # 保存到缓存
        cache_manager.save_fsm(
            dataset='mmlu',
            fsm_config=mas_config.get('fsm', {}),
            agents=mas_config.get('agents', []),
            category=category,
            metadata={
                'generation_cost': cost,
                'generation_time': generation_time,
                'category_description': category_desc
            }
        )
        
        print(f"  ✅ 生成完成")
        print(f"     - 成本: ${cost:.4f}")
        print(f"     - 时间: {generation_time:.2f}秒")
        print(f"     - 状态数: {len(mas_config.get('fsm', {}).get('states', []))}")
        print(f"     - 智能体数: {len(mas_config.get('agents', []))}")
        
        return {
            'category': category,
            'status': 'success',
            'cost': cost,
            'generation_time': generation_time,
            'num_states': len(mas_config.get('fsm', {}).get('states', [])),
            'num_agents': len(mas_config.get('agents', [])),
            'created_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'category': category,
            'status': 'failed',
            'error': str(e),
            'cost': 0.0
        }


async def generate_all_category_fsms(categories: Optional[List[str]] = None,
                                     output_dir: str = "./fsm_cache/mmlu",
                                     use_azure: bool = False,
                                     skip_existing: bool = True) -> Dict[str, Any]:
    """
    为所有MMLU类别生成FSM
    
    Args:
        categories: 要生成的类别列表（None表示所有57个）
        output_dir: 输出目录
        use_azure: 是否使用Azure OpenAI
        skip_existing: 是否跳过已存在的FSM
    
    Returns:
        生成结果摘要
    """
    if categories is None:
        categories = MMLU_CATEGORIES
    
    print("="*80)
    print("🚀 MMLU类别级FSM生成")
    print("="*80)
    print(f"\n📊 统计:")
    print(f"  - 总类别数: {len(categories)}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 使用Azure: {use_azure}")
    print(f"  - 跳过已存在: {skip_existing}")
    print("="*80)
    
    # 初始化
    cache_manager = create_cache_manager(output_dir)
    fsm_generator = EnhancedFSMGenerator(use_azure=use_azure)
    
    # 检查已存在的类别
    existing_categories = []
    if skip_existing:
        for category in categories:
            if cache_manager.has_cache('mmlu', category):
                existing_categories.append(category)
        
        if existing_categories:
            print(f"\n📦 发现 {len(existing_categories)} 个已存在的FSM:")
            for cat in existing_categories[:10]:
                print(f"  - {cat}")
            if len(existing_categories) > 10:
                print(f"  ... 还有 {len(existing_categories) - 10} 个")
            
            categories = [c for c in categories if c not in existing_categories]
            print(f"\n🔄 将生成 {len(categories)} 个新FSM")
    
    # 生成结果
    results = {
        'total_categories': len(MMLU_CATEGORIES),
        'to_generate': len(categories),
        'existing': len(existing_categories),
        'generated': 0,
        'failed': 0,
        'cached': len(existing_categories),
        'total_cost': 0.0,
        'categories': {}
    }
    
    # 逐个生成（避免并发导致API限制）
    for idx, category in enumerate(categories, 1):
        print(f"\n[{idx}/{len(categories)}] ", end="")
        
        result = await generate_category_fsm(
            category=category,
            fsm_generator=fsm_generator,
            cache_manager=cache_manager,
            output_dir=Path(output_dir),
            use_azure=use_azure
        )
        
        results['categories'][category] = result
        
        if result['status'] == 'success':
            results['generated'] += 1
            results['total_cost'] += result.get('cost', 0.0)
        elif result['status'] == 'cached':
            results['cached'] += 1
            results['total_cost'] += result.get('cost', 0.0)
        else:
            results['failed'] += 1
        
        # 短暂延迟，避免API限制
        if idx < len(categories):
            await asyncio.sleep(1)
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 生成总结")
    print("="*80)
    print(f"  总类别数: {results['total_categories']}")
    print(f"  已存在: {results['cached']}")
    print(f"  新生成: {results['generated']}")
    print(f"  失败: {results['failed']}")
    print(f"  总成本: ${results['total_cost']:.4f}")
    print("="*80)
    
    # 保存结果摘要
    summary_path = Path(output_dir) / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果摘要已保存到: {summary_path}")
    
    return results


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='为MMLU的57个类别生成专属FSM'
    )
    
    parser.add_argument('--output_dir', type=str, default='./fsm_cache/mmlu',
                       help='FSM缓存输出目录')
    parser.add_argument('--use_azure', action='store_true', default=False,
                       help='使用Azure OpenAI（否则使用OpenAI）')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='跳过已存在的FSM（断点续传）')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                       help='指定要生成的类别（默认：所有57个）')
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    results = await generate_all_category_fsms(
        categories=args.categories,
        output_dir=args.output_dir,
        use_azure=args.use_azure,
        skip_existing=args.skip_existing
    )
    
    # 打印失败类别（如果有）
    failed_categories = [
        cat for cat, res in results['categories'].items() 
        if res.get('status') == 'failed'
    ]
    
    if failed_categories:
        print(f"\n⚠️  失败的类别 ({len(failed_categories)}):")
        for cat in failed_categories:
            error = results['categories'][cat].get('error', 'Unknown error')
            print(f"  - {cat}: {error}")
        print(f"\n💡 可以重新运行脚本，只生成失败的类别:")
        print(f"   python generate_mmlu_category_fsms.py --categories {' '.join(failed_categories)}")


if __name__ == "__main__":
    asyncio.run(main())

