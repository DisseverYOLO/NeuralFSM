"""
实验1完整版V3：五目标损失优化 (Five-Objective Optimization)
Experiment 1 Complete V3: Five-Objective Loss with FSM-Aware Optimization

✅ 集成所有新功能：
1. 一个状态对应一个智能体 (State → Agent Mapping)
2. 状态转移条件自动匹配 (Transition Condition Matching) ✨
3. FSM验证和优化 (FSM Validation & Optimization) ✨
4. 循环避免机制 (Loop Prevention via FSMExecutor) ✨
5. LLM成本追踪 (LLM Cost Tracking) ✨
6. FSM状态描述动态优化 (FSM-Aware State Description Optimization) ✨
7. 最大转移次数惩罚 (Max Transitions Penalty) ✨

🎯 五目标组合损失 (Five-Objective Combined Loss)：
   L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ζ·L_max_transitions
   
   目标1: α·L_policy (策略梯度损失, α=1.0)
          - 优化端到端任务准确率
          - 通过强化学习优化状态转移策略
   
   目标2: β·L_transition (状态转移损失, β=0.3)
          - 学习最优状态转移序列
          - 鼓励TGN预测高概率给实际转移的状态
   
   目标3: γ·L_listener (监听路径损失, γ=0.2)
          - 优化智能体间通信路径
          - 学习最优监听关系
   
   目标4: δ·L_cost (LLM成本损失, δ集成在四目标框架中)
          - 最小化LLM API调用成本
          - 鼓励高效的推理路径
   
   目标5: ζ·L_max_transitions (最大转移惩罚, ζ=0.5)
          - 惩罚达到最大转移次数的执行
          - 鼓励更短、更高效的状态转移路径

🆚 版本对比：
- V1: 策略梯度 + MSE重构 (间接优化)
- V2: 策略梯度 + 状态转移 + 监听路径 (三目标)
- V3: V2 + LLM成本 + 最大转移惩罚 (五目标) ✨ LATEST

📝 详细说明见: 
- docs/STATE_DESCRIPTION_OPTIMIZATION.md (状态描述优化)
- docs/FSM_LOSS_FUNCTIONS_EXPLAINED.md (损失函数详解)
"""

import asyncio
import sys
import argparse
import json
import os
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from neural_fsm_mas.train_fsm_mas_v2 import FSMMultiAgentSystemTrainerV2
from neural_fsm_mas.core_integration import (
    EnhancedFSMIntegration,
    create_enhanced_integration,
    StateAgentCorrespondenceLearner
)  # ✨ 导入核心集成模块
from neural_fsm_mas.fsm_cache_manager import create_cache_manager  # ✨ FSM缓存
from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator  # ✨ FSM生成器


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='实验1完整版V2：新FSM架构 + 三目标优化'
    )
    
    # 基本参数
    parser.add_argument('--domains', type=str, nargs='+',
                       default=['gsm8k'],
                       help='训练领域 (mmlu, gsm8k, gpqa, gaia, humaneval, hotpotqa, alfworld, math)')
    parser.add_argument('--dataset_root', type=str, default='./datasets',
                       help='数据集根目录')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/experiment1_fsm_v2',
                       help='输出目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    
    # 四目标组合损失权重 ✨ 新增delta
    parser.add_argument('--policy_gradient_weight', type=float, default=1.0,
                       help='策略梯度损失权重 (α)')
    parser.add_argument('--transition_loss_weight', type=float, default=0.3,
                       help='状态转移损失权重 (β)')
    parser.add_argument('--listener_loss_weight', type=float, default=0.2,
                       help='监听路径损失权重 (γ)')
    parser.add_argument('--cost_loss_weight', type=float, default=0.001,
                       help='LLM成本损失权重 (δ) ✨ NEW')
    
    # FSM验证和执行参数 ✨ 新增
    parser.add_argument('--max_visits_per_state', type=int, default=3,
                       help='每个状态最大访问次数（循环避免）')
    parser.add_argument('--max_total_steps', type=int, default=20,
                       help='每个episode最大总步数')
    parser.add_argument('--max_transitions', type=int, default=8,
                       help='最大状态转移次数（每个问题的最大状态转移步数）')
    parser.add_argument('--auto_fix_fsm', action='store_true', default=True,
                       help='自动修复FSM问题（添加救援转移等）')
    parser.add_argument('--use_llm_for_conditions', action='store_true', default=False,
                       help='使用LLM生成缺失的转移条件')
    
    # TGN参数
    parser.add_argument('--memory_dim', type=int, default=128,
                       help='TGN记忆维度')
    parser.add_argument('--time_dim', type=int, default=32,
                       help='时间编码维度')
    parser.add_argument('--agent_embedding_dim', type=int, default=256,
                       help='智能体嵌入维度')
    parser.add_argument('--state_feature_dim', type=int, default=256,
                       help='状态特征维度')
    
    # LLM参数
    parser.add_argument('--llm_name', type=str, default='gpt-5-nano',
                       help='LLM模型名称')
    
    # FSM缓存参数 ✨ 新增
    parser.add_argument('--fsm_cache_dir', type=str, default='./fsm_cache',
                       help='FSM缓存目录')
    # FSM缓存默认启用，使用 --no-use-fsm-cache 可以关闭
    parser.add_argument('--no-use-fsm-cache', dest='use_fsm_cache', action='store_false',
                       help='禁用FSM缓存（强制重新生成，默认是启用缓存的）')
    parser.add_argument('--force_regenerate_fsm', action='store_true', default=False,
                       help='每次运行强制重新生成FSM（忽略已有缓存）')
    parser.add_argument('--generate_fsm_if_missing', action='store_true', default=True,
                       help='如果缓存不存在，自动生成FSM')
    parser.add_argument('--mmlu_use_category_fsm', action='store_true', default=False,
                       help='MMLU使用类别级FSM（每个类别一套FSM），默认False使用统一FSM')
    
    # ✨ 状态描述优化参数（新增）
    parser.add_argument('--enable_prompt_optimization', action='store_true', default=False,
                       help='启用状态描述动态优化（根据执行情况增强状态描述）')
    parser.add_argument('--max_transitions_threshold', type=int, default=3,
                       help='达到最大转移次数N次后触发效率优化（默认3）')
    parser.add_argument('--accuracy_enhancement_threshold', type=int, default=5,
                       help='失败N次后触发准确率优化（默认5）')
    parser.add_argument('--efficiency_enhancement_threshold', type=int, default=3,
                       help='低效执行N次后触发效率优化（默认3）')
    
    # ✨ 训练数据采样参数（新增）
    parser.add_argument('--train_samples', type=int, default=None,
                       help='从训练集中随机采样的样本数（例如100），不指定则使用全部')
    parser.add_argument('--train_only', action='store_true', default=False,
                       help='只进行训练，跳过验证和测试阶段')
    parser.add_argument('--full_train_data', action='store_true', default=False,
                       help='不划分验证/测试集：使用 100% 数据作为训练集（train_ratio=1,val_ratio=0,test_ratio=0）')

    # ✨ GAIA：按难度 level 过滤（1/2/3）；不指定则合并全部 level
    parser.add_argument('--gaia_level', type=int, default=None,
                       help='GAIA难度等级（1/2/3）。仅在 domains 包含 gaia 时生效')

    # ✨ 实验1消融开关（GSM8K/MATH等）
    parser.add_argument('--disable_transition_prediction', action='store_true', default=False,
                       help='消融：禁用TGN状态转移预测（不使用transition_probs采样下一状态）')
    parser.add_argument('--disable_comm_sampling', action='store_true', default=False,
                       help='消融：禁用通信路径采样（不使用listener_weights采样监听者）')
    parser.add_argument('--ablation', action='store_true', default=False,
                       help='消融实验模式：输出目录自动写入 ablation/<variant>/ 子目录，避免覆盖正常实验1结果')
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    # 将命令行指定的LLM名称写入环境变量，供所有LLM客户端统一使用
    if args.llm_name:
        os.environ["NEURALFSM_LLM_MODEL"] = args.llm_name
    
    print("="*80)
    print("🚀 实验1完整版V3：四目标优化 + 全功能集成")
    print("="*80)
    print("\n✨ 新功能（V3）:")
    print("  1. ✅ 四目标组合损失（策略梯度 + 状态转移 + 监听路径 + LLM成本）")
    print("  2. ✅ FSM验证和自动优化（可达性检查 + 救援转移）")
    print("  3. ✅ 转移条件自动匹配（执行时动态匹配条件）")
    print("  4. ✅ 循环避免机制（访问次数限制 + 步数限制）")
    print("  5. ✅ LLM成本实时追踪（12+模型价格清单）")
    print("\n📋 核心功能:")
    print("  • 一个状态对应一个智能体")
    print("  • TGN学习最优状态转移和通信路径")
    print("  • 支持所有数据集（MMLU、GSM8K、HumanEval、HotpotQA、ALFWorld、MATH、MBPP）✨")
    print("\n🆚 版本对比:")
    print("  • V1: α*策略梯度 + β*MSE重构 (间接优化)")
    print("  • V2: α*策略梯度 + β*状态转移 + γ*监听路径 (三目标)")
    print("  • V3: V2 + δ*LLM成本 + FSM验证 + 转移匹配 (四目标) ✨")
    print("\n📊 配置:")
    print(f"  • 训练领域: {args.domains}")
    print(f"  • 训练轮数: {args.num_epochs}")
    print(f"  • 批次大小: {args.batch_size}")
    print(f"  • 学习率: {args.learning_rate}")
    print(f"\n🎯 五目标损失权重 (Five-Objective Loss):")
    print(f"  L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ζ·L_max_transitions")
    print(f"  • 目标1 - 策略梯度 (α): {args.policy_gradient_weight}")
    print(f"  • 目标2 - 状态转移 (β): {args.transition_loss_weight}")
    print(f"  • 目标3 - 监听路径 (γ): {args.listener_loss_weight}")
    print(f"  • 目标4 - LLM成本   (δ): {args.cost_loss_weight}")
    print(f"  • 目标5 - 最大转移惩罚 (ζ): 0.5 (默认)")
    print(f"\n🛡️  FSM执行配置:")
    print(f"  • 每状态最大访问次数: {args.max_visits_per_state}")
    print(f"  • Episode最大步数: {args.max_total_steps}")
    print(f"  • 最大状态转移次数: {args.max_transitions}")
    print(f"  • 自动修复FSM: {args.auto_fix_fsm}")
    print(f"\n✨ 状态描述优化配置:")
    print(f"  • 启用优化: {args.enable_prompt_optimization}")
    if args.enable_prompt_optimization:
        print(f"  • 准确率优化: 失败 {args.accuracy_enhancement_threshold} 次后触发")
        print(f"  • 效率优化: 达到最大转移 {args.max_transitions_threshold} 次后触发")
    print(f"\n🤖 模型配置:")
    print(f"  • LLM模型: {args.llm_name}")
    print(f"  • 记忆维度: {args.memory_dim}")
    print(f"  • 时间编码维度: {args.time_dim}")
    print(f"  • 智能体嵌入维度: {args.agent_embedding_dim}")
    print(f"  • 状态特征维度: {args.state_feature_dim}")
    
    # ✨ 训练数据采样配置
    if args.train_samples or args.train_only or args.full_train_data:
        print(f"\n📋 数据采样配置:")
        if args.train_samples:
            print(f"  • 训练样本数: {args.train_samples}（随机采样）")
        if args.train_only:
            print(f"  • 模式: 只训练（跳过验证和测试）")
        if args.full_train_data:
            print(f"  • 划分: 100% 数据用于训练（不划分验证/测试）")
    print("="*80 + "\n")
    
    # 构建配置（V3版本：添加新参数）
    config = {
        # 基础训练参数
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        
        # 四目标损失权重 ✨
        'alpha': args.policy_gradient_weight,
        'beta': args.transition_loss_weight,
        'gamma': args.listener_loss_weight,
        'delta': args.cost_loss_weight,  # ✨ NEW: LLM成本权重
        'policy_gradient_weight': args.policy_gradient_weight,
        'transition_loss_weight': args.transition_loss_weight,
        'listener_loss_weight': args.listener_loss_weight,
        'cost_loss_weight': args.cost_loss_weight,  # ✨ NEW
        
        # FSM验证和执行参数 ✨
        'max_visits_per_state': args.max_visits_per_state,
        'max_total_steps': args.max_total_steps,
        'max_transitions': args.max_transitions,
        'auto_fix_fsm': args.auto_fix_fsm,
        'use_llm_for_conditions': args.use_llm_for_conditions,
        
        # TGN参数
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
        'llm_name': args.llm_name,
        'model_name': args.llm_name,  # ✨ 用于成本追踪和日志
        'agent_embedding_dim': args.agent_embedding_dim,
        'state_feature_dim': args.state_feature_dim,
        
        # 数据集参数
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        
        # FSM缓存参数 ✨
        'fsm_cache_dir': args.fsm_cache_dir,
        'use_fsm_cache': args.use_fsm_cache if hasattr(args, 'use_fsm_cache') and args.use_fsm_cache is not None else True,
        'force_regenerate_fsm': args.force_regenerate_fsm,
        'generate_fsm_if_missing': args.generate_fsm_if_missing,
        'mmlu_use_category_fsm': args.mmlu_use_category_fsm,
        
        # ✨ 状态描述优化配置（新增）
        'enable_prompt_optimization': args.enable_prompt_optimization,
        'max_transitions_threshold': args.max_transitions_threshold,
        'accuracy_enhancement_threshold': args.accuracy_enhancement_threshold,
        'efficiency_enhancement_threshold': args.efficiency_enhancement_threshold,
        
        # ✨ 训练数据采样配置（新增）
        'train_samples': args.train_samples,
        'train_only': args.train_only,
        'full_train_data': args.full_train_data,
        'gaia_level': args.gaia_level,

        # ✨ 消融开关
        'enable_transition_prediction': (not args.disable_transition_prediction),
        'enable_comm_sampling': (not args.disable_comm_sampling),
        'ablation': bool(args.ablation),
        
        # 成本估算参数 ✨
        'avg_input_tokens': 500,
        'avg_output_tokens': 200,
        'avg_calls_per_episode': 5
    }

    # ✨ 使用 100% 数据训练：覆盖默认划分比例，并避免与 train_samples 冲突
    if args.full_train_data:
        config['train_ratio'] = 1.0
        config['val_ratio'] = 0.0
        config['test_ratio'] = 0.0
        if args.train_samples:
            # 用户要求 100% 数据训练时，随机采样会违背该目标，直接忽略
            config['train_samples'] = None
            print("  ⚠️  已启用 --full_train_data，忽略 --train_samples（将使用全部样本训练）")
    
    # ✨ 初始化FSM缓存管理器（新增）
    print("\n📦 初始化FSM缓存管理器...")
    cache_manager = create_cache_manager(args.fsm_cache_dir)
    
    # ✨ 为每个数据集准备FSM（新增）
    print("\n🏗️  准备FSM配置...")
    fsm_generator = EnhancedFSMGenerator(use_azure=False)
    
    for domain in args.domains:
        print(f"\n  📋 检查 {domain.upper()} 的FSM...")
        
        if domain == 'mmlu' and args.mmlu_use_category_fsm:
            # MMLU特殊处理：需要为每个类别生成FSM
            print(f"    ℹ️  MMLU使用类别级FSM")
            
            # ✨ 检查并生成缺失的类别FSM
            if args.generate_fsm_if_missing:
                print(f"    🔍 检查MMLU类别FSM缓存...")
                from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor
                data_processor = UnifiedDataProcessor(args.dataset_root)
                mmlu_data = data_processor.load_dataset('mmlu', split='test')
                
                # 获取所有唯一的类别（从所有数据中获取，确保覆盖所有类别）
                # MMLU数据使用'subject'字段表示类别
                categories = set()
                for item in mmlu_data:
                    cat = item.get('subject', '')  # MMLU使用subject字段
                    if cat:
                        categories.add(cat)
                
                print(f"    📊 发现 {len(categories)} 个类别需要FSM")
                
                generated_count = 0
                cached_count = 0
                # ✨ 检查是否启用缓存
                use_cache = getattr(args, 'use_fsm_cache', True)
                for category in sorted(categories):
                    if use_cache and cache_manager.has_cache('mmlu', category):
                        cached_count += 1
                    elif args.generate_fsm_if_missing:
                        print(f"      🔨 生成类别 '{category}' 的FSM...")
                        try:
                            mas_config, cost = fsm_generator.generate_complete_mas(
                                dataset='mmlu',
                                task_description=f"MMLU {category} category questions",
                                save_path=None
                            )
                            cache_manager.save_fsm(
                                dataset='mmlu',
                                category=category,
                                fsm_config=mas_config.get('fsm', {}),
                                agents=mas_config.get('agents', []),
                                metadata={'generation_cost': cost, 'category': category}
                            )
                            generated_count += 1
                            print(f"        ✅ 完成 (成本: ${cost:.4f} USD)")
                        except Exception as e:
                            print(f"        ⚠️  生成失败: {e}")
                            print(f"        将在训练时使用默认FSM")
                
                print(f"    📊 统计: {cached_count} 个已缓存, {generated_count} 个新生成")
                if generated_count == 0 and cached_count == 0:
                    print(f"    ⚠️  未找到任何类别FSM，训练时将使用默认FSM")
            else:
                print(f"    ℹ️  自动生成已禁用，将在训练时按需加载或使用默认FSM")
        else:
            # 其他数据集：检查或生成FSM
            # ✨ 先检查是否启用缓存
            use_cache = getattr(args, 'use_fsm_cache', True)
            if use_cache and cache_manager.has_cache(domain):
                print(f"    ✅ 找到缓存FSM")
                cached = cache_manager.load_fsm(domain)
                print(f"      - 状态数: {cached['metadata']['num_states']}")
                print(f"      - 智能体数: {cached['metadata']['num_agents']}")
            elif args.generate_fsm_if_missing:
                print(f"    🔨 生成新FSM...")
                try:
                    mas_config, cost = fsm_generator.generate_complete_mas(
                        dataset=domain,
                        save_path=None  # 不保存到文件，只返回配置
                    )
                    # 保存到缓存
                    cache_manager.save_fsm(
                        dataset=domain,
                        fsm_config=mas_config.get('fsm', {}),
                        agents=mas_config.get('agents', []),
                        metadata={'generation_cost': cost}
                    )
                    print(f"    ✅ FSM生成完成 (成本: ${cost:.4f} USD)")
                except Exception as e:
                    print(f"    ⚠️  FSM生成失败: {e}")
                    print(f"    将在训练时使用默认配置")
            else:
                print(f"    ⚠️  未找到缓存且未启用自动生成，将使用默认配置")
    
    # ✨ 创建核心集成模块（新增）
    print("\n🔧 初始化核心集成模块...")
    integration = create_enhanced_integration(config)
    
    # 创建训练器（按 llm_name 拆分输出目录）
    # ✨ 输出目录结构：
    #    - GAIA: output_dir/level{N}/{llm_name}/
    #    - 消融实验: output_dir/ablation/{variant}/{llm_name}/
    #    - 正常实验: output_dir/{llm_name}/
    base_output = Path(args.output_dir)
    
    # ✨ GAIA level 区分（如果指定了 --gaia_level）
    if 'gaia' in args.domains and args.gaia_level is not None:
        base_output = base_output / f"level{args.gaia_level}"
    
    if args.ablation:
        variant_parts = []
        if args.disable_transition_prediction:
            variant_parts.append("no_trans")
        if args.disable_comm_sampling:
            variant_parts.append("no_comm")
        # “去掉提示优化”即 enable_prompt_optimization=False
        if not args.enable_prompt_optimization:
            variant_parts.append("no_promptopt")
        # Full ablation baseline: 开启提示优化且无其它禁用项
        if not variant_parts and args.enable_prompt_optimization:
            variant = "full"
        else:
            variant = "_".join(variant_parts) if variant_parts else "ablation"
        base_output = base_output / "ablation" / variant

    llm_specific_output = base_output / args.llm_name
    print("🔧 初始化FSM训练器...")
    trainer = FSMMultiAgentSystemTrainerV2(
        config=config,
        dataset_root=args.dataset_root,
        output_dir=llm_specific_output
    )
    
    # ✨ 注入FSM缓存管理器（新增）
    trainer.fsm_cache_manager = cache_manager
    trainer.fsm_generator = fsm_generator
    
    # ✨ 将集成模块注入训练器（新增）
    print("🔗 集成四目标损失和FSM验证功能...")
    integration.integrate_with_trainer(trainer)
    
    try:
        # 训练所有领域
        print(f"\n🎯 开始训练领域: {args.domains}")
        print("="*80)
        results = await trainer.train_all_domains(args.domains)
        
        # 打印详细结果
        print("\n" + "="*80)
        print("📊 训练结果摘要")
        print("="*80)
        
        for domain, result in results.items():
            print(f"\n🔸 领域: {domain}")
            print(f"  • 最佳准确率: {result['best_accuracy']:.4f} (Epoch {result['best_epoch']})")
            print(f"  • 最终训练准确率: {result['epoch_accuracies'][-1]:.4f}" if result['epoch_accuracies'] else "  • 最终训练准确率: N/A")
            print(f"  • 最终验证准确率: {result['validation_accuracies'][-1]:.4f}" if result['validation_accuracies'] else "  • 最终验证准确率: N/A")
            
            if result['epoch_losses']:
                print(f"  • 最终总损失: {result['epoch_losses'][-1]:.4f}")
                print(f"    ├─ 策略梯度损失: {result['epoch_policy_losses'][-1]:.4f}")
                print(f"    ├─ 状态转移损失: {result['epoch_transition_losses'][-1]:.4f}")
                print(f"    ├─ 监听路径损失: {result['epoch_listener_losses'][-1]:.4f}")
                # ✨ 新增：显示LLM成本损失
                if 'epoch_cost_losses' in result and result['epoch_cost_losses']:
                    print(f"    └─ LLM成本损失: {result['epoch_cost_losses'][-1]:.4f} ✨")
        
        # 保存总结
        summary = {
            'config': config,
            'domains': args.domains,
            'results': results,
            'best_domain': max(results.items(), key=lambda x: x[1]['best_accuracy'])[0],
            'average_best_accuracy': sum(r['best_accuracy'] for r in results.values()) / len(results)
        }
        
        summary_path = llm_specific_output / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化格式
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(make_serializable(summary), f, indent=2, ensure_ascii=False)
        
        # ✨ 输出状态描述优化摘要（新增）
        if args.enable_prompt_optimization:
            print("\n" + "="*80)
            print("✨ 状态描述优化摘要")
            print("="*80)
            for domain in args.domains:
                if hasattr(trainer, 'prompt_optimizers') and domain in trainer.prompt_optimizers:
                    optimizer = trainer.prompt_optimizers[domain]
                    summary = optimizer.get_performance_summary()
                    print(f"\n📊 {domain.upper()}:")
                    print(f"  • 总问题数: {summary['total_questions']}")
                    print(f"  • 整体准确率: {summary['overall_accuracy']:.2%}")
                    print(f"  • 达到最大转移: {summary['max_transitions_count']}次 ({summary['max_transitions_rate']:.1%})")
                    print(f"  • 被增强状态数: {summary['enhanced_states']} / {summary['total_states_tracked']}")
                    if summary['problem_states']:
                        print(f"\n  ⚠️  问题状态（按严重程度排序）:")
                        for state in summary['problem_states'][:5]:  # 只显示前5个
                            enhanced_mark = ""
                            if state['is_enhanced']:
                                if state['enhancement_type'] == 'accuracy':
                                    enhanced_mark = "✅ 已增强(准确率)"
                                elif state['enhancement_type'] == 'efficiency':
                                    enhanced_mark = "⚡ 已增强(效率)"
                                elif state['enhancement_type'] == 'combined':
                                    enhanced_mark = "🔥 已增强(准确率+效率)"
                            else:
                                enhanced_mark = "⏳ 待观察"
                            print(f"    • {state['state_name']}:")
                            print(f"        失败率: {state['failure_rate']:.1%} ({state['failure_count']}次)")
                            print(f"        低效率: {state['inefficiency_rate']:.1%} ({state['max_trans_count']}次)")
                            print(f"        状态: {enhanced_mark}")
        
        # ✨ 打印成本统计（修复：使用trainer的cost_tracker，而不是integration的）
        # 因为trainer.cost_tracker是全局追踪器，实际记录了LLM调用成本
        if hasattr(trainer, 'cost_tracker') and trainer.cost_tracker is not None:
            print("\n" + "="*80)
            print("💰 LLM成本统计")
            print("="*80)
            trainer.cost_tracker.print_statistics()
        elif hasattr(integration, 'cost_tracker'):
            # 回退到integration的cost_tracker
            print("\n" + "="*80)
            print("💰 LLM成本统计")
            print("="*80)
            integration.cost_tracker.print_statistics()
        
        print("\n" + "="*80)
        print("🎉 训练完成！")
        print("="*80)
        print(f"\n📈 最佳领域: {summary['best_domain']}")
        print(f"📊 平均最佳准确率: {summary['average_best_accuracy']:.4f}")
        print(f"💾 结果已保存到: {llm_specific_output}")
        print(f"📄 摘要文件: {summary_path}")
        print(f"\n💡 详细说明文档:")
        print(f"  • 损失函数: FSM_LOSS_FUNCTIONS_EXPLAINED.md")
        print(f"  • FSM设计: RANDOM_SAMPLED_FSM_DESIGN.md")
        print(f"  • 集成指南: FINAL_INTEGRATION_GUIDE.md")
        print(f"  • 核心集成: neural_fsm_mas/core_integration.py")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

