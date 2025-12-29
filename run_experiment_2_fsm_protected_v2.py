"""
实验2完整版V2：FSM模式 + 六目标优化 + 保护机制 + 攻击注入
Experiment 2 Complete V2: FSM Mode + Six-Objective Optimization + Protection + Attack Injection

🎯 六目标组合损失优化 (Six-Objective Optimization)：
   L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ε·L_protection + ζ·L_max_transitions
   
   目标1: α·L_policy (策略梯度损失, α=1.0)
          - 优化端到端任务准确率
   
   目标2: β·L_transition (状态转移损失, β=0.3)
          - 学习最优状态转移序列
   
   目标3: γ·L_listener (监听路径损失, γ=0.2)
          - 优化智能体间通信路径
   
   目标4: δ·L_cost (LLM成本损失, δ集成在四目标框架中)
          - 最小化LLM API调用成本
   
   目标5: ε·L_protection (保护损失, ε=0.3) ✨ 实验2核心
          - 防御异常攻击（频率/语义/中心性异常）
          - 惩罚"异常源→关键节点"的强消息传递
   
   目标6: ζ·L_max_transitions (最大转移惩罚, ζ=0.5)
          - 惩罚达到最大转移次数的执行
          - 鼓励更高效的状态转移路径

✨ 核心特性：
- FSM架构（每个状态对应一个智能体）
- 保护机制（ProtectedTGN，防御异常攻击）
- 攻击注入机制（频率/语义/混合攻击）
- TGN学习最优FSM结构和保护策略

🆚 与实验1的区别：
- 实验1: 五目标优化（α + β + γ + δ + ζ）+ FSM状态描述优化
- 实验2: 六目标优化（α + β + γ + δ + ε + ζ）+ 保护机制 + 攻击注入
  核心区别：增加保护损失 ε，专注于防御能力
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from neural_fsm_mas.train_fsm_mas_v2 import FSMMultiAgentSystemTrainerV2
from neural_fsm_mas.defense_mechanisms import ProtectedTGN
from neural_fsm_mas.fsm_cache_manager import create_cache_manager
from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='实验2完整版V2：FSM + 保护机制 + 提示词优化'
    )
    
    # 基本参数（继承自实验1）
    parser.add_argument('--domains', type=str, nargs='+',
                       default=['gsm8k'],
                       help='训练领域 (mmlu, gsm8k, gpqa, humaneval, hotpotqa, alfworld, math)')
    parser.add_argument('--dataset_root', type=str, default='./datasets',
                       help='数据集根目录')
    parser.add_argument('--output_dir', type=str, 
                       default=None,  # ✨ 默认为None，将自动根据配置生成
                       help='输出目录（留空则自动根据保护/攻击配置生成）')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--train_samples', type=int, default=None,
                       help='随机选取的训练样本数量（None表示使用全部数据）')
    parser.add_argument('--train_only', action='store_true', default=False,
                       help='只进行训练，跳过验证和测试阶段')
    
    # 五目标损失权重（实验2特色：四目标 + 保护损失）
    parser.add_argument('--policy_gradient_weight', type=float, default=1.0,
                       help='策略梯度损失权重 (α) - 任务准确率')
    parser.add_argument('--transition_loss_weight', type=float, default=0.3,
                       help='状态转移损失权重 (β) - 优化转移路径')
    parser.add_argument('--listener_loss_weight', type=float, default=0.2,
                       help='监听路径损失权重 (γ) - 优化通信效率')
    parser.add_argument('--cost_loss_weight', type=float, default=0.02,
                       help='LLM成本损失权重 (δ) - 降低推理成本')
    parser.add_argument('--protection_loss_weight', type=float, default=0.3,
                       help='保护损失权重 (ε) - 防御异常攻击 ✨ 实验2核心')
    
    # 保护机制参数
    parser.add_argument('--use_protection', action='store_true', default=False,
                       help='启用保护机制（使用此标志启用）')
    parser.add_argument('--no_protection', dest='use_protection', action='store_false',
                       help='禁用保护机制（不使用保护）')
    parser.add_argument('--w_betweenness', type=float, default=0.6,
                       help='Betweenness中心性权重')
    parser.add_argument('--w_pagerank', type=float, default=0.4,
                       help='PageRank权重')
    parser.add_argument('--lambda_freq', type=float, default=0.3,
                       help='频率异常检测权重')
    parser.add_argument('--lambda_semantic', type=float, default=0.7,
                       help='语义异常检测权重')
    parser.add_argument('--lambda_protect', type=float, default=0.3,
                       help='保护损失在ProtectedTGN内部的权重（建议与protection_loss_weight一致）')
    parser.add_argument('--lambda_reg', type=float, default=0.01,
                       help='正则化损失权重')
    parser.add_argument('--learnable_weights', action='store_true', default=True,
                       help='使用可学习的MLP消息权重计算器')
    
    # ✨ 攻击注入参数（新增）
    parser.add_argument('--enable_attack', action='store_true', default=False,
                       help='启用攻击注入（用于测试防御效果）')
    parser.add_argument('--attack_type', type=str, default='selfish',
                       choices=['frequency', 'semantic', 'selfish'],
                       help='攻击类型：frequency(频率), semantic(语义), selfish(自私)')
    parser.add_argument('--attack_ratio', type=float, default=0.3,
                       help='被攻击节点的比例（0.3表示30%%节点被攻击）')
    parser.add_argument('--attack_strength', type=float, default=1.5,
                       help='攻击强度（1.0为正常，>1.0为增强攻击）')
    parser.add_argument('--attack_frequency', type=str, default='always',
                       choices=['always', 'periodic', 'random'],
                       help='攻击频率：always(每步), periodic(周期性), random(随机)')
    parser.add_argument('--attack_seed', type=int, default=None,
                       help='攻击目标选择的随机种子（不传则每次运行随机；传入整数则可复现，例如42）')
    parser.add_argument('--attack_start_epoch', type=int, default=0,
                       help='开始攻击的epoch（可用于测试训练稳定性）')
    
    # FSM参数（与实验1保持一致）
    parser.add_argument('--max_transitions', type=int, default=8,
                       help='最大状态转移次数（每个问题的最大状态转移步数）')
    parser.add_argument('--mmlu_use_category_fsm', action='store_true', default=False,
                       help='MMLU使用类别级FSM（默认关闭，使用统一FSM）')
    
    # FSM缓存参数（与实验1保持一致）
    parser.add_argument('--fsm_cache_dir', type=str, default='./fsm_cache',
                       help='FSM缓存目录')
    # FSM缓存默认启用，使用 --no-use-fsm-cache 可以关闭
    parser.add_argument('--no-use-fsm-cache', dest='use_fsm_cache', action='store_false',
                       help='禁用FSM缓存（强制重新生成，默认是启用缓存的）')
    parser.add_argument('--generate_fsm_if_missing', action='store_true', default=True,
                       help='如果缓存不存在，自动生成FSM')
    
    # 注意：实验2不使用提示词优化（专注于保护机制）
    # 提示词优化功能仅在实验1中可用
    
    # TGN参数
    parser.add_argument('--memory_dim', type=int, default=128,
                       help='TGN记忆维度')
    parser.add_argument('--time_dim', type=int, default=32,
                       help='时间编码维度')
    parser.add_argument('--llm_name', type=str, default='gpt-5-nano',
                       help='LLM模型名称')
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()

    # ✨ 若启用攻击但未指定 attack_seed，则为本次运行生成随机seed（用于攻击目标选择 + 输出目录区分）
    if args.enable_attack and args.attack_seed is None:
        import secrets
        args.attack_seed = secrets.randbits(31)  # int，避免过大影响路径
        print(f"🎲 未指定 --attack_seed，已为本次运行随机生成 attack_seed={args.attack_seed}")
    
    # ✨ 自动生成输出目录（根据保护/攻击配置）
    if args.output_dir is None:
        # 构建描述性目录名
        dir_parts = ['./results/experiment2']
        
        # 模型名（清理特殊字符）
        model_name = args.llm_name.replace('/', '-').replace(':', '-')
        dir_parts.append(model_name)
        
        # 保护状态
        if args.use_protection:
            dir_parts.append('protected')
        else:
            dir_parts.append('unprotected')
        
        # 攻击状态
        if args.enable_attack:
            dir_parts.append(f'attack_{args.attack_type}')
            dir_parts.append(f'ratio{int(args.attack_ratio*100)}')
            dir_parts.append(f'str{args.attack_strength}')
            dir_parts.append(f'freq_{args.attack_frequency}')
            # ✨ 加入随机种子（用于区分可复现实验的输出目录）
            dir_parts.append(f'seed{args.attack_seed}')
        else:
            dir_parts.append('no_attack')
        
        # 领域
        dir_parts.append('_'.join(args.domains))
        
        args.output_dir = '_'.join(dir_parts)
    
    print("="*80)
    print("🛡️  实验2完整版V2：FSM + 保护机制 + 提示词优化")
    print("="*80)
    print("\n✨ 核心功能:")
    print("  1. ✅ FSM架构（每个状态对应一个智能体）")
    print("  2. ✅ 六目标组合损失（4个损失函数 + 2个惩罚项）")
    print("  3. ✅ 保护机制（ProtectedTGN，防御异常攻击）")
    print("  4. ✅ 攻击注入机制（频率/语义/混合攻击）")
    print("  5. ✅ TGN学习最优FSM结构和保护策略")
    print("\n📊 配置:")
    print(f"  • 训练领域: {args.domains}")
    print(f"\n🎯 六目标优化 (4个损失函数 + 2个惩罚项):")
    print(f"  L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ε·L_protection + ζ·P_max_trans")
    print(f"\n  【4个损失函数】:")
    print(f"  • L_policy    (α={args.policy_gradient_weight}): 策略梯度损失 - 任务准确率")
    print(f"  • L_transition(β={args.transition_loss_weight}): 状态转移损失 - 优化转移路径")
    print(f"  • L_listener  (γ={args.listener_loss_weight}): 监听路径损失 - 优化通信效率")
    print(f"  • L_cost      (δ={args.cost_loss_weight}): LLM成本损失 - 降低推理成本")
    print(f"\n  【实验2专属损失函数】:")
    print(f"  • L_protection(ε={args.protection_loss_weight}): 保护损失 - 防御异常攻击 ✨")
    print(f"\n  【惩罚项】:")
    print(f"  • P_max_trans (ζ=0.5): 最大转移惩罚 - 鼓励高效路径")
    print(f"\n🛡️  机制状态:")
    print(f"  • 保护机制: {'✅ 已启用' if args.use_protection else '❌ 未启用'}")
    print(f"  • 攻击注入: {'✅ 已启用' if args.enable_attack else '❌ 未启用'}")
    if args.enable_attack:
        print(f"    ├─ 攻击类型: {args.attack_type}")
        print(f"    ├─ 攻击比例: {args.attack_ratio*100:.0f}%")
        print(f"    ├─ 攻击强度: {args.attack_strength}")
        print(f"    ├─ 攻击频率: {args.attack_frequency}")
        print(f"    └─ 开始epoch: {args.attack_start_epoch}")
    
    # ✨ 显示采样配置
    if args.train_samples:
        print(f"  • 训练样本数: {args.train_samples} (随机采样)")
    if args.train_only:
        print(f"  • 训练模式: 仅训练（跳过验证和测试）")
    
    # ✨ 显示输出目录
    print(f"\n📁 输出目录: {args.output_dir}")
    print("="*80 + "\n")
    
    # 构建配置
    config = {
        # 基础训练参数
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        
        # 五目标损失权重（实验2特色：四目标 + 保护损失）
        'alpha': args.policy_gradient_weight,
        'beta': args.transition_loss_weight,
        'gamma': args.listener_loss_weight,
        'delta': args.cost_loss_weight,
        'epsilon': args.protection_loss_weight,  # ✨ 保护损失（实验2核心）
        'policy_gradient_weight': args.policy_gradient_weight,
        'transition_loss_weight': args.transition_loss_weight,
        'listener_loss_weight': args.listener_loss_weight,
        'cost_loss_weight': args.cost_loss_weight,
        'protection_loss_weight': args.protection_loss_weight,  # ✨ 新增
        
        # 保护机制配置
        'use_protection': args.use_protection,
        'protection': {
            'w_betweenness': args.w_betweenness,
            'w_pagerank': args.w_pagerank,
            'lambda_freq': args.lambda_freq,
            'lambda_semantic': args.lambda_semantic,
            'lambda_protect': args.lambda_protect,
            'lambda_reg': args.lambda_reg,
            'learnable_weights': args.learnable_weights,
            'weight_hidden_dim': 64
        },
        
        # ✨ 攻击注入配置（新增）
        'enable_attack': args.enable_attack,
        'attack': {
            'attack_type': args.attack_type,
            'attack_ratio': args.attack_ratio,
            'attack_strength': args.attack_strength,
            'attack_frequency': args.attack_frequency,
            'attack_seed': args.attack_seed,
            'attack_start_epoch': args.attack_start_epoch
        },
        
        # FSM参数
        'max_transitions': args.max_transitions,
        'mmlu_use_category_fsm': args.mmlu_use_category_fsm,
        
        # FSM缓存参数
        'fsm_cache_dir': args.fsm_cache_dir,
        'generate_fsm_if_missing': args.generate_fsm_if_missing,
        'use_fsm_cache': getattr(args, 'use_fsm_cache', True),
        
        # 注意：实验2不使用提示词优化（专注于保护机制和攻击注入）
        'enable_prompt_optimization': False,
        
        # TGN参数
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
        'llm_name': args.llm_name,
        'agent_embedding_dim': 256,
        'state_feature_dim': 256,
        
        # 数据集参数
        'train_ratio': 1.0 if args.train_only else 0.7,  # 只训练时使用全部数据
        'val_ratio': 0.0 if args.train_only else 0.15,
        'test_ratio': 0.0 if args.train_only else 0.15,
        'random_seed': 42,
        
        # ✨ 采样参数（新增）
        'train_samples': args.train_samples,  # 随机选取的训练样本数
        'train_only': args.train_only,  # 只训练不验证/测试
    }
    
    # 初始化FSM缓存管理器
    print("\n📦 初始化FSM缓存管理器...")
    cache_manager = create_cache_manager(args.fsm_cache_dir)
    
    # 准备FSM配置
    print("\n🏗️  准备FSM配置...")
    fsm_generator = EnhancedFSMGenerator(use_azure=False)
    
    for domain in args.domains:
        print(f"\n  📋 检查 {domain.upper()} 的FSM...")
        if domain == 'mmlu' and args.mmlu_use_category_fsm:
            print(f"    ℹ️  MMLU使用类别级FSM，将在训练时动态加载")
        else:
            if cache_manager.has_cache(domain):
                print(f"    ✅ 找到缓存FSM")
            elif args.generate_fsm_if_missing:
                print(f"    🔨 生成新FSM...")
                try:
                    mas_config, cost = fsm_generator.generate_complete_mas(
                        dataset=domain,
                        save_path=None
                    )
                    cache_manager.save_fsm(
                        dataset=domain,
                        fsm_config=mas_config.get('fsm', {}),
                        agents=mas_config.get('agents', []),
                        metadata={'generation_cost': cost}
                    )
                    print(f"    ✅ FSM生成完成 (成本: ${cost:.4f})")
                except Exception as e:
                    print(f"    ⚠️  FSM生成失败: {e}")
    
    # 创建训练器
    print("\n🔧 初始化FSM+保护训练器...")
    trainer = FSMMultiAgentSystemTrainerV2(
        config=config,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir
    )
    
    # 注入FSM缓存管理器
    trainer.fsm_cache_manager = cache_manager
    trainer.fsm_generator = fsm_generator
    
    # ✨ 保护机制集成说明（新增）
    if args.use_protection:
        print("\n🛡️  保护机制将在创建FSM系统时自动集成到FSM-TGN")
        print("    （在train_fsm_mas_v2.py的create_domain_fsm_system中完成）")
    
    # 开始训练
    print("\n🎯 开始训练...")
    results = await trainer.train_all_domains(domains=args.domains)
    
    # 输出结果
    print("\n" + "="*80)
    print("📊 训练结果汇总")
    print("="*80)
    
    for domain, domain_results in results.items():
        print(f"\n📚 {domain.upper()}:")
        print(f"  • 最佳准确率: {domain_results.get('best_accuracy', 0.0):.4f}")
        print(f"  • 最佳轮次: {domain_results.get('best_epoch', 0)}")
    
    # ✨ 打印成本统计
    if hasattr(trainer, 'cost_tracker') and trainer.cost_tracker is not None:
        print("\n" + "="*80)
        print("💰 LLM成本统计")
        print("="*80)
        trainer.cost_tracker.print_statistics()
    
    # 保存结果
    summary = {
        'config': config,
        'domains': args.domains,
        'results': results,
        'best_domain': max(results.items(), key=lambda x: x[1]['best_accuracy'])[0] if results else None,
        'average_best_accuracy': sum(r['best_accuracy'] for r in results.values()) / len(results) if results else 0.0
    }
    
    summary_path = Path(args.output_dir) / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    print(f"\n✅ 结果已保存到: {summary_path}")
    print("\n" + "="*80)
    print("🎉 训练完成！")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

