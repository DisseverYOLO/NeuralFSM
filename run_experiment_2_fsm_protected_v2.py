import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from neural_fsm_mas.train_fsm_mas_v2 import FSMMultiAgentSystemTrainerV2
from neural_fsm_mas.defense_mechanisms import ProtectedTGN
from neural_fsm_mas.fsm_cache_manager import create_cache_manager
from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment 2 Complete V2: FSM + protection mechanisms + prompt optimization'
    )
    
    # Basic arguments (inherited from Experiment 1)
    parser.add_argument('--domains', type=str, nargs='+',
                       default=['gsm8k'],
                       help='Training domains (mmlu, gsm8k, gpqa, humaneval, hotpotqa, alfworld, math)')
    parser.add_argument('--dataset_root', type=str, default='./datasets',
                       help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, 
                       default=None,  # ✨ Default is None; it will be generated automatically from the configuration
                       help='Output directory (leave empty to auto-generate from protection/attack settings)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=None,
                       help='Number of randomly selected training samples (None means use all data)')
    parser.add_argument('--train_only', action='store_true', default=False,
                       help='Train only, skipping validation and testing')
    
    # Five-objective loss weights (Experiment 2 feature: four objectives + protection loss)
    parser.add_argument('--policy_gradient_weight', type=float, default=1.0,
                       help='Policy gradient loss weight (α) - task accuracy')
    parser.add_argument('--transition_loss_weight', type=float, default=0.3,
                       help='State transition loss weight (β) - optimize transition paths')
    parser.add_argument('--listener_loss_weight', type=float, default=0.2,
                       help='Listener path loss weight (γ) - optimize communication efficiency')
    parser.add_argument('--cost_loss_weight', type=float, default=0.02,
                       help='LLM cost loss weight (δ) - reduce reasoning cost')
    parser.add_argument('--protection_loss_weight', type=float, default=0.3,
                       help='Protection loss weight (ε) - defend against anomalous attacks ✨ Core of Experiment 2')
    
    # Protection mechanism arguments
    parser.add_argument('--use_protection', action='store_true', default=False,
                       help='Enable protection mechanisms (use this flag to enable)')
    parser.add_argument('--no_protection', dest='use_protection', action='store_false',
                       help='Disable protection mechanisms')
    parser.add_argument('--w_betweenness', type=float, default=0.6,
                       help='Betweenness centrality weight')
    parser.add_argument('--w_pagerank', type=float, default=0.4,
                       help='PageRank weight')
    parser.add_argument('--lambda_freq', type=float, default=0.3,
                       help='Frequency anomaly detection weight')
    parser.add_argument('--lambda_semantic', type=float, default=0.7,
                       help='Semantic anomaly detection weight')
    parser.add_argument('--lambda_protect', type=float, default=0.3,
                       help='Internal protection loss weight in ProtectedTGN (recommended to match protection_loss_weight)')
    parser.add_argument('--lambda_reg', type=float, default=0.01,
                       help='Regularization loss weight')
    parser.add_argument('--learnable_weights', action='store_true', default=True,
                       help='Use a learnable MLP message weight calculator')
    
    # ✨ Attack injection arguments (new)
    parser.add_argument('--enable_attack', action='store_true', default=False,
                       help='Enable attack injection (used to test defense effectiveness)')
    parser.add_argument('--attack_type', type=str, default='selfish',
                       choices=['frequency', 'semantic', 'selfish'],
                       help='Attack type: frequency, semantic, selfish')
    parser.add_argument('--attack_ratio', type=float, default=0.3,
                       help='Ratio of attacked nodes (0.3 means 30%% of nodes are attacked)')
    parser.add_argument('--attack_strength', type=float, default=1.5,
                       help='Attack strength (1.0 is normal, >1.0 is stronger attack)')
    parser.add_argument('--attack_frequency', type=str, default='always',
                       choices=['always', 'periodic', 'random'],
                       help='Attack frequency: always (every step), periodic, random')
    parser.add_argument('--attack_seed', type=int, default=None,
                       help='Random seed for selecting attack targets (random each run if omitted; reproducible if set, e.g. 42)')
    parser.add_argument('--attack_start_epoch', type=int, default=0,
                       help='Epoch to start attacks (useful for testing training stability)')
    
    # FSM arguments (kept consistent with Experiment 1)
    parser.add_argument('--max_transitions', type=int, default=8,
                       help='Maximum number of state transitions (maximum transition steps per question)')
    parser.add_argument('--mmlu_use_category_fsm', action='store_true', default=False,
                       help='Use category-level FSM for MMLU (disabled by default; uses a unified FSM otherwise)')
    
    # FSM cache arguments (kept consistent with Experiment 1)
    parser.add_argument('--fsm_cache_dir', type=str, default='./fsm_cache',
                       help='FSM cache directory')
    # FSM cache is enabled by default; use --no-use-fsm-cache to disable it
    parser.add_argument('--no-use-fsm-cache', dest='use_fsm_cache', action='store_false',
                       help='Disable FSM cache (force regeneration; cache is enabled by default)')
    parser.add_argument('--generate_fsm_if_missing', action='store_true', default=True,
                       help='Automatically generate FSM if cache is missing')
    
    # Note: Experiment 2 does not use prompt optimization (it focuses on protection mechanisms)
    # Prompt optimization is only available in Experiment 1
    
    # TGN arguments
    parser.add_argument('--memory_dim', type=int, default=128,
                       help='TGN memory dimension')
    parser.add_argument('--time_dim', type=int, default=32,
                       help='Time encoding dimension')
    parser.add_argument('--llm_name', type=str, default='gpt-5-nano',
                       help='LLM model name')
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()

    # ✨ If attacks are enabled but attack_seed is not specified, generate a random seed for this run
    if args.enable_attack and args.attack_seed is None:
        import secrets
        args.attack_seed = secrets.randbits(31)  # int, avoid overly large values affecting paths
        print(f"🎲 --attack_seed was not specified; generated random attack_seed={args.attack_seed} for this run")
    
    # ✨ Automatically generate the output directory based on protection/attack settings
    if args.output_dir is None:
        # Build a descriptive directory name
        dir_parts = ['./results/experiment2']
        
        # Model name (sanitize special characters)
        model_name = args.llm_name.replace('/', '-').replace(':', '-')
        dir_parts.append(model_name)
        
        # Protection status
        if args.use_protection:
            dir_parts.append('protected')
        else:
            dir_parts.append('unprotected')
        
        # Attack status
        if args.enable_attack:
            dir_parts.append(f'attack_{args.attack_type}')
            dir_parts.append(f'ratio{int(args.attack_ratio*100)}')
            dir_parts.append(f'str{args.attack_strength}')
            dir_parts.append(f'freq_{args.attack_frequency}')
            # ✨ Include the random seed to distinguish reproducible experiment output directories
            dir_parts.append(f'seed{args.attack_seed}')
        else:
            dir_parts.append('no_attack')
        
        # Domains
        dir_parts.append('_'.join(args.domains))
        
        args.output_dir = '_'.join(dir_parts)
    
    print("="*80)
    print("🛡️  Experiment 2 Complete V2: FSM + protection mechanisms + prompt optimization")
    print("="*80)
    print("\n✨ Core features:")
    print("  1. ✅ FSM architecture (each state corresponds to one agent)")
    print("  2. ✅ Six-objective combined loss (4 loss terms + 2 penalty terms)")
    print("  3. ✅ Protection mechanisms (ProtectedTGN, defending against anomalous attacks)")
    print("  4. ✅ Attack injection mechanisms (frequency/semantic/hybrid attacks)")
    print("  5. ✅ TGN learns the optimal FSM structure and protection strategy")
    print("\n📊 Configuration:")
    print(f"  • Training domains: {args.domains}")
    print(f"\n🎯 Six-objective optimization (4 loss terms + 2 penalty terms):")
    print(f"  L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ε·L_protection + ζ·P_max_trans")
    print(f"\n  [4 loss terms]:")
    print(f"  • L_policy    (α={args.policy_gradient_weight}): policy gradient loss - task accuracy")
    print(f"  • L_transition(β={args.transition_loss_weight}): state transition loss - optimize transition paths")
    print(f"  • L_listener  (γ={args.listener_loss_weight}): listener path loss - optimize communication efficiency")
    print(f"  • L_cost      (δ={args.cost_loss_weight}): LLM cost loss - reduce reasoning cost")
    print(f"\n  [Experiment 2 specific loss term]:")
    print(f"  • L_protection(ε={args.protection_loss_weight}): protection loss - defend against anomalous attacks ✨")
    print(f"\n  [Penalty term]:")
    print(f"  • P_max_trans (ζ=0.5): max transition penalty - encourage efficient paths")
    print(f"\n🛡️  Mechanism status:")
    print(f"  • Protection mechanisms: {'✅ Enabled' if args.use_protection else '❌ Disabled'}")
    print(f"  • Attack injection: {'✅ Enabled' if args.enable_attack else '❌ Disabled'}")
    if args.enable_attack:
        print(f"    ├─ Attack type: {args.attack_type}")
        print(f"    ├─ Attack ratio: {args.attack_ratio*100:.0f}%")
        print(f"    ├─ Attack strength: {args.attack_strength}")
        print(f"    ├─ Attack frequency: {args.attack_frequency}")
        print(f"    └─ Start epoch: {args.attack_start_epoch}")
    
    # ✨ Show sampling configuration
    if args.train_samples:
        print(f"  • Training samples: {args.train_samples} (random sampling)")
    if args.train_only:
        print(f"  • Training mode: train only (skip validation and testing)")
    
    # ✨ Show the output directory
    print(f"\n📁 Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Build configuration
    config = {
        # Basic training arguments
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        
        # Five-objective loss weights (Experiment 2 feature: four objectives + protection loss)
        'alpha': args.policy_gradient_weight,
        'beta': args.transition_loss_weight,
        'gamma': args.listener_loss_weight,
        'delta': args.cost_loss_weight,
        'epsilon': args.protection_loss_weight,  # ✨ Protection loss (core of Experiment 2)
        'policy_gradient_weight': args.policy_gradient_weight,
        'transition_loss_weight': args.transition_loss_weight,
        'listener_loss_weight': args.listener_loss_weight,
        'cost_loss_weight': args.cost_loss_weight,
        'protection_loss_weight': args.protection_loss_weight,  # ✨ Added
        
        # Protection mechanism configuration
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
        
        # ✨ Attack injection configuration (new)
        'enable_attack': args.enable_attack,
        'attack': {
            'attack_type': args.attack_type,
            'attack_ratio': args.attack_ratio,
            'attack_strength': args.attack_strength,
            'attack_frequency': args.attack_frequency,
            'attack_seed': args.attack_seed,
            'attack_start_epoch': args.attack_start_epoch
        },
        
        # FSM arguments
        'max_transitions': args.max_transitions,
        'mmlu_use_category_fsm': args.mmlu_use_category_fsm,
        
        # FSM cache arguments
        'fsm_cache_dir': args.fsm_cache_dir,
        'generate_fsm_if_missing': args.generate_fsm_if_missing,
        'use_fsm_cache': getattr(args, 'use_fsm_cache', True),
        
        # Note: Experiment 2 does not use prompt optimization (it focuses on protection mechanisms and attack injection)
        'enable_prompt_optimization': False,
        
        # TGN arguments
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
        'llm_name': args.llm_name,
        'agent_embedding_dim': 256,
        'state_feature_dim': 256,
        
        # Dataset arguments
        'train_ratio': 1.0 if args.train_only else 0.7,  # Use the full dataset when train_only is enabled
        'val_ratio': 0.0 if args.train_only else 0.15,
        'test_ratio': 0.0 if args.train_only else 0.15,
        'random_seed': 42,
        
        # ✨ Sampling arguments (new)
        'train_samples': args.train_samples,  # Number of randomly selected training samples
        'train_only': args.train_only,  # Train only, without validation/testing
    }
    
    # Initialize the FSM cache manager
    print("\n📦 Initializing FSM cache manager...")
    cache_manager = create_cache_manager(args.fsm_cache_dir)
    
    # Prepare FSM configuration
    print("\n🏗️  Preparing FSM configuration...")
    fsm_generator = EnhancedFSMGenerator(use_azure=False)
    
    for domain in args.domains:
        print(f"\n  📋 Checking FSM for {domain.upper()}...")
        if domain == 'mmlu' and args.mmlu_use_category_fsm:
            print(f"    ℹ️  MMLU is using category-level FSMs, which will be loaded dynamically during training")
        else:
            if cache_manager.has_cache(domain):
                print(f"    ✅ Cached FSM found")
            elif args.generate_fsm_if_missing:
                print(f"    🔨 Generating a new FSM...")
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
                    print(f"    ✅ FSM generation complete (cost: ${cost:.4f})")
                except Exception as e:
                    print(f"    ⚠️  FSM generation failed: {e}")
    
    # Create the trainer
    print("\n🔧 Initializing FSM + protection trainer...")
    trainer = FSMMultiAgentSystemTrainerV2(
        config=config,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir
    )
    
    # Inject the FSM cache manager
    trainer.fsm_cache_manager = cache_manager
    trainer.fsm_generator = fsm_generator
    
    # ✨ Protection mechanism integration note (new)
    if args.use_protection:
        print("\n🛡️  Protection mechanisms will be automatically integrated into FSM-TGN when the FSM system is created")
        print("    (implemented in create_domain_fsm_system in train_fsm_mas_v2.py)")
    
    # Start training
    print("\n🎯 Starting training...")
    results = await trainer.train_all_domains(domains=args.domains)
    
    # Output results
    print("\n" + "="*80)
    print("📊 Training results summary")
    print("="*80)
    
    for domain, domain_results in results.items():
        print(f"\n📚 {domain.upper()}:")
        print(f"  • Best accuracy: {domain_results.get('best_accuracy', 0.0):.4f}")
        print(f"  • Best epoch: {domain_results.get('best_epoch', 0)}")
    
    # ✨ Print cost statistics
    if hasattr(trainer, 'cost_tracker') and trainer.cost_tracker is not None:
        print("\n" + "="*80)
        print("💰 LLM cost statistics")
        print("="*80)
        trainer.cost_tracker.print_statistics()
    
    # Save results
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
    
    print(f"\n✅ Results saved to: {summary_path}")
    print("\n" + "="*80)
    print("🎉 Training complete!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
