import asyncio
import sys
import argparse
import json
import os
from pathlib import Path

# Add path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from neural_fsm_mas.train_fsm_mas_v2 import FSMMultiAgentSystemTrainerV2
from neural_fsm_mas.core_integration import (
    EnhancedFSMIntegration,
    create_enhanced_integration,
    StateAgentCorrespondenceLearner
)  # ✨ Import core integration modules
from neural_fsm_mas.fsm_cache_manager import create_cache_manager  # ✨ FSM cache
from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator  # ✨ FSM generator


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment 1 Complete V2: new FSM architecture + three-objective optimization'
    )
    
    # Basic arguments
    parser.add_argument('--domains', type=str, nargs='+',
                       default=['gsm8k'],
                       help='Training domains (mmlu, gsm8k, gpqa, gaia, humaneval, hotpotqa, alfworld, math)')
    parser.add_argument('--dataset_root', type=str, default='./datasets',
                       help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/experiment1_fsm_v2',
                       help='Output directory')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Four-objective combined loss weights ✨ Added delta
    parser.add_argument('--policy_gradient_weight', type=float, default=1.0,
                       help='Policy gradient loss weight (α)')
    parser.add_argument('--transition_loss_weight', type=float, default=0.3,
                       help='State transition loss weight (β)')
    parser.add_argument('--listener_loss_weight', type=float, default=0.2,
                       help='Listener path loss weight (γ)')
    parser.add_argument('--cost_loss_weight', type=float, default=0.001,
                       help='LLM cost loss weight (δ) ✨ NEW')
    
    # FSM validation and execution arguments ✨ Added
    parser.add_argument('--max_visits_per_state', type=int, default=3,
                       help='Maximum visits per state (loop prevention)')
    parser.add_argument('--max_total_steps', type=int, default=20,
                       help='Maximum total steps per episode')
    parser.add_argument('--max_transitions', type=int, default=8,
                       help='Maximum number of state transitions (maximum transition steps per question)')
    parser.add_argument('--auto_fix_fsm', action='store_true', default=True,
                       help='Automatically fix FSM issues (such as adding rescue transitions)')
    parser.add_argument('--use_llm_for_conditions', action='store_true', default=False,
                       help='Use LLM to generate missing transition conditions')
    
    # TGN arguments
    parser.add_argument('--memory_dim', type=int, default=128,
                       help='TGN memory dimension')
    parser.add_argument('--time_dim', type=int, default=32,
                       help='Time encoding dimension')
    parser.add_argument('--agent_embedding_dim', type=int, default=256,
                       help='Agent embedding dimension')
    parser.add_argument('--state_feature_dim', type=int, default=256,
                       help='State feature dimension')
    
    # LLM arguments
    parser.add_argument('--llm_name', type=str, default='gpt-5-nano',
                       help='LLM model name')
    
    # FSM cache arguments ✨ Added
    parser.add_argument('--fsm_cache_dir', type=str, default='./fsm_cache',
                       help='FSM cache directory')
    # FSM cache is enabled by default; use --no-use-fsm-cache to disable it
    parser.add_argument('--no-use-fsm-cache', dest='use_fsm_cache', action='store_false',
                       help='Disable FSM cache (force regeneration; cache is enabled by default)')
    parser.add_argument('--force_regenerate_fsm', action='store_true', default=False,
                       help='Force FSM regeneration on every run (ignore existing cache)')
    parser.add_argument('--generate_fsm_if_missing', action='store_true', default=True,
                       help='Automatically generate FSM if cache is missing')
    parser.add_argument('--mmlu_use_category_fsm', action='store_true', default=False,
                       help='Use category-level FSM for MMLU (one FSM per category); default False uses a unified FSM')
    
    # ✨ State description optimization arguments (new)
    parser.add_argument('--enable_prompt_optimization', action='store_true', default=False,
                       help='Enable dynamic state description optimization (enhance descriptions based on execution performance)')
    parser.add_argument('--max_transitions_threshold', type=int, default=3,
                       help='Trigger efficiency optimization after reaching the maximum transition count N times (default: 3)')
    parser.add_argument('--accuracy_enhancement_threshold', type=int, default=5,
                       help='Trigger accuracy optimization after N failures (default: 5)')
    parser.add_argument('--efficiency_enhancement_threshold', type=int, default=3,
                       help='Trigger efficiency optimization after N inefficient executions (default: 3)')
    
    # ✨ Training data sampling arguments (new)
    parser.add_argument('--train_samples', type=int, default=None,
                       help='Number of samples randomly drawn from the training set (e.g. 100); use all samples if omitted')
    parser.add_argument('--train_only', action='store_true', default=False,
                       help='Train only, skipping validation and testing')
    parser.add_argument('--full_train_data', action='store_true', default=False,
                       help='Do not split validation/test sets: use 100% of the data for training (train_ratio=1,val_ratio=0,test_ratio=0)')

    # ✨ GAIA: filter by difficulty level (1/2/3); merge all levels if unspecified
    parser.add_argument('--gaia_level', type=int, default=None,
                       help='GAIA difficulty level (1/2/3). Only takes effect when domains includes gaia')

    # ✨ Experiment 1 ablation switches (GSM8K/MATH etc.)
    parser.add_argument('--disable_transition_prediction', action='store_true', default=False,
                       help='Ablation: disable TGN state transition prediction (do not sample the next state with transition_probs)')
    parser.add_argument('--disable_comm_sampling', action='store_true', default=False,
                       help='Ablation: disable communication path sampling (do not sample listeners with listener_weights)')
    parser.add_argument('--ablation', action='store_true', default=False,
                       help='Ablation mode: automatically write outputs to ablation/<variant>/ subdirectories to avoid overwriting regular Experiment 1 results')
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()
    
    # Write the CLI-specified LLM name to an environment variable for all LLM clients
    if args.llm_name:
        os.environ["NEURALFSM_LLM_MODEL"] = args.llm_name
    
    print("="*80)
    print("🚀 Experiment 1 Complete V3: four-objective optimization + full feature integration")
    print("="*80)
    print("\n✨ New features (V3):")
    print("  1. ✅ Four-objective combined loss (policy gradient + state transition + listener path + LLM cost)")
    print("  2. ✅ FSM validation and automatic optimization (reachability checks + rescue transitions)")
    print("  3. ✅ Automatic transition condition matching (dynamically matched during execution)")
    print("  4. ✅ Loop prevention mechanism (visit limits + step limits)")
    print("  5. ✅ Real-time LLM cost tracking (12+ model pricing entries)")
    print("\n📋 Core features:")
    print("  • One state corresponds to one agent")
    print("  • TGN learns optimal state transitions and communication paths")
    print("  • Supports all datasets (MMLU, GSM8K, HumanEval, HotpotQA, ALFWorld, MATH, MBPP) ✨")
    print("\n🆚 Version comparison:")
    print("  • V1: α*policy gradient + β*MSE reconstruction (indirect optimization)")
    print("  • V2: α*policy gradient + β*state transition + γ*listener path (three objectives)")
    print("  • V3: V2 + δ*LLM cost + FSM validation + transition matching (four objectives) ✨")
    print("\n📊 Configuration:")
    print(f"  • Training domains: {args.domains}")
    print(f"  • Epochs: {args.num_epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {args.learning_rate}")
    print(f"\n🎯 Five-objective loss weights:")
    print(f"  L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ζ·L_max_transitions")
    print(f"  • Objective 1 - Policy gradient (α): {args.policy_gradient_weight}")
    print(f"  • Objective 2 - State transition (β): {args.transition_loss_weight}")
    print(f"  • Objective 3 - Listener path (γ): {args.listener_loss_weight}")
    print(f"  • Objective 4 - LLM cost   (δ): {args.cost_loss_weight}")
    print(f"  • Objective 5 - Max transition penalty (ζ): 0.5 (default)")
    print(f"\n🛡️  FSM execution configuration:")
    print(f"  • Max visits per state: {args.max_visits_per_state}")
    print(f"  • Max episode steps: {args.max_total_steps}")
    print(f"  • Max state transitions: {args.max_transitions}")
    print(f"  • Auto-fix FSM: {args.auto_fix_fsm}")
    print(f"\n✨ State description optimization configuration:")
    print(f"  • Optimization enabled: {args.enable_prompt_optimization}")
    if args.enable_prompt_optimization:
        print(f"  • Accuracy optimization: triggered after {args.accuracy_enhancement_threshold} failures")
        print(f"  • Efficiency optimization: triggered after reaching the max transitions {args.max_transitions_threshold} times")
    print(f"\n🤖 Model configuration:")
    print(f"  • LLM model: {args.llm_name}")
    print(f"  • Memory dimension: {args.memory_dim}")
    print(f"  • Time encoding dimension: {args.time_dim}")
    print(f"  • Agent embedding dimension: {args.agent_embedding_dim}")
    print(f"  • State feature dimension: {args.state_feature_dim}")
    
    # ✨ Training data sampling configuration
    if args.train_samples or args.train_only or args.full_train_data:
        print(f"\n📋 Data sampling configuration:")
        if args.train_samples:
            print(f"  • Training samples: {args.train_samples} (randomly sampled)")
        if args.train_only:
            print(f"  • Mode: train only (skip validation and testing)")
        if args.full_train_data:
            print(f"  • Split: 100% of data used for training (no validation/test split)")
    print("="*80 + "\n")
    
    # Build configuration (V3: with new arguments)
    config = {
        # Basic training arguments
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        
        # Four-objective loss weights ✨
        'alpha': args.policy_gradient_weight,
        'beta': args.transition_loss_weight,
        'gamma': args.listener_loss_weight,
        'delta': args.cost_loss_weight,  # ✨ NEW: LLM cost weight
        'policy_gradient_weight': args.policy_gradient_weight,
        'transition_loss_weight': args.transition_loss_weight,
        'listener_loss_weight': args.listener_loss_weight,
        'cost_loss_weight': args.cost_loss_weight,  # ✨ NEW
        
        # FSM validation and execution arguments ✨
        'max_visits_per_state': args.max_visits_per_state,
        'max_total_steps': args.max_total_steps,
        'max_transitions': args.max_transitions,
        'auto_fix_fsm': args.auto_fix_fsm,
        'use_llm_for_conditions': args.use_llm_for_conditions,
        
        # TGN arguments
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
        'llm_name': args.llm_name,
        'model_name': args.llm_name,  # ✨ Used for cost tracking and logs
        'agent_embedding_dim': args.agent_embedding_dim,
        'state_feature_dim': args.state_feature_dim,
        
        # Dataset arguments
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        
        # FSM cache arguments ✨
        'fsm_cache_dir': args.fsm_cache_dir,
        'use_fsm_cache': args.use_fsm_cache if hasattr(args, 'use_fsm_cache') and args.use_fsm_cache is not None else True,
        'force_regenerate_fsm': args.force_regenerate_fsm,
        'generate_fsm_if_missing': args.generate_fsm_if_missing,
        'mmlu_use_category_fsm': args.mmlu_use_category_fsm,
        
        # ✨ State description optimization configuration (new)
        'enable_prompt_optimization': args.enable_prompt_optimization,
        'max_transitions_threshold': args.max_transitions_threshold,
        'accuracy_enhancement_threshold': args.accuracy_enhancement_threshold,
        'efficiency_enhancement_threshold': args.efficiency_enhancement_threshold,
        
        # ✨ Training data sampling configuration (new)
        'train_samples': args.train_samples,
        'train_only': args.train_only,
        'full_train_data': args.full_train_data,
        'gaia_level': args.gaia_level,

        # ✨ Ablation switches
        'enable_transition_prediction': (not args.disable_transition_prediction),
        'enable_comm_sampling': (not args.disable_comm_sampling),
        'ablation': bool(args.ablation),
        
        # Cost estimation arguments ✨
        'avg_input_tokens': 500,
        'avg_output_tokens': 200,
        'avg_calls_per_episode': 5
    }

    # ✨ Use 100% of the data for training: override the default split ratio and avoid conflicts with train_samples
    if args.full_train_data:
        config['train_ratio'] = 1.0
        config['val_ratio'] = 0.0
        config['test_ratio'] = 0.0
        if args.train_samples:
            # Ignore random sampling when the user explicitly requests training on 100% of the data
            config['train_samples'] = None
            print("  ⚠️  --full_train_data is enabled, so --train_samples will be ignored (all samples will be used for training)")
    
    # ✨ Initialize the FSM cache manager (new)
    print("\n📦 Initializing FSM cache manager...")
    cache_manager = create_cache_manager(args.fsm_cache_dir)
    
    # ✨ Prepare FSMs for each dataset (new)
    print("\n🏗️  Preparing FSM configuration...")
    fsm_generator = EnhancedFSMGenerator(use_azure=False)
    
    for domain in args.domains:
        print(f"\n  📋 Checking FSM for {domain.upper()}...")
        
        if domain == 'mmlu' and args.mmlu_use_category_fsm:
            # Special handling for MMLU: generate one FSM per category
            print(f"    ℹ️  MMLU is using category-level FSMs")
            
            # ✨ Check and generate missing category FSMs
            if args.generate_fsm_if_missing:
                print(f"    🔍 Checking MMLU category FSM cache...")
                from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor
                data_processor = UnifiedDataProcessor(args.dataset_root)
                mmlu_data = data_processor.load_dataset('mmlu', split='test')
                
                # Collect all unique categories from the full dataset to ensure complete coverage
                # MMLU uses the 'subject' field to represent categories
                categories = set()
                for item in mmlu_data:
                    cat = item.get('subject', '')  # MMLU uses the subject field
                    if cat:
                        categories.add(cat)
                
                print(f"    📊 Found {len(categories)} categories that require FSMs")
                
                generated_count = 0
                cached_count = 0
                # ✨ Check whether caching is enabled
                use_cache = getattr(args, 'use_fsm_cache', True)
                for category in sorted(categories):
                    if use_cache and cache_manager.has_cache('mmlu', category):
                        cached_count += 1
                    elif args.generate_fsm_if_missing:
                        print(f"      🔨 Generating FSM for category '{category}'...")
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
                            print(f"        ✅ Done (cost: ${cost:.4f} USD)")
                        except Exception as e:
                            print(f"        ⚠️  Generation failed: {e}")
                            print(f"        The default FSM will be used during training")
                
                print(f"    📊 Summary: {cached_count} cached, {generated_count} newly generated")
                if generated_count == 0 and cached_count == 0:
                    print(f"    ⚠️  No category FSMs were found; the default FSM will be used during training")
            else:
                print(f"    ℹ️  Auto-generation is disabled; FSMs will be loaded on demand during training or fall back to the default FSM")
        else:
            # Other datasets: check or generate FSM
            # ✨ Check first whether caching is enabled
            use_cache = getattr(args, 'use_fsm_cache', True)
            if use_cache and cache_manager.has_cache(domain):
                print(f"    ✅ Cached FSM found")
                cached = cache_manager.load_fsm(domain)
                print(f"      - Number of states: {cached['metadata']['num_states']}")
                print(f"      - Number of agents: {cached['metadata']['num_agents']}")
            elif args.generate_fsm_if_missing:
                print(f"    🔨 Generating a new FSM...")
                try:
                    mas_config, cost = fsm_generator.generate_complete_mas(
                        dataset=domain,
                        save_path=None  # Do not save to a file; just return the configuration
                    )
                    # Save to cache
                    cache_manager.save_fsm(
                        dataset=domain,
                        fsm_config=mas_config.get('fsm', {}),
                        agents=mas_config.get('agents', []),
                        metadata={'generation_cost': cost}
                    )
                    print(f"    ✅ FSM generation complete (cost: ${cost:.4f} USD)")
                except Exception as e:
                    print(f"    ⚠️  FSM generation failed: {e}")
                    print(f"    The default configuration will be used during training")
            else:
                print(f"    ⚠️  No cache was found and auto-generation is disabled; the default configuration will be used")
    
    # ✨ Create the core integration module (new)
    print("\n🔧 Initializing core integration module...")
    integration = create_enhanced_integration(config)
    
    # Create trainer (split output directories by llm_name)
    # ✨ Output directory structure:
    #    - GAIA: output_dir/level{N}/{llm_name}/
    #    - Ablation experiments: output_dir/ablation/{variant}/{llm_name}/
    #    - Regular experiments: output_dir/{llm_name}/
    base_output = Path(args.output_dir)
    
    # ✨ Distinguish GAIA levels (if --gaia_level is specified)
    if 'gaia' in args.domains and args.gaia_level is not None:
        base_output = base_output / f"level{args.gaia_level}"
    
    if args.ablation:
        variant_parts = []
        if args.disable_transition_prediction:
            variant_parts.append("no_trans")
        if args.disable_comm_sampling:
            variant_parts.append("no_comm")
        # "Remove prompt optimization" means enable_prompt_optimization=False
        if not args.enable_prompt_optimization:
            variant_parts.append("no_promptopt")
        # Full ablation baseline: prompt optimization enabled and no other features disabled
        if not variant_parts and args.enable_prompt_optimization:
            variant = "full"
        else:
            variant = "_".join(variant_parts) if variant_parts else "ablation"
        base_output = base_output / "ablation" / variant

    llm_specific_output = base_output / args.llm_name
    print("🔧 Initializing FSM trainer...")
    trainer = FSMMultiAgentSystemTrainerV2(
        config=config,
        dataset_root=args.dataset_root,
        output_dir=llm_specific_output
    )
    
    # ✨ Inject the FSM cache manager (new)
    trainer.fsm_cache_manager = cache_manager
    trainer.fsm_generator = fsm_generator
    
    # ✨ Inject the integration module into the trainer (new)
    print("🔗 Integrating the four-objective loss and FSM validation features...")
    integration.integrate_with_trainer(trainer)
    
    try:
        # Train all domains
        print(f"\n🎯 Starting training for domains: {args.domains}")
        print("="*80)
        results = await trainer.train_all_domains(args.domains)
        
        # Print detailed results
        print("\n" + "="*80)
        print("📊 Training summary")
        print("="*80)
        
        for domain, result in results.items():
            print(f"\n🔸 Domain: {domain}")
            print(f"  • Best accuracy: {result['best_accuracy']:.4f} (Epoch {result['best_epoch']})")
            print(f"  • Final training accuracy: {result['epoch_accuracies'][-1]:.4f}" if result['epoch_accuracies'] else "  • Final training accuracy: N/A")
            print(f"  • Final validation accuracy: {result['validation_accuracies'][-1]:.4f}" if result['validation_accuracies'] else "  • Final validation accuracy: N/A")
            
            if result['epoch_losses']:
                print(f"  • Final total loss: {result['epoch_losses'][-1]:.4f}")
                print(f"    ├─ Policy gradient loss: {result['epoch_policy_losses'][-1]:.4f}")
                print(f"    ├─ State transition loss: {result['epoch_transition_losses'][-1]:.4f}")
                print(f"    ├─ Listener path loss: {result['epoch_listener_losses'][-1]:.4f}")
                # ✨ New: show LLM cost loss
                if 'epoch_cost_losses' in result and result['epoch_cost_losses']:
                    print(f"    └─ LLM cost loss: {result['epoch_cost_losses'][-1]:.4f} ✨")
        
        # Save summary
        summary = {
            'config': config,
            'domains': args.domains,
            'results': results,
            'best_domain': max(results.items(), key=lambda x: x[1]['best_accuracy'])[0],
            'average_best_accuracy': sum(r['best_accuracy'] for r in results.values()) / len(results)
        }
        
        summary_path = llm_specific_output / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to a serializable format
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
        
        # ✨ Output the state description optimization summary (new)
        if args.enable_prompt_optimization:
            print("\n" + "="*80)
            print("✨ State description optimization summary")
            print("="*80)
            for domain in args.domains:
                if hasattr(trainer, 'prompt_optimizers') and domain in trainer.prompt_optimizers:
                    optimizer = trainer.prompt_optimizers[domain]
                    summary = optimizer.get_performance_summary()
                    print(f"\n📊 {domain.upper()}:")
                    print(f"  • Total questions: {summary['total_questions']}")
                    print(f"  • Overall accuracy: {summary['overall_accuracy']:.2%}")
                    print(f"  • Reached max transitions: {summary['max_transitions_count']} times ({summary['max_transitions_rate']:.1%})")
                    print(f"  • Enhanced states: {summary['enhanced_states']} / {summary['total_states_tracked']}")
                    if summary['problem_states']:
                        print(f"\n  ⚠️  Problem states (sorted by severity):")
                        for state in summary['problem_states'][:5]:  # Show only the top 5
                            enhanced_mark = ""
                            if state['is_enhanced']:
                                if state['enhancement_type'] == 'accuracy':
                                    enhanced_mark = "✅ Enhanced (accuracy)"
                                elif state['enhancement_type'] == 'efficiency':
                                    enhanced_mark = "⚡ Enhanced (efficiency)"
                                elif state['enhancement_type'] == 'combined':
                                    enhanced_mark = "🔥 Enhanced (accuracy + efficiency)"
                            else:
                                enhanced_mark = "⏳ Under observation"
                            print(f"    • {state['state_name']}:")
                            print(f"        Failure rate: {state['failure_rate']:.1%} ({state['failure_count']} times)")
                            print(f"        Inefficiency rate: {state['inefficiency_rate']:.1%} ({state['max_trans_count']} times)")
                            print(f"        Status: {enhanced_mark}")
        
        # ✨ Print cost statistics (fixed: use trainer.cost_tracker instead of integration.cost_tracker)
        # trainer.cost_tracker is the global tracker that actually records LLM call costs
        if hasattr(trainer, 'cost_tracker') and trainer.cost_tracker is not None:
            print("\n" + "="*80)
            print("💰 LLM cost statistics")
            print("="*80)
            trainer.cost_tracker.print_statistics()
        elif hasattr(integration, 'cost_tracker'):
            # Fall back to integration.cost_tracker
            print("\n" + "="*80)
            print("💰 LLM cost statistics")
            print("="*80)
            integration.cost_tracker.print_statistics()
        
        print("\n" + "="*80)
        print("🎉 Training complete!")
        print("="*80)
        print(f"\n📈 Best-performing domain: {summary['best_domain']}")
        print(f"📊 Average best accuracy: {summary['average_best_accuracy']:.4f}")
        print(f"💾 Results saved to: {llm_specific_output}")
        print(f"📄 Summary file: {summary_path}")
        print(f"\n💡 Detailed documentation:")
        print(f"  • Loss functions: FSM_LOSS_FUNCTIONS_EXPLAINED.md")
        print(f"  • FSM design: RANDOM_SAMPLED_FSM_DESIGN.md")
        print(f"  • Integration guide: FINAL_INTEGRATION_GUIDE.md")
        print(f"  • Core integration: neural_fsm_mas/core_integration.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
