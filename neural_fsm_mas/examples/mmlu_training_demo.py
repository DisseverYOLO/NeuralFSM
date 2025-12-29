"""
MMLU Training Demo
MMLU训练演示脚本

演示完整的MMLU数据集训练过程，包括：
1. 智能体交互日志
2. 策略梯度优化
3. 训练和验证准确率
4. 详细的训练过程记录
"""

import sys
import json
import time
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_fsm_mas.train_fsm_mas import FSMMultiAgentSystemTrainer


def run_mmlu_training_demo():
    """运行MMLU训练演示"""
    
    print("🌟 MMLU训练演示")
    print("="*80)
    print("本演示将展示以下功能：")
    print("1. 📊 智能体交互日志记录")
    print("2. 🎯 策略梯度优化过程")
    print("3. 📈 训练和验证准确率跟踪")
    print("4. 💾 详细的训练过程记录")
    print("="*80)
    
    # 配置参数
    config = {
        'use_neural_learning': True,
        'training_episodes': 20,  # 演示用较少轮数
        'batch_size': 8,
        'learning_rate': 0.001,
        'memory_dim': 64,
        'time_dim': 16,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'available_tools': [
            'code_interpreter', 'web_search', 'calculator',
            'knowledge_retrieval', 'logical_reasoning', 'analysis',
            'critical_thinking', 'synthesis', 'verification'
        ]
    }
    
    # 输出目录
    output_dir = "./mmlu_training_demo_outputs"
    
    print(f"📁 输出目录: {output_dir}")
    print(f"⚙️ 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 创建训练器（不使用真实MMLU数据，使用模拟数据演示）
    trainer = FSMMultiAgentSystemTrainer(
        config=config,
        data_path=None,  # 使用模拟数据
        output_dir=output_dir
    )
    
    print(f"\n🎯 开始演示训练过程...")
    
    try:
        # 演示单个任务的完整训练过程
        task_description = "Solve STEM questions including mathematics, physics, chemistry, and biology problems"
        
        print(f"\n📋 任务描述: {task_description}")
        print(f"🚀 开始生成FSM-MAS系统...")
        
        # 生成FSM-MAS系统
        fsm_mas_config = trainer.generate_fsm_mas_for_task(task_description)
        
        print(f"✅ FSM-MAS系统生成完成")
        print(f"   智能体数量: {fsm_mas_config['system_metadata']['num_agents']}")
        print(f"   状态数量: {fsm_mas_config['system_metadata']['num_states']}")
        
        # 开始训练（这里会展示完整的训练过程）
        print(f"\n🎓 开始TGN训练过程...")
        print("注意观察以下内容：")
        print("- 📊 每个episode的训练和验证准确率")
        print("- 🤖 智能体交互日志")
        print("- 📉 策略梯度优化更新")
        print("- 🌟 最佳模型更新")
        
        learning_results = trainer.train_fsm_mas_with_tgn(
            fsm_mas_config,
            training_episodes=config['training_episodes'],
            domain="STEM_Demo"
        )
        
        # 展示训练结果
        print(f"\n🎉 训练完成！")
        print("="*80)
        print("📊 最终训练结果:")
        print("="*80)
        
        print(f"🎯 最终训练准确率: {learning_results['final_training_accuracy']:.4f}")
        print(f"🎯 最终验证准确率: {learning_results['final_validation_accuracy']:.4f}")
        print(f"🌟 最佳验证准确率: {learning_results['best_validation_accuracy']:.4f} (轮次 {learning_results['best_episode']})")
        print(f"📈 平均训练准确率: {learning_results['average_training_accuracy']:.4f}")
        print(f"📈 平均验证准确率: {learning_results['average_validation_accuracy']:.4f}")
        print(f"🤖 总交互次数: {learning_results['total_interactions']}")
        
        # 展示训练曲线
        print(f"\n📈 训练曲线 (最近10个episode):")
        recent_train_acc = learning_results['training_metrics']['episode_accuracies'][-10:]
        recent_val_acc = learning_results['training_metrics']['validation_accuracies'][-10:]
        
        for i, (train_acc, val_acc) in enumerate(zip(recent_train_acc, recent_val_acc)):
            episode_num = len(learning_results['training_metrics']['episode_accuracies']) - len(recent_train_acc) + i + 1
            print(f"   Episode {episode_num:2d}: 训练={train_acc:.4f}, 验证={val_acc:.4f}")
        
        # 展示策略梯度更新信息
        print(f"\n🎯 策略梯度优化信息:")
        recent_updates = learning_results['training_metrics']['policy_gradient_updates'][-5:]
        for update in recent_updates:
            print(f"   Episode {update['episode']:2d}: 梯度范数={update['gradient_norm']:.4f}, 学习率={update['learning_rate']:.6f}")
        
        # 展示智能体交互统计
        interaction_logs = learning_results['training_metrics']['agent_interaction_logs']
        if interaction_logs:
            print(f"\n🤖 智能体交互统计:")
            
            # 统计成功率
            successful_interactions = sum(1 for log in interaction_logs if log.get('is_correct', log.get('success', False)))
            success_rate = successful_interactions / len(interaction_logs)
            print(f"   总交互次数: {len(interaction_logs)}")
            print(f"   成功交互次数: {successful_interactions}")
            print(f"   成功率: {success_rate:.4f}")
            
            # 展示最近几次交互的详细信息
            print(f"\n📝 最近5次交互详情:")
            recent_interactions = interaction_logs[-5:]
            for i, log in enumerate(recent_interactions):
                if 'question' in log:
                    print(f"   交互 {i+1}:")
                    print(f"      问题: {log['question'][:80]}...")
                    print(f"      正确答案: {log.get('correct_answer', 'N/A')}")
                    print(f"      智能体答案: {log.get('predicted_answer', 'N/A')}")
                    print(f"      结果: {'✅ 正确' if log.get('is_correct') else '❌ 错误'}")
                    print(f"      执行时间: {log.get('execution_time', 0):.2f}秒")
                elif 'simulated' in log:
                    print(f"   模拟交互 {i+1}:")
                    print(f"      智能体: {log.get('agent_id', 'N/A')}")
                    print(f"      状态: {log.get('state_id', 'N/A')}")
                    print(f"      动作: {log.get('action', 'N/A')}")
                    print(f"      成功: {'✅' if log.get('success') else '❌'}")
        
        # 保存演示结果摘要
        demo_summary = {
            'demo_type': 'MMLU Training Demo',
            'task_description': task_description,
            'config': config,
            'results': {
                'final_training_accuracy': learning_results['final_training_accuracy'],
                'final_validation_accuracy': learning_results['final_validation_accuracy'],
                'best_validation_accuracy': learning_results['best_validation_accuracy'],
                'best_episode': learning_results['best_episode'],
                'total_interactions': learning_results['total_interactions'],
                'training_episodes': config['training_episodes']
            },
            'timestamp': time.time()
        }
        
        summary_path = Path(output_dir) / "demo_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(demo_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 演示摘要已保存到: {summary_path}")
        
        # 获取训练摘要
        training_summary = trainer.get_training_summary()
        print(f"\n📋 训练器摘要:")
        for key, value in training_summary.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎉 MMLU训练演示完成！")
        print(f"📁 所有结果文件保存在: {output_dir}")
        
        return learning_results
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_quick_demo():
    """运行快速演示（更少的episode）"""
    
    print("⚡ 快速演示模式")
    print("="*50)
    
    # 快速配置
    config = {
        'use_neural_learning': True,
        'training_episodes': 5,  # 只有5个episode
        'batch_size': 4,
        'learning_rate': 0.001,
        'memory_dim': 32,
        'time_dim': 8,
        'random_seed': 42,
        'available_tools': ['calculator', 'logical_reasoning', 'analysis']
    }
    
    output_dir = "./quick_demo_outputs"
    
    trainer = FSMMultiAgentSystemTrainer(
        config=config,
        data_path=None,
        output_dir=output_dir
    )
    
    # 简单任务
    task_description = "Solve basic math problems"
    
    print(f"🎯 任务: {task_description}")
    print(f"🚀 开始快速训练...")
    
    try:
        # 生成和训练
        fsm_mas_config = trainer.generate_fsm_mas_for_task(task_description)
        learning_results = trainer.train_fsm_mas_with_tgn(fsm_mas_config)
        
        print(f"\n✅ 快速演示完成！")
        print(f"📊 最终验证准确率: {learning_results['final_validation_accuracy']:.4f}")
        print(f"🤖 总交互次数: {learning_results['total_interactions']}")
        
        return learning_results
        
    except Exception as e:
        print(f"❌ 快速演示失败: {e}")
        return None


if __name__ == "__main__":
    print("选择演示模式:")
    print("1. 完整MMLU训练演示 (20个episodes, 详细日志)")
    print("2. 快速演示 (5个episodes, 基本功能)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        run_mmlu_training_demo()
    elif choice == "2":
        run_quick_demo()
    else:
        print("无效选择，运行快速演示...")
        run_quick_demo()

