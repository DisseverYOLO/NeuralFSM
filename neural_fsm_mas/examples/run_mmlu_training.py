"""
MMLU Training Example
MMLU训练示例

演示如何使用Neural MAS系统训练MMLU数据集
"""

import asyncio
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_fsm_mas.train_neural_mas import NeuralMASTrainer
import json


async def run_mmlu_training_example():
    """运行MMLU训练示例"""
    
    # 配置参数
    config = {
        'num_epochs': 20,  # 示例用较少的epoch
        'batch_size': 8,   # 示例用较小的batch size
        'learning_rate': 0.001,
        'num_rounds': 2,   # 示例用较少的轮次
        'memory_dim': 64,  # 示例用较小的维度
        'time_dim': 16,
        'llm_name': 'gpt-4o-mini',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'available_tools': [
            'knowledge_retrieval', 'logical_reasoning', 'calculation', 
            'analysis', 'critical_thinking'
        ]
    }
    
    # MMLU数据路径 (需要根据实际情况修改)
    mmlu_data_path = "path/to/mmlu/data"  # 请修改为实际的MMLU数据路径
    output_dir = "./neural_mas_example_outputs"
    
    print("🌟 Neural MAS MMLU训练示例")
    print(f"📁 数据路径: {mmlu_data_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"⚙️ 配置: {json.dumps(config, indent=2)}")
    
    # 创建训练器
    trainer = NeuralMASTrainer(
        config=config,
        mmlu_data_path=mmlu_data_path,
        output_dir=output_dir
    )
    
    try:
        # 只训练一个领域作为示例
        print("\n📚 准备训练数据...")
        statistics = trainer.prepare_training_data()
        
        # 选择第一个领域进行示例训练
        example_domain = list(statistics.keys())[0]
        print(f"\n🎯 示例训练领域: {example_domain}")
        
        # 创建任务描述
        task_description = f"Solve {example_domain} questions from MMLU dataset"
        
        # 创建拓扑管理器
        topology_manager = trainer.create_domain_topology(example_domain, task_description)
        
        # 训练
        print(f"\n🚀 开始训练领域 {example_domain}...")
        training_results = await trainer.train_domain_topology(
            example_domain, 
            topology_manager, 
            num_epochs=config['num_epochs']
        )
        
        print(f"\n✅ 训练完成!")
        print(f"📊 最佳验证准确率: {training_results['best_accuracy']:.4f}")
        print(f"📊 最佳epoch: {training_results['best_epoch']}")
        
        # 简单评估
        print(f"\n🧪 评估领域 {example_domain}...")
        trainer.domain_topologies[example_domain] = topology_manager
        evaluation_results = await trainer.evaluate_all_domains()
        
        print(f"📊 测试准确率: {evaluation_results[example_domain]:.4f}")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


def run_single_domain_test():
    """运行单个领域的快速测试"""
    
    print("🧪 Neural MAS单领域快速测试")
    
    # 这里可以添加单个领域的快速测试逻辑
    # 例如：测试智能体创建、拓扑构建等
    
    from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
    from neural_fsm_mas.reasoning_agents.agent_factory import ReasoningAgentFactory
    
    # 测试智能体工厂
    print("🔧 测试智能体工厂...")
    available_types = ReasoningAgentFactory.get_available_agent_types()
    print(f"可用智能体类型: {available_types}")
    
    # 测试创建智能体
    if 'mathematical_reasoning' in available_types:
        math_agent = ReasoningAgentFactory.create_agent(
            'mathematical_reasoning',
            node_id='test_math_agent'
        )
        print(f"✅ 创建数学推理智能体: {math_agent}")
    
    print("✅ 单领域测试完成")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 完整MMLU训练示例")
    print("2. 单领域快速测试")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_mmlu_training_example())
    elif choice == "2":
        run_single_domain_test()
    else:
        print("无效选择")
