"""
Complete FSM-MAS Example
完整的FSM-MAS使用示例

演示如何使用Neural FSM-MAS系统：
1. 自动生成智能体和FSM状态
2. 随机采样拓扑图
3. 使用TGN学习最优路径
"""

import asyncio
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_fsm_mas.fsm_integration.fsm_mas_generator import (
    FSMMultiAgentSystemGenerator,
    generate_and_learn_fsm_mas
)
from neural_fsm_mas.train_fsm_mas import FSMMultiAgentSystemTrainer
import json


def example_1_basic_fsm_mas_generation():
    """示例1: 基础FSM-MAS生成"""
    print("🎯 示例1: 基础FSM-MAS生成")
    print("="*50)
    
    # 创建FSM-MAS生成器
    generator = FSMMultiAgentSystemGenerator(
        use_neural_learning=True,
        memory_dimension=64,  # 示例用较小维度
        temporal_dimension=16
    )
    
    # 定义任务
    task_description = "Solve mathematical word problems step by step with verification"
    
    # 生成完整的FSM-MAS系统
    fsm_mas_config = generator.generate_complete_fsm_mas(
        task_description=task_description,
        available_tools=['calculator', 'logical_reasoning', 'verification']
    )
    
    # 打印生成结果
    print(f"✅ 生成的智能体数量: {fsm_mas_config['system_metadata']['num_agents']}")
    print(f"✅ 生成的状态数量: {fsm_mas_config['system_metadata']['num_states']}")
    print(f"✅ 状态转移边数量: {fsm_mas_config['system_metadata']['state_topology_edges']}")
    print(f"✅ 通信拓扑边数量: {fsm_mas_config['system_metadata']['listening_topology_edges']}")
    
    # 显示生成的智能体
    print("\n🤖 生成的智能体:")
    for agent in fsm_mas_config['agents']:
        print(f"  - {agent['name']} (ID: {agent['agent_id']})")
        print(f"    工具: {agent['tools']}")
    
    # 显示生成的状态
    print("\n🔄 生成的FSM状态:")
    for state in fsm_mas_config['fsm']['states']:
        print(f"  - 状态 {state['state_id']}: {state['instruction']}")
        print(f"    智能体: {state['agent_id']}, 监听者: {state.get('listener', [])}")
    
    return fsm_mas_config


def example_2_tgn_learning():
    """示例2: TGN学习最优拓扑"""
    print("\n🎯 示例2: TGN学习最优拓扑")
    print("="*50)
    
    # 使用便捷函数生成并学习
    task_description = "Analyze scientific papers and extract key findings"
    
    complete_results = generate_and_learn_fsm_mas(
        task_description=task_description,
        available_tools=['web_search', 'analysis', 'synthesis'],
        training_episodes=20,  # 示例用较少轮数
        output_path="./example_fsm_mas_system.json"
    )
    
    # 显示学习结果
    learning_results = complete_results['learning_results']
    print(f"✅ 训练轮数: {learning_results['training_episodes']}")
    print(f"✅ 最终状态损失: {learning_results['final_state_loss']:.4f}")
    print(f"✅ 最终通信损失: {learning_results['final_communication_loss']:.4f}")
    
    # 显示优化后的拓扑
    optimized_state_topo = learning_results['optimized_state_topology']
    optimized_comm_topo = learning_results['optimized_communication_topology']
    
    print(f"\n🕸️ 优化后的状态转移连接: {len(optimized_state_topo['edges'])} 条")
    print(f"📡 优化后的通信连接: {len(optimized_comm_topo['edges'])} 条")
    
    return complete_results


def example_3_training_with_data():
    """示例3: 使用训练器进行数据训练"""
    print("\n🎯 示例3: 使用训练器进行数据训练")
    print("="*50)
    
    # 配置
    config = {
        'use_neural_learning': True,
        'training_episodes': 15,  # 示例用较少轮数
        'learning_rate': 0.001,
        'memory_dim': 64,
        'time_dim': 16,
        'available_tools': [
            'code_interpreter', 'web_search', 'calculator',
            'logical_reasoning', 'analysis'
        ]
    }
    
    # 创建训练器
    trainer = FSMMultiAgentSystemTrainer(
        config=config,
        output_dir="./example_training_outputs"
    )
    
    # 训练单个任务
    task_description = "Generate Python code for data visualization and analysis"
    results = trainer.train_single_task(task_description)
    
    # 显示结果
    fsm_config = results['fsm_mas_config']
    learning_results = results['learning_results']
    
    print(f"✅ 任务: {task_description}")
    print(f"✅ 智能体数量: {fsm_config['system_metadata']['num_agents']}")
    print(f"✅ 状态数量: {fsm_config['system_metadata']['num_states']}")
    print(f"✅ 最终状态损失: {learning_results['final_state_loss']:.4f}")
    print(f"✅ 最终通信损失: {learning_results['final_communication_loss']:.4f}")
    
    return results


def example_4_demonstrate_capabilities():
    """示例4: 演示系统能力"""
    print("\n🎯 示例4: 演示系统能力")
    print("="*50)
    
    # 配置
    config = {
        'use_neural_learning': True,
        'training_episodes': 10,  # 演示用很少轮数
        'memory_dim': 32,
        'time_dim': 8
    }
    
    # 创建训练器
    trainer = FSMMultiAgentSystemTrainer(
        config=config,
        output_dir="./demo_outputs"
    )
    
    # 运行演示
    demo_results = trainer.demonstrate_fsm_mas_capabilities()
    
    # 显示演示结果
    print("\n🎪 演示结果摘要:")
    for task_key, result in demo_results.items():
        if 'error' not in result:
            print(f"  {task_key}: {result['num_agents']}个智能体, {result['num_states']}个状态")
            losses = result['final_losses']
            print(f"    状态损失: {losses['state_loss']:.4f}, 通信损失: {losses['communication_loss']:.4f}")
        else:
            print(f"  {task_key}: 错误 - {result['error']}")
    
    return demo_results


def example_5_analyze_generated_system():
    """示例5: 分析生成的系统"""
    print("\n🎯 示例5: 分析生成的系统")
    print("="*50)
    
    # 生成一个复杂任务的系统
    generator = FSMMultiAgentSystemGenerator()
    
    complex_task = """
    Develop a comprehensive solution for analyzing customer feedback data:
    1. Collect and preprocess text data
    2. Perform sentiment analysis
    3. Extract key topics and themes
    4. Generate actionable insights
    5. Create visualization reports
    """
    
    fsm_mas_config = generator.generate_complete_fsm_mas(
        task_description=complex_task,
        available_tools=[
            'code_interpreter', 'web_search', 'analysis', 
            'logical_reasoning', 'synthesis', 'verification'
        ]
    )
    
    # 详细分析生成的系统
    print("📊 系统分析:")
    print(f"  智能体总数: {len(fsm_mas_config['agents'])}")
    print(f"  状态总数: {len(fsm_mas_config['fsm']['states'])}")
    print(f"  转移总数: {len(fsm_mas_config['fsm']['transitions'])}")
    
    # 分析智能体角色分布
    agent_tools = {}
    for agent in fsm_mas_config['agents']:
        for tool in agent['tools']:
            agent_tools[tool] = agent_tools.get(tool, 0) + 1
    
    print(f"\n🔧 工具使用统计:")
    for tool, count in sorted(agent_tools.items()):
        print(f"  {tool}: {count}个智能体使用")
    
    # 分析状态类型
    initial_states = [s for s in fsm_mas_config['fsm']['states'] if s['is_initial']]
    final_states = [s for s in fsm_mas_config['fsm']['states'] if s['is_final']]
    
    print(f"\n🔄 状态类型分析:")
    print(f"  初始状态: {len(initial_states)}个")
    print(f"  最终状态: {len(final_states)}个")
    print(f"  中间状态: {len(fsm_mas_config['fsm']['states']) - len(initial_states) - len(final_states)}个")
    
    # 分析监听关系
    total_listeners = 0
    for state in fsm_mas_config['fsm']['states']:
        total_listeners += len(state.get('listener', []))
    
    print(f"\n📡 通信分析:")
    print(f"  总监听关系: {total_listeners}个")
    print(f"  平均每状态监听者: {total_listeners / len(fsm_mas_config['fsm']['states']):.1f}个")
    
    return fsm_mas_config


def main():
    """运行所有示例"""
    print("🌟 Neural FSM-MAS 完整示例演示")
    print("="*80)
    
    try:
        # 运行所有示例
        example_1_basic_fsm_mas_generation()
        example_2_tgn_learning()
        example_3_training_with_data()
        example_4_demonstrate_capabilities()
        example_5_analyze_generated_system()
        
        print("\n🎉 所有示例运行完成！")
        print("📁 查看生成的文件:")
        print("  - ./example_fsm_mas_system.json")
        print("  - ./example_training_outputs/")
        print("  - ./demo_outputs/")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
