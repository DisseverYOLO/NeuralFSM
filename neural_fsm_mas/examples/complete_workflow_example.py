"""
Complete Workflow Example
完整工作流程示例

演示集成MultiAgent.py后的完整功能：
1. 生成智能体和FSM状态 (MetaAgent功能)
2. 随机采样拓扑图
3. TGN学习最优路径
4. 执行任务 (MultiAgent.py功能)
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_fsm_mas import (
    FSMMultiAgentSystemGenerator,
    generate_learn_and_execute_fsm_mas
)


def example_complete_workflow():
    """示例：完整的生成-学习-执行工作流程"""
    print("🌟 完整工作流程示例")
    print("="*60)
    
    # 任务描述（用于生成FSM）
    task_description = "Solve mathematical word problems step by step with verification"
    
    # 具体任务输入（用于执行）
    task_input = "A store has 120 apples. They sell 30% of them in the morning and 25% of the remaining apples in the afternoon. How many apples are left?"
    
    # 可用工具
    available_tools = ['calculator', 'logical_reasoning', 'verification']
    
    print(f"📋 任务描述: {task_description}")
    print(f"🎯 具体任务: {task_input}")
    print(f"🔧 可用工具: {available_tools}")
    
    # 执行完整工作流程
    results = generate_learn_and_execute_fsm_mas(
        task_description=task_description,
        task_input=task_input,
        available_tools=available_tools,
        training_episodes=20,  # 示例用较少轮数
        max_transitions=10
    )
    
    # 显示结果
    print("\n🎉 完整工作流程执行完成！")
    print("="*60)
    
    # FSM生成结果
    fsm_config = results['fsm_mas_config']
    print(f"🤖 生成智能体数量: {fsm_config['system_metadata']['num_agents']}")
    print(f"🔄 生成状态数量: {fsm_config['system_metadata']['num_states']}")
    
    # TGN学习结果
    learning_results = results['learning_results']
    print(f"🧠 TGN训练轮数: {learning_results['training_episodes']}")
    print(f"📊 最终状态损失: {learning_results['final_state_loss']:.4f}")
    print(f"📊 最终通信损失: {learning_results['final_communication_loss']:.4f}")
    
    # 执行结果
    print(f"✅ 任务执行结果: {results['execution_result']}")
    print(f"💰 执行成本: {results['execution_cost']}")
    
    return results


def example_step_by_step_workflow():
    """示例：分步骤的工作流程"""
    print("\n🎯 分步骤工作流程示例")
    print("="*60)
    
    # 创建生成器
    generator = FSMMultiAgentSystemGenerator()
    
    # Step 1: 生成FSM-MAS系统
    print("📝 Step 1: 生成FSM-MAS系统...")
    task_description = "Analyze text sentiment and provide recommendations"
    fsm_config = generator.generate_complete_fsm_mas(
        task_description=task_description,
        available_tools=['analysis', 'logical_reasoning']
    )
    
    print(f"✅ 生成了 {fsm_config['system_metadata']['num_agents']} 个智能体")
    print(f"✅ 生成了 {fsm_config['system_metadata']['num_states']} 个状态")
    
    # 显示生成的智能体
    print("\n🤖 生成的智能体:")
    for agent in fsm_config['agents']:
        print(f"  - {agent['name']} (工具: {agent['tools']})")
    
    # 显示生成的状态
    print("\n🔄 生成的FSM状态:")
    for state in fsm_config['fsm']['states']:
        print(f"  - 状态 {state['state_id']}: {state['instruction'][:50]}...")
    
    # Step 2: TGN学习
    print("\n🧠 Step 2: TGN学习最优拓扑...")
    learning_results = generator.learn_optimal_topologies(training_episodes=15)
    
    print(f"✅ 学习完成，状态损失: {learning_results['final_state_loss']:.4f}")
    print(f"✅ 通信损失: {learning_results['final_communication_loss']:.4f}")
    
    # Step 3: 创建可执行系统
    print("\n🔧 Step 3: 创建可执行系统...")
    executable_system = generator.create_executable_system()
    print("✅ 可执行系统创建完成")
    
    # Step 4: 执行任务
    print("\n🚀 Step 4: 执行具体任务...")
    task_input = "This product is amazing! I love how easy it is to use and the quality is outstanding."
    
    result, cost = generator.execute_task(task_input, max_transitions=8)
    
    print(f"✅ 执行完成")
    print(f"📊 结果: {result}")
    print(f"💰 成本: {cost}")
    
    return {
        'fsm_config': fsm_config,
        'learning_results': learning_results,
        'execution_result': result,
        'execution_cost': cost
    }


def example_comparison_with_without_learning():
    """示例：对比有无TGN学习的差异"""
    print("\n🔬 对比有无TGN学习的差异")
    print("="*60)
    
    task_description = "Debug and fix Python code errors"
    task_input = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This code has a potential division by zero error
result = calculate_average([])
print(result)
"""
    
    # 创建两个生成器
    generator_with_learning = FSMMultiAgentSystemGenerator()
    generator_without_learning = FSMMultiAgentSystemGenerator(use_neural_learning=False)
    
    print("🧠 测试1: 使用TGN学习的系统")
    print("-" * 30)
    
    # 有学习的系统
    fsm_config1 = generator_with_learning.generate_complete_fsm_mas(
        task_description, ['code_interpreter', 'analysis']
    )
    learning_results1 = generator_with_learning.learn_optimal_topologies(10)
    result1, cost1 = generator_with_learning.execute_task(task_input)
    
    print(f"✅ 结果: {result1[:100]}...")
    print(f"💰 成本: {cost1}")
    
    print("\n🔧 测试2: 不使用TGN学习的系统")
    print("-" * 30)
    
    # 无学习的系统
    fsm_config2 = generator_without_learning.generate_complete_fsm_mas(
        task_description, ['code_interpreter', 'analysis']
    )
    result2, cost2 = generator_without_learning.execute_task(task_input)
    
    print(f"✅ 结果: {result2[:100]}...")
    print(f"💰 成本: {cost2}")
    
    print(f"\n📊 对比总结:")
    print(f"  TGN学习系统成本: {cost1}")
    print(f"  传统系统成本: {cost2}")
    print(f"  成本差异: {cost1 - cost2:.4f}")
    
    return {
        'with_learning': {'result': result1, 'cost': cost1},
        'without_learning': {'result': result2, 'cost': cost2}
    }


def main():
    """运行所有示例"""
    print("🌟 Neural FSM-MAS 完整工作流程演示")
    print("="*80)
    print("现在集成了MultiAgent.py，支持完整的生成-学习-执行流程！")
    print("="*80)
    
    try:
        # 示例1: 完整工作流程
        results1 = example_complete_workflow()
        
        # 示例2: 分步骤工作流程
        results2 = example_step_by_step_workflow()
        
        # 示例3: 对比有无学习
        results3 = example_comparison_with_without_learning()
        
        print("\n🎉 所有示例运行完成！")
        print("="*80)
        print("📋 总结:")
        print("✅ 成功集成了MetaAgent的生成能力")
        print("✅ 成功实现了随机拓扑采样")
        print("✅ 成功实现了TGN学习优化")
        print("✅ 成功集成了MultiAgent.py的执行能力")
        print("✅ 实现了完整的生成-学习-执行工作流程")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
