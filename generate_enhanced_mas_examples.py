"""
Enhanced MAS Generation Examples
增强版多智能体系统生成示例

演示如何使用Enhanced_FSM_Gen.py为不同数据集生成定制化的FSM和Agent
"""

import sys
from pathlib import Path
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator, generate_enhanced_mas


def example_1_gsm8k():
    """
    示例1: 为GSM8K数学题生成MAS
    
    特点:
    - 细分计算过程的状态
    - 包含验证和纠错机制
    - 明确的数值完成条件
    """
    print("=" * 80)
    print("示例1: 生成GSM8K数学题求解系统")
    print("=" * 80)
    
    generator = EnhancedFSMGenerator(use_azure=False)
    
    # 自定义任务描述（可选）
    custom_task = """
    设计一个多智能体系统来解决GSM8K数学应用题。
    
    要求:
    1. 能够处理多步骤数学推理
    2. 支持中间计算验证
    3. 能够识别并纠正计算错误
    4. 最终答案为精确数值
    
    示例问题:
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning 
    and bakes muffins for her friends every day with four. She sells the remainder 
    at the farmers' market daily for $2 per fresh duck egg. How much in dollars does 
    she make every day at the farmers' market?"
    """
    
    # 生成完整系统
    save_path = "workspace/gsm8k_enhanced_mas.json"
    system, cost = generator.generate_complete_mas(
        dataset='gsm8k',
        save_path=save_path,
        task_description=custom_task
    )
    
    # 展示生成结果
    print("\n" + "=" * 80)
    print("生成结果摘要:")
    print("=" * 80)
    print(f"Agent数量: {len(system['agents'])}")
    print(f"状态数量: {len(system['fsm']['states'])}")
    print(f"转移数量: {len(system['fsm']['transitions'])}")
    print(f"Token成本: {cost:.2f}")
    
    # 展示Agent角色
    print("\n📝 Agent列表:")
    for agent in system['agents']:
        print(f"  - {agent['name']} (ID: {agent['agent_id']})")
        if 'role' in agent:
            print(f"    角色: {agent['role']}")
        if 'state_completion_criteria' in agent:
            print(f"    完成标准: {agent['state_completion_criteria']}")
    
    # 展示状态序列
    print("\n🔄 状态序列:")
    for state in system['fsm']['states']:
        marker = "🚀" if state['is_initial'] else ("🎯" if state['is_final'] else "📍")
        print(f"  {marker} State {state['state_id']}: {state['state_name']}")
        print(f"     负责Agent: {state['agent_id']}")
        if 'completion_condition' in state:
            print(f"     完成条件: {state['completion_condition'][:60]}...")
    
    print(f"\n💾 完整配置已保存到: {save_path}")
    return system


def example_2_mmlu():
    """
    示例2: 为MMLU多领域知识问答生成MAS
    
    特点:
    - 多领域知识整合
    - 选项分析和排除
    - 置信度评估
    """
    print("\n\n" + "=" * 80)
    print("示例2: 生成MMLU知识问答系统")
    print("=" * 80)
    
    save_path = "workspace/mmlu_enhanced_mas.json"
    system, cost = generate_enhanced_mas('mmlu', save_path, use_azure=False)
    
    print(f"\n✅ 生成完成!")
    print(f"   - {len(system['agents'])} 个Agent")
    print(f"   - {len(system['fsm']['states'])} 个状态")
    print(f"   - Token成本: {cost:.2f}")
    print(f"   - 保存路径: {save_path}")
    
    return system


def example_3_humaneval():
    """
    示例3: 为HumanEval代码生成生成MAS
    
    特点:
    - 需求分析到代码实现的完整流程
    - 测试驱动开发
    - 代码调试和优化
    """
    print("\n\n" + "=" * 80)
    print("示例3: 生成HumanEval代码生成系统")
    print("=" * 80)
    
    save_path = "workspace/humaneval_enhanced_mas.json"
    system, cost = generate_enhanced_mas('humaneval', save_path, use_azure=False)
    
    print(f"\n✅ 生成完成!")
    print(f"   - {len(system['agents'])} 个Agent")
    print(f"   - {len(system['fsm']['states'])} 个状态")
    print(f"   - Token成本: {cost:.2f}")
    print(f"   - 保存路径: {save_path}")
    
    return system


def example_4_custom_agents_only():
    """
    示例4: 仅生成Agent描述（不生成FSM）
    
    适用场景: 想先查看Agent设计，再决定FSM结构
    """
    print("\n\n" + "=" * 80)
    print("示例4: 仅生成Agent描述")
    print("=" * 80)
    
    generator = EnhancedFSMGenerator(use_azure=False)
    
    # 仅生成Agent
    agent_dict, agent_llm = generator.generate_agents('gsm8k')
    
    print(f"\n✅ 生成了 {len(agent_dict)} 个Agent")
    
    # 详细展示第一个Agent
    print("\n📋 示例Agent详情 (Agent 0):")
    agent_0 = agent_dict[0]
    print(json.dumps(agent_0, indent=2, ensure_ascii=False))
    
    # 保存仅Agent的配置
    agents_only_path = "workspace/gsm8k_agents_only.json"
    with open(agents_only_path, 'w', encoding='utf-8') as f:
        json.dump({"agents": agent_dict}, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Agent配置已保存到: {agents_only_path}")
    print(f"   Token成本: {agent_llm.get_token_cost():.2f}")
    
    return agent_dict


def example_5_custom_fsm_only():
    """
    示例5: 基于已有Agent生成FSM
    
    适用场景: 已经有Agent定义，想生成对应的FSM
    """
    print("\n\n" + "=" * 80)
    print("示例5: 基于已有Agent生成FSM")
    print("=" * 80)
    
    # 加载之前生成的Agent
    agents_path = "workspace/gsm8k_agents_only.json"
    try:
        with open(agents_path, 'r', encoding='utf-8') as f:
            agents_data = json.load(f)
            agent_dict = agents_data['agents']
        
        print(f"✅ 加载了 {len(agent_dict)} 个Agent")
        
        # 生成FSM
        generator = EnhancedFSMGenerator(use_azure=False)
        fsm, fsm_llm = generator.generate_fsm_states('gsm8k', agent_dict)
        
        print(f"\n✅ 生成了 {len(fsm['states'])} 个状态")
        print(f"   Token成本: {fsm_llm.get_token_cost():.2f}")
        
        # 保存完整系统
        complete_system = {
            "dataset": "gsm8k",
            "agents": agent_dict,
            "fsm": fsm
        }
        
        complete_path = "workspace/gsm8k_complete_from_agents.json"
        with open(complete_path, 'w', encoding='utf-8') as f:
            json.dump(complete_system, f, indent=2, ensure_ascii=False)
        
        print(f"💾 完整系统已保存到: {complete_path}")
        
        return complete_system
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {agents_path}")
        print("   请先运行 example_4_custom_agents_only()")
        return None


def example_6_analyze_completion_conditions():
    """
    示例6: 分析生成的完成条件
    
    展示如何检查和使用completion_condition
    """
    print("\n\n" + "=" * 80)
    print("示例6: 分析状态完成条件")
    print("=" * 80)
    
    # 加载已生成的系统
    system_path = "workspace/gsm8k_enhanced_mas.json"
    try:
        with open(system_path, 'r', encoding='utf-8') as f:
            system = json.load(f)
        
        print(f"✅ 加载系统: {len(system['fsm']['states'])} 个状态\n")
        
        # 分析每个状态的完成条件
        print("📋 状态完成条件分析:")
        print("=" * 80)
        
        for state in system['fsm']['states']:
            print(f"\n🔹 State {state['state_id']}: {state['state_name']}")
            print(f"   负责Agent: {state['agent_id']}")
            
            if 'completion_condition' in state:
                condition = state['completion_condition']
                print(f"   完成条件: {condition}")
                
                # 分析条件类型
                if "contains" in condition.lower() and "<" in condition:
                    marker = condition.split("<")[1].split(">")[0] if "<" in condition else "Unknown"
                    print(f"   ✓ 类型: 结构化标记检查")
                    print(f"   ✓ 标记: <{marker}>")
                
                elif "field" in condition.lower() or "has" in condition.lower():
                    print(f"   ✓ 类型: 字段存在性检查")
                
                elif "passes" in condition.lower() or "validation" in condition.lower():
                    print(f"   ✓ 类型: 验证逻辑检查")
                
                else:
                    print(f"   ✓ 类型: 语义条件检查")
            else:
                print(f"   ⚠️  未定义completion_condition")
        
        # 统计完成条件类型
        print("\n" + "=" * 80)
        print("📊 完成条件统计:")
        
        total_states = len(system['fsm']['states'])
        states_with_conditions = sum(1 for s in system['fsm']['states'] 
                                     if 'completion_condition' in s and s['completion_condition'])
        
        print(f"   - 总状态数: {total_states}")
        print(f"   - 有完成条件的状态: {states_with_conditions}")
        print(f"   - 覆盖率: {states_with_conditions/total_states*100:.1f}%")
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {system_path}")
        print("   请先运行 example_1_gsm8k()")


def example_7_compare_prompts():
    """
    示例7: 对比原始和增强版生成的提示词
    
    展示增强版在提示词质量上的改进
    """
    print("\n\n" + "=" * 80)
    print("示例7: 提示词质量对比")
    print("=" * 80)
    
    print("\n📝 原始版 (FSM_Gen.py) Agent提示词示例:")
    print("-" * 80)
    original_prompt = """
    You are the designer of a multi-agent system. Given a general task description,
    you first need to design several agents that can cooperatively solve this type of task.
    
    Each agent contain three features:
    - name: <The name of the agent>
    - system_prompt: <The system prompt for agent>
    - tools: <The equiped tool name, a list>
    """
    print(original_prompt)
    
    print("\n📝 增强版 (Enhanced_FSM_Gen.py) Agent提示词示例:")
    print("-" * 80)
    enhanced_prompt = """
    You are an expert Multi-Agent System architect specializing in GSM8K tasks.
    
    📋 TASK CONTEXT:
    数学应用题求解任务
    特点:
    - 问题涉及多步骤推理
    - 需要精确的数值计算
    ...
    
    🎯 YOUR MISSION:
    Design a team of specialized agents that can COOPERATIVELY solve GSM8K problems
    with high accuracy.
    
    Agent设计建议:
    1. 问题理解者(Problem Analyzer): 提取问题关键信息和数量关系
    2. 方案规划者(Solution Planner): 制定分步求解方案
    ...
    
    🚨 CRITICAL REQUIREMENTS:
    1. **Granular Specialization**: Each agent should have a NARROW, well-defined responsibility
    2. **State Completion Criteria**: For EACH agent, specify HOW to determine if its task is completed
       - Use observable outputs (e.g., "output contains calculation result")
       - Use structured markers (e.g., "output ends with <DONE> tag")
    ...
    """
    print(enhanced_prompt)
    
    print("\n" + "=" * 80)
    print("🆚 对比分析:")
    print("=" * 80)
    
    comparison = """
    | 维度 | 原始版 | 增强版 |
    |------|--------|--------|
    | 任务上下文 | 通用 | 针对数据集定制 |
    | Agent职责 | 模糊 | 明确细分 |
    | 完成标准 | 缺失 | 每个Agent都有 |
    | 输出格式 | 未要求 | 结构化要求 |
    | 提示词长度 | ~200词 | ~800词 |
    | 数据集适配 | 无 | GSM8K/MMLU/HumanEval |
    | 状态细分指导 | 无 | 详细的状态数量和划分建议 |
    """
    print(comparison)
    
    print("\n💡 增强版的核心改进:")
    print("  1. ✅ 针对数据集特性的任务上下文")
    print("  2. ✅ 明确的Agent职责和完成标准")
    print("  3. ✅ 结构化的输出格式要求")
    print("  4. ✅ 细粒度状态划分指导")
    print("  5. ✅ 完成条件的多层次设计")


def main():
    """
    主函数: 运行所有示例
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Enhanced MAS Generation Examples" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 确保workspace目录存在
    Path("workspace").mkdir(exist_ok=True)
    
    # 运行示例（根据需要注释/取消注释）
    
    # 示例1: 生成GSM8K完整系统
    # system_gsm8k = example_1_gsm8k()
    
    # 示例2: 生成MMLU完整系统
    # system_mmlu = example_2_mmlu()
    
    # 示例3: 生成HumanEval完整系统
    # system_humaneval = example_3_humaneval()
    
    # 示例4: 仅生成Agent
    # agents = example_4_custom_agents_only()
    
    # 示例5: 基于已有Agent生成FSM
    # system = example_5_custom_fsm_only()
    
    # 示例6: 分析完成条件
    # example_6_analyze_completion_conditions()
    
    # 示例7: 提示词对比
    example_7_compare_prompts()
    
    print("\n" + "=" * 80)
    print("🎉 所有示例运行完成!")
    print("=" * 80)
    print("\n💡 提示:")
    print("   - 生成的配置文件保存在 workspace/ 目录")
    print("   - 可以编辑本文件来运行不同的示例")
    print("   - 根据需要注释/取消注释 main() 中的示例调用")


if __name__ == "__main__":
    main()



