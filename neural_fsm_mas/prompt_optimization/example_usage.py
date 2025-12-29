"""
提示词优化框架使用示例
Prompt Optimization Framework Example Usage
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_fsm_mas.prompt_optimization import (
    create_prompt_optimizer,
    PromptOptimizationConfig
)


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("示例1: 基本使用")
    print("=" * 80)
    
    # 创建优化器
    config = PromptOptimizationConfig(
        enable_ab_test=True,
        ab_test_ratio=0.1,
        auto_switch_best=True,
        switch_interval=50,
        few_shot_strategy="hybrid",
        max_few_shot_examples=3
    )
    
    optimizer = create_prompt_optimizer('gsm8k', config)
    
    # 注册多个模板版本
    optimizer.template_manager.register_template(
        template_id="basic",
        template_name="Basic Template",
        system_prompt="You are a helpful assistant. Solve math problems step by step."
    )
    
    optimizer.template_manager.register_template(
        template_id="detailed",
        template_name="Detailed Template",
        system_prompt="You are an expert mathematician. Think carefully and show all your work. "
                     "Break down the problem into steps and explain each step clearly."
    )
    
    # 添加Few-shot示例
    optimizer.few_shot_selector.add_example(
        example_id="ex1",
        question="What is 2 + 2?",
        answer="4",
        reasoning="Adding 2 and 2 gives 4."
    )
    
    optimizer.few_shot_selector.add_example(
        example_id="ex2",
        question="What is 3 * 4?",
        answer="12",
        reasoning="Multiplying 3 by 4 gives 12."
    )
    
    # 获取优化后的提示词
    prompt, metadata = optimizer.get_prompt(
        question="What is 5 + 7?",
        agent_role="MathematicalReasoner"
    )
    
    print(f"\n生成的提示词:\n{prompt[:200]}...")
    print(f"\n元数据: {metadata}")
    
    # 模拟记录结果
    optimizer.record_result(
        template_id=metadata['template_id'],
        is_correct=True,
        response_time=1.2,
        token_usage=150,
        cost=0.001,
        few_shot_example_ids=metadata.get('few_shot_example_ids', [])
    )
    
    # 查看性能摘要
    summary = optimizer.get_performance_summary()
    print(f"\n性能摘要:\n{summary}")


def example_ab_testing():
    """A/B测试示例"""
    print("\n" + "=" * 80)
    print("示例2: A/B测试")
    print("=" * 80)
    
    optimizer = create_prompt_optimizer('gsm8k')
    
    # 注册两个版本的模板
    optimizer.template_manager.register_template(
        template_id="version_a",
        template_name="Version A - Concise",
        system_prompt="Solve problems concisely."
    )
    
    optimizer.template_manager.register_template(
        template_id="version_b",
        template_name="Version B - Detailed",
        system_prompt="Solve problems with detailed explanations."
    )
    
    # 模拟多次调用，进行A/B测试
    print("\n进行A/B测试...")
    for i in range(10):
        prompt, metadata = optimizer.get_prompt(question=f"Question {i}")
        is_test = metadata.get('is_test', False)
        print(f"  调用 {i+1}: 模板={metadata['template_id']}, 测试={is_test}")
        
        # 模拟结果（假设version_b更好）
        is_correct = True if metadata['template_id'] == 'version_b' else (i % 2 == 0)
        optimizer.record_result(
            template_id=metadata['template_id'],
            is_correct=is_correct,
            response_time=1.0,
            token_usage=100,
            cost=0.001
        )
    
    # 查看性能对比
    summary = optimizer.get_performance_summary()
    print("\n性能对比:")
    for tid, perf in summary['template_summary']['templates'].items():
        print(f"  {tid}: 准确率={perf['accuracy']:.2%}, 样本数={perf['total_count']}")


def example_few_shot_selection():
    """Few-shot选择示例"""
    print("\n" + "=" * 80)
    print("示例3: Few-shot动态选择")
    print("=" * 80)
    
    optimizer = create_prompt_optimizer('gsm8k')
    
    # 添加多个示例
    examples = [
        ("ex1", "What is 2+2?", "4", "Simple addition"),
        ("ex2", "What is 10*5?", "50", "Multiplication"),
        ("ex3", "What is 20/4?", "5", "Division"),
        ("ex4", "What is 3^2?", "9", "Exponentiation"),
    ]
    
    for ex_id, question, answer, reasoning in examples:
        optimizer.few_shot_selector.add_example(
            example_id=ex_id,
            question=question,
            answer=answer,
            reasoning=reasoning
        )
    
    # 测试不同策略
    strategies = ["similarity", "performance", "hybrid"]
    
    for strategy in strategies:
        print(f"\n策略: {strategy}")
        optimizer.config.few_shot_strategy = strategy
        prompt, metadata = optimizer.get_prompt(question="What is 15+8?")
        selected_ids = metadata.get('few_shot_example_ids', [])
        print(f"  选择的示例: {selected_ids}")


if __name__ == "__main__":
    example_basic_usage()
    example_ab_testing()
    example_few_shot_selection()
    
    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("=" * 80)

