"""
GSM8K训练脚本
Run GSM8K Training

使用GSM8K数据集训练NeuralFSM多智能体系统
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neural_fsm_mas.train_fsm_mas import NeuralFSMTrainer
from neural_fsm_mas.training_data.gsm8k_data_processor import GSM8KDataProcessor


def main():
    """GSM8K训练主函数"""
    
    print("=" * 80)
    print("🚀 NeuralFSM - GSM8K数学推理训练")
    print("=" * 80)
    
    # ========== 1. 配置参数 ==========
    print("\n📋 训练配置:")
    
    # GSM8K数据路径（需要根据实际情况修改）
    gsm8k_data_path = project_root / "datasets" / "gsm8k" / "gsm8k.jsonl"
    
    if not gsm8k_data_path.exists():
        print(f"❌ 错误: GSM8K数据文件不存在: {gsm8k_data_path}")
        print(f"请下载GSM8K数据集并放置在: {gsm8k_data_path}")
        return
    
    # 训练参数
    num_episodes = 30
    learning_rate = 0.001
    policy_gradient_weight = 0.7
    reconstruction_weight = 0.3
    num_train_samples = 100  # 使用100个训练样本
    num_val_samples = 50     # 使用50个验证样本
    
    print(f"  数据集: GSM8K (数学推理)")
    print(f"  训练Episodes: {num_episodes}")
    print(f"  学习率: {learning_rate}")
    print(f"  策略梯度权重: {policy_gradient_weight}")
    print(f"  MSE重构权重: {reconstruction_weight}")
    print(f"  训练样本数: {num_train_samples}")
    print(f"  验证样本数: {num_val_samples}")
    
    # ========== 2. 加载数据 ==========
    print("\n📂 加载GSM8K数据...")
    
    processor = GSM8KDataProcessor(str(gsm8k_data_path))
    processor.load_gsm8k_data()
    processor.process_gsm8k_data()
    train_data, val_data = processor.split_train_val(train_ratio=0.8, shuffle=True)
    
    # 获取子集
    train_data_subset = processor.get_train_data(num_train_samples)
    val_data_subset = processor.get_val_data(num_val_samples)
    
    # 格式化数据
    formatted_train = processor.format_for_mas_training(train_data_subset)
    formatted_val = processor.format_for_mas_training(val_data_subset)
    
    print(f"✅ 数据加载完成:")
    print(f"   训练数据: {len(formatted_train)} 条")
    print(f"   验证数据: {len(formatted_val)} 条")
    
    # ========== 3. 初始化训练器 ==========
    print("\n🤖 初始化NeuralFSM训练器...")
    
    trainer = NeuralFSMTrainer(
        use_neural_learning=True,
        memory_dimension=128,
        temporal_dimension=32,
        random_seed=42
    )
    
    print("✅ 训练器初始化完成")
    
    # ========== 4. 开始训练 ==========
    print("\n🎯 开始训练...")
    print("-" * 80)
    
    task_description = """
    Solve grade school math problems using step-by-step reasoning.
    The problems involve arithmetic operations, word problems, and algebraic thinking.
    You need to understand the problem, break it down into steps, and calculate the final answer.
    """
    
    results = trainer.train_fsm_mas_with_tgn(
        task_description=task_description,
        training_data=formatted_train,
        validation_data=formatted_val,
        num_episodes=num_episodes,
        learning_rate=learning_rate,
        policy_gradient_weight=policy_gradient_weight,
        reconstruction_weight=reconstruction_weight,
        save_results=True,
        result_file_prefix="gsm8k"
    )
    
    # ========== 5. 显示结果 ==========
    print("\n" + "=" * 80)
    print("📊 训练完成！最终结果:")
    print("=" * 80)
    
    print(f"\n✅ 最终训练准确率: {results['final_training_accuracy']:.2%}")
    print(f"✅ 最终验证准确率: {results['final_validation_accuracy']:.2%}")
    print(f"✅ 策略梯度损失: {results['final_policy_gradient_loss']:.4f}")
    print(f"✅ MSE重构损失: {results['final_reconstruction_loss']:.4f}")
    print(f"✅ 组合损失: {results['final_combined_loss']:.4f}")
    
    print(f"\n💾 训练记录已保存到:")
    print(f"   {results['result_file']}")
    print(f"   {results['system_file']}")
    
    print("\n🎉 GSM8K训练完成！")


if __name__ == "__main__":
    main()

