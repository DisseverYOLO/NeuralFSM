"""
多数据集训练脚本
Multi-Dataset Training Script

支持MMLU、GSM8K、HumanEval等多个数据集的训练
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neural_fsm_mas.train_fsm_mas import NeuralFSMTrainer
from neural_fsm_mas.training_data.mmlu_data_processor import MMLUDataProcessor
from neural_fsm_mas.training_data.gsm8k_data_processor import GSM8KDataProcessor


# 数据集配置
DATASET_CONFIGS = {
    "mmlu": {
        "name": "MMLU",
        "description": "Massive Multitask Language Understanding - 多学科知识问答",
        "task_description": """
        Answer multiple-choice questions across various academic domains including STEM, 
        humanities, social sciences, and other subjects. Select the correct answer from 
        four options (A, B, C, D) based on your knowledge and reasoning.
        """,
        "default_path": "datasets/MMLU/data/test",
        "processor_class": MMLUDataProcessor
    },
    
    "gsm8k": {
        "name": "GSM8K",
        "description": "Grade School Math 8K - 小学数学问题",
        "task_description": """
        Solve grade school math problems using step-by-step reasoning.
        The problems involve arithmetic operations, word problems, and algebraic thinking.
        You need to understand the problem, break it down into steps, and calculate the final answer.
        """,
        "default_path": "datasets/gsm8k/gsm8k.jsonl",
        "processor_class": GSM8KDataProcessor
    },
    
    "humaneval": {
        "name": "HumanEval",
        "description": "Human Eval - 代码生成评估（待实现）",
        "task_description": """
        Write Python functions that pass given test cases. Understand the function signature,
        docstring requirements, and implement correct, efficient code.
        """,
        "default_path": "datasets/humaneval/humaneval-py.jsonl",
        "processor_class": None  # 待实现
    }
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="NeuralFSM Multi-Dataset Training",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 数据集选择
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mmlu", "gsm8k", "humaneval"],
        required=True,
        help="选择数据集:\n" + 
             "\n".join([f"  {k}: {v['description']}" for k, v in DATASET_CONFIGS.items()])
    )
    
    # 数据路径
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="数据集路径（默认使用配置中的路径）"
    )
    
    # 训练参数
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=30,
        help="训练Episodes数量 (默认: 30)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="学习率 (默认: 0.001)"
    )
    
    parser.add_argument(
        "--policy_gradient_weight",
        type=float,
        default=0.7,
        help="策略梯度损失权重 (默认: 0.7)"
    )
    
    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        default=0.3,
        help="MSE重构损失权重 (默认: 0.3)"
    )
    
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=100,
        help="训练样本数量 (默认: 100)"
    )
    
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=50,
        help="验证样本数量 (默认: 50)"
    )
    
    parser.add_argument(
        "--memory_dimension",
        type=int,
        default=128,
        help="TGN记忆维度 (默认: 128)"
    )
    
    parser.add_argument(
        "--temporal_dimension",
        type=int,
        default=32,
        help="TGN时间维度 (默认: 32)"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    return parser.parse_args()


def load_dataset(dataset_name: str, data_path: str, args):
    """
    加载指定数据集
    
    Args:
        dataset_name: 数据集名称
        data_path: 数据路径
        args: 命令行参数
        
    Returns:
        (formatted_train_data, formatted_val_data)
    """
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"\n📂 加载数据集: {config['name']}")
    print(f"   描述: {config['description']}")
    
    if dataset_name == "mmlu":
        processor = MMLUDataProcessor(data_path)
        domain_data = processor.load_and_split_by_domain()
        
        # 使用第一个领域作为示例
        first_domain = list(domain_data.keys())[0]
        train_data = domain_data[first_domain]['train']
        val_data = domain_data[first_domain]['validation']
        
        # 采样
        train_data_subset = train_data[:args.num_train_samples]
        val_data_subset = val_data[:args.num_val_samples]
        
        # 格式化
        formatted_train = processor.format_for_mas_training(train_data_subset)
        formatted_val = processor.format_for_mas_training(val_data_subset)
        
    elif dataset_name == "gsm8k":
        processor = GSM8KDataProcessor(data_path)
        processor.load_gsm8k_data()
        processor.process_gsm8k_data()
        train_data, val_data = processor.split_train_val(train_ratio=0.8, shuffle=True)
        
        # 采样
        train_data_subset = processor.get_train_data(args.num_train_samples)
        val_data_subset = processor.get_val_data(args.num_val_samples)
        
        # 格式化
        formatted_train = processor.format_for_mas_training(train_data_subset)
        formatted_val = processor.format_for_mas_training(val_data_subset)
    
    elif dataset_name == "humaneval":
        raise NotImplementedError("HumanEval数据集处理器尚未实现")
    
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    print(f"✅ 数据加载完成:")
    print(f"   训练数据: {len(formatted_train)} 条")
    print(f"   验证数据: {len(formatted_val)} 条")
    
    return formatted_train, formatted_val


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("🚀 NeuralFSM - 多数据集训练系统")
    print("=" * 80)
    
    # 获取数据集配置
    dataset_config = DATASET_CONFIGS[args.dataset]
    
    # 确定数据路径
    if args.data_path is None:
        data_path = project_root / dataset_config["default_path"]
    else:
        data_path = Path(args.data_path)
    
    if not data_path.exists():
        print(f"❌ 错误: 数据路径不存在: {data_path}")
        print(f"请确保数据集已下载并放置在正确位置")
        return
    
    # 显示配置
    print(f"\n📋 训练配置:")
    print(f"  数据集: {dataset_config['name']}")
    print(f"  数据路径: {data_path}")
    print(f"  训练Episodes: {args.num_episodes}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  策略梯度权重: {args.policy_gradient_weight}")
    print(f"  MSE重构权重: {args.reconstruction_weight}")
    print(f"  训练样本数: {args.num_train_samples}")
    print(f"  验证样本数: {args.num_val_samples}")
    print(f"  记忆维度: {args.memory_dimension}")
    print(f"  时间维度: {args.temporal_dimension}")
    
    # 加载数据
    formatted_train, formatted_val = load_dataset(args.dataset, str(data_path), args)
    
    # 初始化训练器
    print("\n🤖 初始化NeuralFSM训练器...")
    trainer = NeuralFSMTrainer(
        use_neural_learning=True,
        memory_dimension=args.memory_dimension,
        temporal_dimension=args.temporal_dimension,
        random_seed=args.random_seed
    )
    print("✅ 训练器初始化完成")
    
    # 开始训练
    print("\n🎯 开始训练...")
    print("-" * 80)
    
    results = trainer.train_fsm_mas_with_tgn(
        task_description=dataset_config["task_description"],
        training_data=formatted_train,
        validation_data=formatted_val,
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        policy_gradient_weight=args.policy_gradient_weight,
        reconstruction_weight=args.reconstruction_weight,
        save_results=True,
        result_file_prefix=args.dataset
    )
    
    # 显示结果
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
    
    print(f"\n🎉 {dataset_config['name']} 训练完成！")


if __name__ == "__main__":
    main()

