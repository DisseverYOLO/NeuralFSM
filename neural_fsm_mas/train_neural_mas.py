"""
Neural Multi-Agent System Training Script
神经多智能体系统训练脚本

整合MetaAgent和GDesigner功能，使用多数据集训练TGN网络
支持数据集：MMLU、GSM8K、HumanEval
学习状态转移规则和智能体通信网络

注意：这是简化版的协作式MAS训练器
- 使用预定义的智能体配置（来自DomainPromptManager）
- 多轮交互模式（num_rounds轮协作）
- 策略梯度损失优化

如需完整的FSM自动生成功能，请使用 train_fsm_mas.py
"""

import asyncio
import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# 添加路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph, MultiLayerPerceptron
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor
from neural_fsm_mas.domain_prompts.prompt_manager import DomainPromptManager
from baseclass.FSM_Gen import Generate_Agent_Description, Generate_State_Description


class NeuralMASTrainer:
    """
    神经多智能体系统训练器
    
    核心功能：
    1. 使用MMLU数据集训练TGN网络
    2. 学习最优的智能体通信网络拓扑
    3. 学习有限状态机的状态转移规则
    4. 支持多领域的分别训练和评估
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 dataset_root: str = "./datasets",
                 output_dir: str = "./neural_mas_outputs",
                 mmlu_data_path: str = None):  # 保留兼容性
        
        self.config = config
        self.dataset_root = dataset_root if not mmlu_data_path else Path(mmlu_data_path).parent.parent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统一数据处理器
        self.data_processor = UnifiedDataProcessor(str(self.dataset_root))
        
        # 训练状态
        self.training_history = []
        self.domain_topologies = {}
        self.best_models = {}
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def prepare_training_data(self, domains: List[str] = None):
        """
        准备训练数据
        
        Args:
            domains: 要准备的数据集列表，如['mmlu', 'gsm8k', 'humaneval']
                    如果为None，则默认使用config中的domains或只使用mmlu
        """
        if domains is None:
            domains = self.config.get('domains', ['mmlu'])
        
        print(f"🔄 准备训练数据: {', '.join(domains)}")
        
        statistics = {}
        
        for domain in domains:
            try:
                print(f"\n  处理 {domain.upper()}...")
                
                # 创建数据分割
                self.data_processor.create_domain_splits(
                    domain=domain,
                    train_ratio=self.config.get('train_ratio', 0.7),
                    val_ratio=self.config.get('val_ratio', 0.15),
                    test_ratio=self.config.get('test_ratio', 0.15),
                    random_seed=self.config.get('random_seed', 42)
                )
                
                # 获取统计信息
                stats = self.data_processor.get_domain_statistics(domain)
                statistics[domain] = stats[domain]
                
                print(f"  ✅ {domain}: {stats[domain]}")
                
            except Exception as e:
                print(f"  ⚠️  {domain} 准备失败: {e}")
                statistics[domain] = {'error': str(e)}
        
        print("\n📊 数据集统计汇总:")
        for domain, stats in statistics.items():
            if 'error' not in stats:
                print(f"  {domain.upper()}: Train={stats.get('train', 0)}, "
                      f"Val={stats.get('val', 0)}, Test={stats.get('test', 0)}")
            else:
                print(f"  {domain.upper()}: ❌ {stats['error']}")
        
        return statistics
    
    def create_domain_topology(self, domain: str, task_description: str = None) -> MultiAgentTopologyManager:
        """
        为特定领域创建多智能体拓扑
        
        Args:
            domain: 领域名称 (mmlu/gsm8k/humaneval)
            task_description: 任务描述（可选，如果为None则自动生成）
        """
        print(f"🏗️ 为领域 {domain} 创建多智能体拓扑...")
        
        # 如果没有提供任务描述，从数据处理器获取
        if task_description is None:
            task_description = self.data_processor.get_task_description(domain)
        
        # 优先使用领域提示集中的智能体配置
        try:
            prompt_set = DomainPromptManager.get_manager(domain)
            agent_names = prompt_set.get_available_roles()
            print(f"✅ 使用预定义智能体配置: {agent_names}")
        except Exception as e:
            print(f"⚠️ 使用LLM生成智能体...")
            # 使用MetaAgent生成智能体描述
            available_tools = self.config.get('available_tools', [
                'knowledge_retrieval', 'logical_reasoning', 'calculation', 'analysis'
            ])
            
            try:
                agents_description, _ = Generate_Agent_Description(task_description, available_tools)
                agent_names = [agent['name'] for agent in agents_description]
                print(f"✅ 生成了 {len(agent_names)} 个智能体: {agent_names}")
            except Exception as e:
                print(f"⚠️ 智能体生成失败，使用默认配置: {e}")
                # 使用默认智能体配置
                agent_names = self._get_default_agents_for_domain(domain)
        
        # 创建多智能体拓扑管理器
        topology_manager = MultiAgentTopologyManager(
            task_domain=domain,
            language_model_name=self.config.get('llm_name', 'gpt-5-nano'),
            agent_role_names=agent_names,
            decision_strategy='final_decision',
            enable_spatial_optimization=True,
            enable_temporal_optimization=True,
            use_neural_temporal_graph=True,
            memory_bank_dimension=self.config.get('memory_dim', 128),
            temporal_encoding_dimension=self.config.get('time_dim', 32)
        )
        
        return topology_manager
    
    def _get_default_agents_for_domain(self, domain: str) -> List[str]:
        """获取领域的默认智能体配置"""
        domain_agents = {
            "mmlu": ["Knowledge Expert", "Subject Specialist", "Critical Analyzer", "Mathematician"],
            "gsm8k": ["Math Problem Solver", "Problem Analyzer", "Calculation Verifier", "Solution Critic"],
            "humaneval": ["Code Designer", "Code Writer", "Code Reviewer", "Test Engineer"],
            "STEM": ["Mathematical Reasoner", "Scientific Analyzer", "Problem Solver", "Verifier"],
            "Humanities": ["Historical Analyst", "Cultural Expert", "Critical Thinker", "Synthesizer"],
            "Social_Sciences": ["Social Analyst", "Economic Reasoner", "Policy Expert", "Evaluator"],
            "Other": ["General Expert", "Domain Specialist", "Analyst", "Reviewer"]
        }
        return domain_agents.get(domain, ["Expert Agent", "Analyst Agent", "Critic Agent"])
    
    async def train_domain_topology(self, 
                                  domain: str, 
                                  topology_manager: MultiAgentTopologyManager,
                                  num_epochs: int = 100) -> Dict[str, Any]:
        """训练特定领域的拓扑结构"""
        print(f"🎯 开始训练领域 {domain} 的拓扑结构...")
        
        # 准备训练数据
        training_batches = self.data_processor.prepare_multi_agent_training_data(
            domain=domain,
            split="train",
            batch_size=self.config.get('batch_size', 16)
        )
        
        validation_batches = self.data_processor.prepare_multi_agent_training_data(
            domain=domain,
            split="val",
            batch_size=self.config.get('batch_size', 16)
        )
        
        # 设置优化器
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            model_params = (
                list(topology_manager.neural_temporal_graph.parameters()) +
                list(topology_manager.decision_decoder.parameters())
            )
        else:
            model_params = list(topology_manager.compatibility_graph_network.parameters())
        
        optimizer = optim.Adam(model_params, lr=self.config.get('learning_rate', 0.001))
        
        # 训练循环
        training_results = {
            'domain': domain,
            'epoch_losses': [],
            'epoch_accuracies': [],
            'validation_accuracies': [],
            'best_accuracy': 0.0,
            'best_epoch': 0
        }
        
        for epoch in range(num_epochs):
            print(f"📚 Epoch {epoch + 1}/{num_epochs} for domain {domain}")
            
            # 训练阶段
            epoch_loss, epoch_accuracy = await self._train_epoch(
                topology_manager, training_batches, optimizer, domain
            )
            
            # 验证阶段
            val_accuracy = await self._validate_epoch(
                topology_manager, validation_batches, domain
            )
            
            # 记录结果
            training_results['epoch_losses'].append(epoch_loss)
            training_results['epoch_accuracies'].append(epoch_accuracy)
            training_results['validation_accuracies'].append(val_accuracy)
            
            # 保存最佳模型
            if val_accuracy > training_results['best_accuracy']:
                training_results['best_accuracy'] = val_accuracy
                training_results['best_epoch'] = epoch
                self._save_best_model(topology_manager, domain, epoch, val_accuracy)
            
            print(f"  📊 Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # 早停检查
            if self._should_early_stop(training_results, patience=10):
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        print(f"✅ 完成领域 {domain} 的训练，最佳验证准确率: {training_results['best_accuracy']:.4f}")
        return training_results
    
    async def _train_epoch(self, 
                          topology_manager: MultiAgentTopologyManager,
                          training_batches: List[Dict],
                          optimizer: torch.optim.Optimizer,
                          domain: str) -> tuple[float, float]:
        """训练一个epoch"""
        
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            topology_manager.neural_temporal_graph.train()
        topology_manager.compatibility_graph_network.train()
        
        total_loss = 0.0
        total_correct = 0
        total_questions = 0
        
        for batch_idx, batch in enumerate(training_batches):
            batch_loss = 0.0
            batch_correct = 0
            
            for question_data in batch['questions']:
                # 格式化问题
                formatted_question = self.mmlu_processor.format_question_for_agents(question_data)
                
                try:
                    # 执行多智能体推理
                    agent_responses, log_probs = await topology_manager.execute_multi_agent_reasoning(
                        task_input=formatted_question,
                        num_interaction_rounds=self.config.get('num_rounds', 3)
                    )
                    
                    # 评估响应
                    if agent_responses:
                        evaluation = self.mmlu_processor.evaluate_agent_response(
                            str(agent_responses[0]), 
                            question_data.get('answer', '')
                        )
                        
                        # 计算奖励
                        reward = 1.0 if evaluation['is_correct'] else 0.0
                        
                        # 策略梯度损失
                        loss = -log_probs * reward
                        batch_loss += loss
                        
                        if evaluation['is_correct']:
                            batch_correct += 1
                    
                    total_questions += 1
                    
                except Exception as e:
                    print(f"⚠️ 训练错误: {e}")
                    continue
            
            # 反向传播
            if batch_loss != 0:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in optimizer.param_groups[0]['params']], 
                    max_norm=1.0
                )
                optimizer.step()
                
                total_loss += batch_loss.item()
            
            total_correct += batch_correct
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx + 1}/{len(training_batches)}, "
                      f"Loss: {batch_loss:.4f}, Acc: {batch_correct}/{len(batch['questions'])}")
        
        avg_loss = total_loss / len(training_batches) if training_batches else 0.0
        avg_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    async def _validate_epoch(self, 
                            topology_manager: MultiAgentTopologyManager,
                            validation_batches: List[Dict],
                            domain: str) -> float:
        """验证一个epoch"""
        
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            topology_manager.neural_temporal_graph.eval()
        topology_manager.compatibility_graph_network.eval()
        
        total_correct = 0
        total_questions = 0
        
        with torch.no_grad():
            for batch in validation_batches:
                for question_data in batch['questions']:
                    formatted_question = self.mmlu_processor.format_question_for_agents(question_data)
                    
                    try:
                        agent_responses, _ = await topology_manager.execute_multi_agent_reasoning(
                            task_input=formatted_question,
                            num_interaction_rounds=self.config.get('num_rounds', 3)
                        )
                        
                        if agent_responses:
                            evaluation = self.mmlu_processor.evaluate_agent_response(
                                str(agent_responses[0]), 
                                question_data.get('answer', '')
                            )
                            
                            if evaluation['is_correct']:
                                total_correct += 1
                        
                        total_questions += 1
                        
                    except Exception as e:
                        print(f"⚠️ 验证错误: {e}")
                        continue
        
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        return accuracy
    
    def _should_early_stop(self, training_results: Dict, patience: int = 10) -> bool:
        """检查是否应该早停"""
        if len(training_results['validation_accuracies']) < patience:
            return False
        
        recent_accuracies = training_results['validation_accuracies'][-patience:]
        return all(acc <= training_results['best_accuracy'] for acc in recent_accuracies)
    
    def _save_best_model(self, 
                        topology_manager: MultiAgentTopologyManager,
                        domain: str,
                        epoch: int,
                        accuracy: float):
        """保存最佳模型"""
        model_dir = self.output_dir / "best_models" / domain
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        model_state = {
            'epoch': epoch,
            'accuracy': accuracy,
            'domain': domain,
            'config': self.config
        }
        
        if topology_manager.use_neural_temporal_graph and topology_manager.neural_temporal_graph:
            model_state['neural_temporal_graph'] = topology_manager.neural_temporal_graph.state_dict()
        
        model_state['compatibility_graph_network'] = topology_manager.compatibility_graph_network.state_dict()
        model_state['decision_decoder'] = topology_manager.decision_decoder.state_dict()
        
        model_path = model_dir / f"best_model_epoch_{epoch}.pth"
        torch.save(model_state, model_path)
        
        # 保存拓扑配置
        topology_config = {
            'agent_role_names': topology_manager.agent_role_names,
            'task_domain': topology_manager.task_domain,
            'num_agents': topology_manager.num_agents
        }
        
        config_path = model_dir / "topology_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(topology_config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 保存最佳模型到 {model_path}")
    
    async def train_all_domains(self) -> Dict[str, Dict[str, Any]]:
        """训练所有领域"""
        print("🚀 开始训练所有领域的神经多智能体系统...")
        
        # 准备数据
        statistics = self.prepare_training_data()
        
        # 为每个领域训练拓扑
        all_results = {}
        
        for domain in statistics.keys():
            print(f"\n{'='*60}")
            print(f"🎯 开始训练领域: {domain}")
            print(f"{'='*60}")
            
            # 创建任务描述
            task_description = f"Solve {domain} questions from MMLU dataset with multiple choice answers"
            
            # 创建拓扑管理器
            topology_manager = self.create_domain_topology(domain, task_description)
            self.domain_topologies[domain] = topology_manager
            
            # 训练
            training_results = await self.train_domain_topology(
                domain, 
                topology_manager, 
                num_epochs=self.config.get('num_epochs', 50)
            )
            
            all_results[domain] = training_results
        
        # 保存训练历史
        self._save_training_results(all_results)
        
        return all_results
    
    def _save_training_results(self, results: Dict[str, Dict[str, Any]]):
        """保存训练结果"""
        results_path = self.output_dir / "training_results.json"
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for domain, result in results.items():
            serializable_results[domain] = {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in result.items()
            }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📊 训练结果已保存到 {results_path}")
    
    async def evaluate_all_domains(self) -> Dict[str, float]:
        """评估所有领域"""
        print("🧪 开始评估所有领域...")
        
        evaluation_results = {}
        
        for domain, topology_manager in self.domain_topologies.items():
            print(f"📝 评估领域: {domain}")
            
            # 准备测试数据
            test_batches = self.mmlu_processor.prepare_multi_agent_training_data(
                domain=domain,
                split="test",
                batch_size=self.config.get('batch_size', 16)
            )
            
            # 评估
            test_accuracy = await self._validate_epoch(topology_manager, test_batches, domain)
            evaluation_results[domain] = test_accuracy
            
            print(f"  📊 {domain} 测试准确率: {test_accuracy:.4f}")
        
        return evaluation_results


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Neural Multi-Agent System Training")
    
    parser.add_argument("--mmlu_data_path", type=str, required=True,
                       help="Path to MMLU dataset")
    parser.add_argument("--output_dir", type=str, default="./neural_mas_outputs",
                       help="Output directory for results")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--num_rounds", type=int, default=3,
                       help="Number of interaction rounds")
    
    # 模型参数
    parser.add_argument("--memory_dim", type=int, default=128,
                       help="Memory dimension for TGN")
    parser.add_argument("--time_dim", type=int, default=32,
                       help="Time encoding dimension")
    parser.add_argument("--llm_name", type=str, default="gpt-5-nano",
                       help="Language model name (default: gpt-5-nano)")
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    # 构建配置
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_rounds': args.num_rounds,
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
        'llm_name': args.llm_name,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'available_tools': [
            'knowledge_retrieval', 'logical_reasoning', 'calculation', 
            'analysis', 'critical_thinking', 'synthesis'
        ]
    }
    
    # 如果提供了配置文件，加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    print("🌟 神经多智能体系统训练开始")
    print(f"📁 MMLU数据路径: {args.mmlu_data_path}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"⚙️ 配置: {json.dumps(config, indent=2)}")
    
    # 创建训练器
    trainer = NeuralMASTrainer(
        config=config,
        mmlu_data_path=args.mmlu_data_path,
        output_dir=args.output_dir
    )
    
    try:
        # 训练所有领域
        training_results = await trainer.train_all_domains()
        
        # 评估所有领域
        evaluation_results = await trainer.evaluate_all_domains()
        
        # 打印最终结果
        print("\n" + "="*80)
        print("🎉 训练完成！最终结果:")
        print("="*80)
        
        for domain in training_results.keys():
            train_acc = training_results[domain]['best_accuracy']
            test_acc = evaluation_results.get(domain, 0.0)
            print(f"📊 {domain:15} | 最佳验证准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")
        
        # 计算平均准确率
        avg_test_acc = np.mean(list(evaluation_results.values()))
        print(f"\n🏆 平均测试准确率: {avg_test_acc:.4f}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
