"""
FSM Multi-Agent System Training Script
FSM多智能体系统训练脚本

完整集成MetaAgent的FSM生成能力和TGN学习能力
支持：
1. 自动生成智能体角色和状态描述
2. 随机采样状态转移拓扑和Listening通信拓扑
3. 使用TGN学习最优路径
4. 支持MMLU等数据集训练

🎯 损失函数设计（多任务学习）：
- 组合损失 = α * 策略梯度损失 + β * MSE重构损失
- α (策略梯度权重): 1.0 - 直接优化任务准确率
- β (MSE重构权重): 0.1 - 帮助TGN学习稳定的节点表示
- 策略梯度确保任务性能，MSE损失提供训练稳定性
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
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random

# 添加路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from neural_fsm_mas.fsm_integration.fsm_mas_generator import FSMMultiAgentSystemGenerator
from neural_fsm_mas.training_data.mmlu_data_processor import MMLUDataProcessor
from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph
from baseclass.FSM_Gen import Generate_Agent_Description, Generate_FSM


class FSMMultiAgentSystemTrainer:
    """
    FSM多智能体系统训练器
    
    核心功能：
    1. 使用MetaAgent生成智能体和FSM
    2. 随机采样拓扑图
    3. 使用TGN学习最优路径
    4. 支持多数据集训练
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 data_path: str = None,
                 output_dir: str = "./fsm_mas_outputs"):
        
        self.config = config
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化FSM-MAS生成器
        self.fsm_mas_generator = FSMMultiAgentSystemGenerator(
            use_neural_learning=config.get('use_neural_learning', True),
            memory_dimension=config.get('memory_dim', 128),
            temporal_dimension=config.get('time_dim', 32),
            random_seed=config.get('random_seed', 42)
        )
        
        # 数据处理器（如果提供了数据路径）
        self.data_processor = None
        if data_path and os.path.exists(data_path):
            self.data_processor = MMLUDataProcessor(data_path)
        
        # 训练状态
        self.training_history = []
        self.generated_systems = {}
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def generate_fsm_mas_for_task(self, task_description: str) -> Dict[str, Any]:
        """
        为特定任务生成FSM多智能体系统
        
        Args:
            task_description: 任务描述
            
        Returns:
            生成的FSM-MAS系统配置
        """
        print(f"🎯 为任务生成FSM-MAS系统: {task_description}")
        
        # 获取可用工具
        available_tools = self.config.get('available_tools', [
            'code_interpreter', 'web_search', 'calculator', 
            'knowledge_retrieval', 'logical_reasoning', 'analysis'
        ])
        
        # 生成完整的FSM-MAS系统
        fsm_mas_config = self.fsm_mas_generator.generate_complete_fsm_mas(
            task_description=task_description,
            available_tools=available_tools
        )
        
        # 保存生成的系统
        task_hash = hash(task_description) % 10000
        system_path = self.output_dir / f"fsm_mas_system_{task_hash}.json"
        self.fsm_mas_generator.save_fsm_mas_system(str(system_path))
        
        return fsm_mas_config
    
    def train_fsm_mas_with_tgn(self, 
                              fsm_mas_config: Dict[str, Any],
                              training_episodes: int = None,
                              domain: str = None) -> Dict[str, Any]:
        """
        使用TGN训练FSM-MAS系统，包含完整的训练和验证过程
        
        Args:
            fsm_mas_config: FSM-MAS系统配置
            training_episodes: 训练轮数
            domain: 训练领域（用于MMLU数据集）
            
        Returns:
            训练结果，包含准确率、损失、交互日志等
        """
        if training_episodes is None:
            training_episodes = self.config.get('training_episodes', 100)
        
        print(f"🚀 开始TGN训练，训练轮数: {training_episodes}")
        if domain:
            print(f"📚 训练领域: {domain}")
        
        # 准备训练和验证数据
        training_data = []
        validation_data = []
        
        if domain and self.data_processor:
            training_batches = self.data_processor.prepare_multi_agent_training_data(
                domain=domain, 
                split="train", 
                batch_size=self.config.get('batch_size', 16)
            )
            validation_batches = self.data_processor.prepare_multi_agent_training_data(
                domain=domain, 
                split="validation", 
                batch_size=self.config.get('batch_size', 16)
            )
            training_data = training_batches
            validation_data = validation_batches
            print(f"📊 训练批次: {len(training_data)}, 验证批次: {len(validation_data)}")
        
        # 组合损失权重配置
        policy_gradient_weight = self.config.get('policy_gradient_weight', 1.0)  # α
        reconstruction_weight = self.config.get('reconstruction_weight', 0.1)    # β
        
        print(f"📊 组合损失配置:")
        print(f"  - 策略梯度权重 (α): {policy_gradient_weight}")
        print(f"  - MSE重构权重 (β): {reconstruction_weight}")
        
        # 初始化训练记录
        training_metrics = {
            'episode_losses': [],
            'policy_gradient_losses': [],  # 新增：记录策略梯度损失
            'reconstruction_losses': [],   # 新增：记录MSE重构损失
            'combined_losses': [],         # 新增：记录组合损失
            'episode_accuracies': [],
            'validation_accuracies': [],
            'agent_interaction_logs': [],
            'policy_gradient_updates': [],
            'best_validation_accuracy': 0.0,
            'best_episode': 0,
            'loss_weights': {              # 新增：记录损失权重
                'policy_gradient_weight': policy_gradient_weight,
                'reconstruction_weight': reconstruction_weight
            }
        }
        
        # 创建可执行系统用于评估
        executable_system = None
        if all([fsm_mas_config.get('agents'), fsm_mas_config.get('fsm')]):
            try:
                executable_system = self.fsm_mas_generator.create_executable_system()
                print("✅ 创建可执行系统成功")
            except Exception as e:
                print(f"⚠️ 创建可执行系统失败: {e}")
        
        # 🔥 关键：初始化优化器（优化TGN网络参数）
        if self.fsm_mas_generator.use_neural_learning:
            optimizer_params = []
            
            # 添加状态转移TGN的参数
            if self.fsm_mas_generator.neural_state_learner:
                optimizer_params.extend(list(self.fsm_mas_generator.neural_state_learner.parameters()))
            
            # 添加智能体通信TGN的参数
            if self.fsm_mas_generator.neural_communication_learner:
                optimizer_params.extend(list(self.fsm_mas_generator.neural_communication_learner.parameters()))
            
            if optimizer_params:
                optimizer = torch.optim.Adam(optimizer_params, lr=self.config.get('learning_rate', 0.001))
                print(f"✅ 优化器初始化完成，优化 {len(optimizer_params)} 个参数组")
            else:
                optimizer = None
                print("⚠️ 未找到TGN参数，将不执行梯度更新")
        else:
            optimizer = None
        
        # 开始训练循环
        for episode in range(training_episodes):
            episode_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"📈 训练轮次 {episode + 1}/{training_episodes}")
            print(f"{'='*60}")
            
            # 训练阶段
            episode_training_accuracy = 0.0
            episode_loss = 0.0
            episode_interactions = []
            
            if training_data and executable_system:
                # 使用真实MMLU数据进行训练
                episode_training_accuracy, episode_interactions = self._train_episode_with_mmlu_data(
                    executable_system, training_data, episode
                )
            else:
                # 使用模拟数据进行训练
                episode_training_accuracy, episode_interactions = self._train_episode_with_simulation(
                    episode
                )
            
            # TGN网络优化 - 计算MSE重构损失
            learning_results = self.fsm_mas_generator.learn_optimal_topologies(
                training_episodes=1,  # 每个episode只训练一次
                learning_rate=self.config.get('learning_rate', 0.001)
            )
            
            # 获取MSE重构损失
            state_reconstruction_loss = learning_results.get('final_state_loss', 0.0)
            comm_reconstruction_loss = learning_results.get('final_communication_loss', 0.0)
            reconstruction_loss = (state_reconstruction_loss + comm_reconstruction_loss) / 2.0
            
            # ===== 策略梯度损失计算（参考GDesigner） =====
            # GDesigner中：loss = -log_prob * reward
            # log_prob = sum(log(edge_prob)) 来自拓扑连接的采样概率
            
            # 计算用于显示的简化策略梯度损失
            policy_gradient_loss = 1.0 - episode_training_accuracy
            
            # 🔥 关键：反向传播和参数更新
            if optimizer is not None:
                optimizer.zero_grad()
                
                # 重新前向传播计算损失（带梯度）
                state_features = self.fsm_mas_generator._prepare_state_features()
                agent_features = self.fsm_mas_generator._prepare_agent_features()
                
                # ===== 1. MSE重构损失部分 =====
                state_edge_index = self.fsm_mas_generator._build_state_edge_index()
                state_timestamps = torch.tensor([episode] * len(self.fsm_mas_generator.generated_fsm['states']), dtype=torch.float)
                evolved_states = self.fsm_mas_generator.neural_state_learner(
                    state_features, state_edge_index, timestamps=state_timestamps
                )
                state_loss_with_grad = torch.nn.functional.mse_loss(evolved_states, state_features)
                
                agent_edge_index = self.fsm_mas_generator._build_communication_edge_index()
                agent_timestamps = torch.tensor([episode] * len(self.fsm_mas_generator.generated_agents), dtype=torch.float)
                evolved_agents = self.fsm_mas_generator.neural_communication_learner(
                    agent_features, agent_edge_index, timestamps=agent_timestamps
                )
                comm_loss_with_grad = torch.nn.functional.mse_loss(evolved_agents, agent_features)
                
                reconstruction_loss_with_grad = (state_loss_with_grad + comm_loss_with_grad) / 2.0
                
                # ===== 2. 策略梯度损失部分（参考GDesigner实现） =====
                # GDesigner: edge_prob = sigmoid(logit), log_prob = log(edge_prob)
                # 我们需要：1) 计算连接概率 2) 计算log_prob 3) 用-log_prob * reward作为损失
                
                # 计算状态转移的连接概率
                # 使用evolved_states之间的相似度作为logit
                state_logits = torch.matmul(evolved_states, evolved_states.t())  # [n_states, n_states]
                state_edge_probs = torch.sigmoid(state_logits)  # 连接概率
                
                # 计算状态转移的log_prob（对所有可能的连接求和）
                # 注意：实际采样的连接会增加log(prob)，未采样的增加log(1-prob)
                # 这里简化为所有可能连接的概率之和
                state_log_probs = torch.log(state_edge_probs + 1e-10).sum() / (state_edge_probs.numel())
                
                # 计算智能体通信的连接概率
                agent_logits = torch.matmul(evolved_agents, evolved_agents.t())  # [n_agents, n_agents]
                agent_edge_probs = torch.sigmoid(agent_logits)
                agent_log_probs = torch.log(agent_edge_probs + 1e-10).sum() / (agent_edge_probs.numel())
                
                # 总log_prob
                total_log_prob = (state_log_probs + agent_log_probs) / 2.0
                
                # 策略梯度损失：-log_prob * reward（GDesigner的标准形式）
                reward = episode_training_accuracy  # 奖励 = MMLU准确率
                policy_loss_with_grad = -total_log_prob * reward
                
                # ===== 3. 组合损失 =====
                total_loss_with_grad = (
                    policy_gradient_weight * policy_loss_with_grad + 
                    reconstruction_weight * reconstruction_loss_with_grad
                )
                
                # 反向传播
                total_loss_with_grad.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                # 记录梯度信息
                total_grad_norm = sum(p.grad.norm().item() for p in optimizer_params if p.grad is not None)
                
                # 更新显示用的combined_loss
                combined_loss = total_loss_with_grad.item()
            else:
                total_grad_norm = 0.0
                combined_loss = (
                    policy_gradient_weight * policy_gradient_loss + 
                    reconstruction_weight * reconstruction_loss
                )
            
            episode_loss = combined_loss
            
            # 验证阶段
            episode_validation_accuracy = 0.0
            if validation_data and executable_system:
                episode_validation_accuracy = self._validate_episode_with_mmlu_data(
                    executable_system, validation_data, episode
                )
            else:
                # 模拟验证准确率
                episode_validation_accuracy = max(0.0, episode_training_accuracy + np.random.normal(0, 0.05))
                episode_validation_accuracy = min(1.0, episode_validation_accuracy)
            
            # 记录策略梯度更新（实际的梯度信息）
            policy_update = {
                'episode': episode + 1,
                'gradient_norm': total_grad_norm,  # 实际的梯度范数
                'learning_rate': self.config.get('learning_rate', 0.001),
                'policy_gradient_loss': policy_gradient_loss,
                'reconstruction_loss': reconstruction_loss,
                'combined_loss': combined_loss,
                'optimizer_active': optimizer is not None
            }
            
            # 更新最佳模型
            if episode_validation_accuracy > training_metrics['best_validation_accuracy']:
                training_metrics['best_validation_accuracy'] = episode_validation_accuracy
                training_metrics['best_episode'] = episode + 1
                print(f"🌟 新的最佳验证准确率: {episode_validation_accuracy:.4f}")
            
            # 记录训练指标
            training_metrics['episode_losses'].append(episode_loss)
            training_metrics['policy_gradient_losses'].append(policy_gradient_loss)
            training_metrics['reconstruction_losses'].append(reconstruction_loss)
            training_metrics['combined_losses'].append(combined_loss)
            training_metrics['episode_accuracies'].append(episode_training_accuracy)
            training_metrics['validation_accuracies'].append(episode_validation_accuracy)
            training_metrics['agent_interaction_logs'].extend(episode_interactions)
            training_metrics['policy_gradient_updates'].append(policy_update)
            
            # 打印训练进度
            episode_time = time.time() - episode_start_time
            print(f"📊 轮次 {episode + 1} 结果:")
            print(f"   训练准确率: {episode_training_accuracy:.4f}")
            print(f"   验证准确率: {episode_validation_accuracy:.4f}")
            print(f"   组合损失: {combined_loss:.4f}")
            print(f"     └─ 策略梯度损失: {policy_gradient_loss:.4f} (权重={policy_gradient_weight})")
            print(f"     └─ MSE重构损失: {reconstruction_loss:.4f} (权重={reconstruction_weight})")
            print(f"   智能体交互: {len(episode_interactions)} 次")
            print(f"   用时: {episode_time:.2f}秒")
            
            # 每10个episode保存一次中间结果
            if (episode + 1) % 10 == 0:
                self._save_intermediate_results(training_metrics, episode + 1, domain)
        
        # 合并最终结果
        final_results = {
            **learning_results,
            'training_metrics': training_metrics,
            'final_training_accuracy': training_metrics['episode_accuracies'][-1] if training_metrics['episode_accuracies'] else 0.0,
            'final_validation_accuracy': training_metrics['validation_accuracies'][-1] if training_metrics['validation_accuracies'] else 0.0,
            'best_validation_accuracy': training_metrics['best_validation_accuracy'],
            'best_episode': training_metrics['best_episode'],
            'total_interactions': len(training_metrics['agent_interaction_logs']),
            'average_training_accuracy': np.mean(training_metrics['episode_accuracies']) if training_metrics['episode_accuracies'] else 0.0,
            'average_validation_accuracy': np.mean(training_metrics['validation_accuracies']) if training_metrics['validation_accuracies'] else 0.0,
            # 新增：组合损失相关统计
            'final_combined_loss': training_metrics['combined_losses'][-1] if training_metrics['combined_losses'] else 0.0,
            'final_policy_gradient_loss': training_metrics['policy_gradient_losses'][-1] if training_metrics['policy_gradient_losses'] else 0.0,
            'final_reconstruction_loss': training_metrics['reconstruction_losses'][-1] if training_metrics['reconstruction_losses'] else 0.0,
            'average_combined_loss': np.mean(training_metrics['combined_losses']) if training_metrics['combined_losses'] else 0.0,
            'average_policy_gradient_loss': np.mean(training_metrics['policy_gradient_losses']) if training_metrics['policy_gradient_losses'] else 0.0,
            'average_reconstruction_loss': np.mean(training_metrics['reconstruction_losses']) if training_metrics['reconstruction_losses'] else 0.0
        }
        
        # 记录训练历史
        training_record = {
            'timestamp': time.time(),
            'task_description': fsm_mas_config.get('task_description', 'Unknown'),
            'domain': domain,
            'training_episodes': training_episodes,
            'final_state_loss': learning_results.get('final_state_loss', 0.0),
            'final_communication_loss': learning_results.get('final_communication_loss', 0.0),
            'final_training_accuracy': final_results['final_training_accuracy'],
            'final_validation_accuracy': final_results['final_validation_accuracy'],
            'best_validation_accuracy': final_results['best_validation_accuracy'],
            'system_metadata': fsm_mas_config.get('system_metadata', {})
        }
        
        self.training_history.append(training_record)
        
        print(f"\n🎉 训练完成!")
        print(f"\n📊 准确率统计:")
        print(f"  - 最终训练准确率: {final_results['final_training_accuracy']:.4f}")
        print(f"  - 最终验证准确率: {final_results['final_validation_accuracy']:.4f}")
        print(f"  - 平均训练准确率: {final_results['average_training_accuracy']:.4f}")
        print(f"  - 平均验证准确率: {final_results['average_validation_accuracy']:.4f}")
        print(f"  🌟 最佳验证准确率: {final_results['best_validation_accuracy']:.4f} (轮次 {final_results['best_episode']})")
        
        print(f"\n📉 损失函数统计:")
        print(f"  - 最终组合损失: {final_results['final_combined_loss']:.4f}")
        print(f"    └─ 策略梯度损失: {final_results['final_policy_gradient_loss']:.4f}")
        print(f"    └─ MSE重构损失: {final_results['final_reconstruction_loss']:.4f}")
        print(f"  - 平均组合损失: {final_results['average_combined_loss']:.4f}")
        
        print(f"\n🤝 智能体交互:")
        print(f"  - 总交互次数: {final_results['total_interactions']}")
        
        return final_results
    
    def train_on_mmlu_domains(self) -> Dict[str, Any]:
        """
        在MMLU数据集的不同领域上训练FSM-MAS系统
        """
        if not self.data_processor:
            raise ValueError("MMLU data processor not initialized. Please provide data_path.")
        
        print("📚 开始在MMLU领域上训练FSM-MAS系统...")
        
        # 准备MMLU数据
        self.data_processor.create_domain_splits(
            train_ratio=self.config.get('train_ratio', 0.7),
            val_ratio=self.config.get('val_ratio', 0.15),
            test_ratio=self.config.get('test_ratio', 0.15)
        )
        
        domain_statistics = self.data_processor.get_domain_statistics()
        all_results = {}
        
        for domain in domain_statistics.keys():
            print(f"\n{'='*60}")
            print(f"🎯 训练领域: {domain}")
            print(f"{'='*60}")
            
            # 为每个领域创建特定的任务描述
            domain_task_descriptions = {
                "STEM": "Solve STEM questions including mathematics, physics, chemistry, biology, and computer science problems from MMLU dataset",
                "Humanities": "Answer humanities questions covering history, philosophy, literature, and cultural studies from MMLU dataset", 
                "Social_Sciences": "Solve social science questions including psychology, sociology, economics, and political science from MMLU dataset",
                "Other": "Answer general knowledge questions covering business, medicine, and miscellaneous topics from MMLU dataset"
            }
            
            task_description = domain_task_descriptions.get(
                domain, 
                f"Solve {domain} questions from MMLU dataset"
            )
            
            # 生成FSM-MAS系统
            fsm_mas_config = self.generate_fsm_mas_for_task(task_description)
            
            # 训练TGN
            learning_results = self.train_fsm_mas_with_tgn(
                fsm_mas_config,
                training_episodes=self.config.get('training_episodes', 50),
                domain=domain
            )
            
            # 存储结果
            all_results[domain] = {
                'fsm_mas_config': fsm_mas_config,
                'learning_results': learning_results,
                'domain_statistics': domain_statistics[domain]
            }
            
            print(f"✅ 领域 {domain} 训练完成")
            print(f"📊 最终状态损失: {learning_results['final_state_loss']:.4f}")
            print(f"📊 最终通信损失: {learning_results['final_communication_loss']:.4f}")
        
        # 保存所有结果
        self._save_training_results(all_results)
        
        return all_results
    
    def train_single_task(self, task_description: str) -> Dict[str, Any]:
        """
        训练单个任务的FSM-MAS系统
        
        Args:
            task_description: 任务描述
            
        Returns:
            训练结果
        """
        print(f"🎯 训练单个任务: {task_description}")
        
        # 生成FSM-MAS系统
        fsm_mas_config = self.generate_fsm_mas_for_task(task_description)
        
        # 训练TGN
        learning_results = self.train_fsm_mas_with_tgn(fsm_mas_config)
        
        # 构建完整结果
        complete_results = {
            'task_description': task_description,
            'fsm_mas_config': fsm_mas_config,
            'learning_results': learning_results
        }
        
        # 保存结果
        task_hash = hash(task_description) % 10000
        results_path = self.output_dir / f"single_task_results_{task_hash}.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = self._make_json_serializable(complete_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存到 {results_path}")
        
        return complete_results
    
    def demonstrate_fsm_mas_capabilities(self):
        """
        演示FSM-MAS系统的核心能力
        """
        print("🎪 演示FSM-MAS系统核心能力...")
        
        # 示例任务列表
        demo_tasks = [
            "Solve mathematical word problems step by step",
            "Analyze scientific research papers and extract key findings", 
            "Generate and debug Python code for data analysis",
            "Answer multiple choice questions across various academic domains"
        ]
        
        demo_results = {}
        
        for i, task in enumerate(demo_tasks):
            print(f"\n🎯 演示任务 {i+1}: {task}")
            
            try:
                # 生成FSM-MAS系统（不进行完整训练，只演示生成过程）
                fsm_mas_config = self.generate_fsm_mas_for_task(task)
                
                # 简短的TGN学习演示
                learning_results = self.train_fsm_mas_with_tgn(
                    fsm_mas_config, 
                    training_episodes=10  # 演示用较少轮数
                )
                
                demo_results[f"task_{i+1}"] = {
                    'task_description': task,
                    'num_agents': fsm_mas_config['system_metadata']['num_agents'],
                    'num_states': fsm_mas_config['system_metadata']['num_states'],
                    'final_losses': {
                        'state_loss': learning_results['final_state_loss'],
                        'communication_loss': learning_results['final_communication_loss']
                    }
                }
                
                print(f"✅ 任务 {i+1} 演示完成")
                
            except Exception as e:
                print(f"❌ 任务 {i+1} 演示失败: {e}")
                demo_results[f"task_{i+1}"] = {'error': str(e)}
        
        # 保存演示结果
        demo_path = self.output_dir / "fsm_mas_demo_results.json"
        with open(demo_path, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False)
        
        print(f"🎉 演示完成！结果保存到 {demo_path}")
        
        return demo_results
    
    def _save_training_results(self, results: Dict[str, Any]):
        """保存训练结果"""
        results_path = self.output_dir / "fsm_mas_training_results.json"
        
        # 转换为可序列化格式
        serializable_results = self._make_json_serializable(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        print(f"📊 训练结果已保存到 {results_path}")
        print(f"📈 训练历史已保存到 {history_path}")
    
    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.training_history:
            return {"message": "No training completed yet"}
        
        summary = {
            "total_training_sessions": len(self.training_history),
            "average_state_loss": np.mean([record['final_state_loss'] for record in self.training_history]),
            "average_communication_loss": np.mean([record['final_communication_loss'] for record in self.training_history]),
            "total_training_episodes": sum([record['training_episodes'] for record in self.training_history]),
            "trained_tasks": [record['task_description'] for record in self.training_history]
        }
        
        return summary
    
    def _train_episode_with_mmlu_data(self, 
                                     executable_system, 
                                     training_data: List[Dict], 
                                     episode: int) -> Tuple[float, List[Dict]]:
        """
        使用MMLU数据训练一个episode
        
        Args:
            executable_system: 可执行的多智能体系统
            training_data: 训练数据批次
            episode: 当前episode编号
            
        Returns:
            Tuple[训练准确率, 智能体交互日志]
        """
        correct_answers = 0
        total_questions = 0
        interaction_logs = []
        
        # 随机选择一些批次进行训练（避免过长的训练时间）
        selected_batches = random.sample(training_data, min(3, len(training_data)))
        
        for batch_idx, batch in enumerate(selected_batches):
            batch_questions = batch['questions']
            
            # 随机选择批次中的一些问题
            selected_questions = random.sample(batch_questions, min(5, len(batch_questions)))
            
            for q_idx, question_data in enumerate(selected_questions):
                # 格式化问题
                formatted_question = self.data_processor.format_question_for_agents(question_data)
                
                try:
                    # 执行多智能体系统
                    start_time = time.time()
                    result, cost = executable_system.start(
                        formatted_question['task'], 
                        max_transitions=5
                    )
                    execution_time = time.time() - start_time
                    
                    # 评估结果
                    evaluation = self.data_processor.evaluate_agent_response(
                        result, 
                        formatted_question['correct_answer']
                    )
                    
                    if evaluation['is_correct']:
                        correct_answers += 1
                    total_questions += 1
                    
                    # 记录交互日志
                    interaction_log = {
                        'episode': episode + 1,
                        'batch_idx': batch_idx,
                        'question_idx': q_idx,
                        'subject': question_data['subject'],
                        'question': formatted_question['original_question'],
                        'correct_answer': formatted_question['correct_answer'],
                        'agent_response': result,
                        'predicted_answer': evaluation['predicted_answer'],
                        'is_correct': evaluation['is_correct'],
                        'confidence_score': evaluation['confidence_score'],
                        'execution_time': execution_time,
                        'cost': cost,
                        'timestamp': time.time()
                    }
                    
                    interaction_logs.append(interaction_log)
                    
                    # 打印详细的交互信息
                    if q_idx < 2:  # 只打印前两个问题的详细信息
                        print(f"   📝 问题 {q_idx + 1}: {formatted_question['original_question'][:100]}...")
                        print(f"      正确答案: {formatted_question['correct_answer']}")
                        print(f"      智能体答案: {evaluation['predicted_answer']}")
                        print(f"      结果: {'✅ 正确' if evaluation['is_correct'] else '❌ 错误'}")
                        print(f"      置信度: {evaluation['confidence_score']:.2f}")
                        print(f"      执行时间: {execution_time:.2f}秒")
                    
                except Exception as e:
                    print(f"   ❌ 问题 {q_idx + 1} 执行失败: {e}")
                    total_questions += 1
                    
                    # 记录失败的交互
                    interaction_logs.append({
                        'episode': episode + 1,
                        'batch_idx': batch_idx,
                        'question_idx': q_idx,
                        'subject': question_data['subject'],
                        'question': formatted_question['original_question'],
                        'error': str(e),
                        'is_correct': False,
                        'timestamp': time.time()
                    })
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        print(f"   📊 Episode训练结果: {correct_answers}/{total_questions} = {accuracy:.4f}")
        
        return accuracy, interaction_logs
    
    def _train_episode_with_simulation(self, episode: int) -> Tuple[float, List[Dict]]:
        """
        使用模拟数据训练一个episode
        
        Args:
            episode: 当前episode编号
            
        Returns:
            Tuple[模拟训练准确率, 模拟交互日志]
        """
        # 模拟训练准确率（随着episode增加而提高）
        base_accuracy = 0.3
        improvement = min(0.4, episode * 0.02)  # 每个episode提高2%，最多提高40%
        noise = np.random.normal(0, 0.05)  # 添加噪声
        accuracy = max(0.0, min(1.0, base_accuracy + improvement + noise))
        
        # 模拟交互日志
        num_interactions = random.randint(8, 15)
        interaction_logs = []
        
        for i in range(num_interactions):
            interaction_log = {
                'episode': episode + 1,
                'interaction_idx': i,
                'simulated': True,
                'agent_id': f"agent_{random.randint(0, 3)}",
                'state_id': f"state_{random.randint(0, 4)}",
                'action': random.choice(['analyze', 'reason', 'calculate', 'verify', 'synthesize']),
                'input_tokens': random.randint(50, 200),
                'output_tokens': random.randint(30, 150),
                'execution_time': random.uniform(0.5, 3.0),
                'success': random.random() > 0.2,  # 80%成功率
                'timestamp': time.time() + i * 0.1
            }
            interaction_logs.append(interaction_log)
        
        print(f"   🎭 模拟训练: {num_interactions} 次交互, 准确率: {accuracy:.4f}")
        
        return accuracy, interaction_logs
    
    def _validate_episode_with_mmlu_data(self, 
                                        executable_system, 
                                        validation_data: List[Dict], 
                                        episode: int) -> float:
        """
        使用MMLU验证数据验证一个episode
        
        Args:
            executable_system: 可执行的多智能体系统
            validation_data: 验证数据批次
            episode: 当前episode编号
            
        Returns:
            验证准确率
        """
        correct_answers = 0
        total_questions = 0
        
        # 随机选择一些批次进行验证
        selected_batches = random.sample(validation_data, min(2, len(validation_data)))
        
        for batch in selected_batches:
            batch_questions = batch['questions']
            
            # 随机选择批次中的一些问题
            selected_questions = random.sample(batch_questions, min(3, len(batch_questions)))
            
            for question_data in selected_questions:
                # 格式化问题
                formatted_question = self.data_processor.format_question_for_agents(question_data)
                
                try:
                    # 执行多智能体系统
                    result, _ = executable_system.start(
                        formatted_question['task'], 
                        max_transitions=5
                    )
                    
                    # 评估结果
                    evaluation = self.data_processor.evaluate_agent_response(
                        result, 
                        formatted_question['correct_answer']
                    )
                    
                    if evaluation['is_correct']:
                        correct_answers += 1
                    total_questions += 1
                    
                except Exception as e:
                    total_questions += 1
                    continue
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        print(f"   🔍 验证结果: {correct_answers}/{total_questions} = {accuracy:.4f}")
        
        return accuracy
    
    def _save_intermediate_results(self, 
                                  training_metrics: Dict[str, Any], 
                                  episode: int, 
                                  domain: str = None):
        """
        保存中间训练结果
        
        Args:
            training_metrics: 训练指标
            episode: 当前episode
            domain: 训练领域
        """
        intermediate_results = {
            'episode': episode,
            'domain': domain,
            'timestamp': time.time(),
            'current_training_accuracy': training_metrics['episode_accuracies'][-1] if training_metrics['episode_accuracies'] else 0.0,
            'current_validation_accuracy': training_metrics['validation_accuracies'][-1] if training_metrics['validation_accuracies'] else 0.0,
            'best_validation_accuracy': training_metrics['best_validation_accuracy'],
            'best_episode': training_metrics['best_episode'],
            'recent_combined_losses': training_metrics['combined_losses'][-10:],  # 最近10个episode的组合损失
            'recent_policy_gradient_losses': training_metrics['policy_gradient_losses'][-10:],  # 策略梯度损失
            'recent_reconstruction_losses': training_metrics['reconstruction_losses'][-10:],  # MSE重构损失
            'recent_interactions': len(training_metrics['agent_interaction_logs'][-50:]),  # 最近50个交互
            'loss_weights': training_metrics['loss_weights']  # 损失权重配置
        }
        
        # 保存到文件
        domain_suffix = f"_{domain}" if domain else ""
        intermediate_path = self.output_dir / f"intermediate_results{domain_suffix}_episode_{episode}.json"
        
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(intermediate_results, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 中间结果已保存到 {intermediate_path}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FSM Multi-Agent System Training")
    
    parser.add_argument("--mode", type=str, choices=['mmlu', 'single_task', 'demo'], 
                       default='demo', help="Training mode")
    parser.add_argument("--task", type=str, default=None,
                       help="Task description for single_task mode")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data (e.g., MMLU)")
    parser.add_argument("--output_dir", type=str, default="./fsm_mas_outputs",
                       help="Output directory for results")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    # 训练参数
    parser.add_argument("--training_episodes", type=int, default=100,
                       help="Number of training episodes")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    
    # 组合损失权重参数
    parser.add_argument("--policy_gradient_weight", type=float, default=1.0,
                       help="Weight for policy gradient loss (alpha)")
    parser.add_argument("--reconstruction_weight", type=float, default=0.1,
                       help="Weight for MSE reconstruction loss (beta)")
    
    # 模型参数
    parser.add_argument("--memory_dim", type=int, default=128,
                       help="Memory dimension for TGN")
    parser.add_argument("--time_dim", type=int, default=32,
                       help="Time encoding dimension")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 构建配置
    config = {
        'use_neural_learning': True,
        'training_episodes': args.training_episodes,
        'learning_rate': args.learning_rate,
        'policy_gradient_weight': args.policy_gradient_weight,  # 新增：策略梯度损失权重
        'reconstruction_weight': args.reconstruction_weight,    # 新增：MSE重构损失权重
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
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
    
    # 如果提供了配置文件，加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    print("🌟 FSM多智能体系统训练开始")
    print(f"🎯 训练模式: {args.mode}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"⚙️ 配置: {json.dumps(config, indent=2)}")
    
    # 创建训练器
    trainer = FSMMultiAgentSystemTrainer(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    try:
        if args.mode == 'mmlu':
            if not args.data_path:
                raise ValueError("MMLU mode requires --data_path")
            results = trainer.train_on_mmlu_domains()
            print("\n🎉 MMLU训练完成！")
            
        elif args.mode == 'single_task':
            if not args.task:
                raise ValueError("Single task mode requires --task")
            results = trainer.train_single_task(args.task)
            print(f"\n🎉 单任务训练完成！")
            
        elif args.mode == 'demo':
            results = trainer.demonstrate_fsm_mas_capabilities()
            print("\n🎉 演示完成！")
        
        # 打印训练摘要
        summary = trainer.get_training_summary()
        print("\n" + "="*80)
        print("📊 训练摘要:")
        print("="*80)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
